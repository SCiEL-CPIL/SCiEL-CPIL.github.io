"""
Combined strategy: exploit scale invariance and use a structured low-dimensional
parameterization (product of two complex polynomials) to find shapes that
maximize the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf).

Key structural insight: represent heights as |P(w^k) * Q(w^k)|^2 where P, Q
are low-degree complex polynomials. This enforces smoothness and naturally
minimizes spectral peaks in the autoconvolution, while the optimization
concentrates mass where it helps most.

The refinement machinery uses L-BFGS-B with a smooth objective on the
polynomial coefficients, followed by a local refinement on the full 50-dim
space.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # Safety margin below 150s limit

    def compute_ratio(h):
        """Compute ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step function."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-30 or sum_sq < 1e-60:
            return 0.0
        ac = np.correlate(h, h, mode='full')
        max_ac = np.max(ac)
        if max_ac < 1e-60:
            return 0.0
        return sum_sq / (d * sum_h * max_ac)

    def smooth_negative_ratio(h):
        """Smooth negative ratio for optimization (minimize this)."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-30 or sum_sq < 1e-60:
            return 1e10
        ac = np.correlate(h, h, mode='full')
        ac_clipped = np.clip(ac, 1e-30, None)
        log_ac = np.log(ac_clipped)
        max_log = np.max(log_ac)
        # Smooth max via log-sum-exp with temperature
        temp = 50.0
        smooth_max = max_log + np.log(np.sum(np.exp((log_ac - max_log) / temp))) / temp
        peak = np.exp(smooth_max)
        if peak < 1e-30:
            return 1e10
        ratio = sum_sq / (d * sum_h * peak)
        return -ratio

    def poly_to_heights(coeffs_p, coeffs_q, degree_p, degree_q):
        """
        Convert complex polynomial coefficients to 50 nonnegative heights.
        h_k = |P(w^k) * Q(w^k)|^2 where w = exp(2*pi*i*k/N)
        coeffs_p: array of length degree_p+1 (complex)
        coeffs_q: array of length degree_q+1 (complex)
        """
        # Evaluate on N-th roots of unity
        k = np.arange(N)
        w = np.exp(2j * np.pi * k / N)
        
        # Evaluate P(w^k) using Horner's method
        p_vals = np.zeros(N, dtype=complex)
        for c in coeffs_p:
            p_vals = p_vals * w + c
        
        # Evaluate Q(w^k)
        q_vals = np.zeros(N, dtype=complex)
        for c in coeffs_q:
            q_vals = q_vals * w + c
        
        # Heights = |P*Q|^2
        heights = np.abs(p_vals * q_vals)**2
        
        # Ensure nonnegativity and handle numerical issues
        heights = np.maximum(0, heights.real)
        
        # If all zero, return a tiny spike
        if np.sum(heights) < 1e-30:
            h = np.zeros(N)
            h[N // 2] = 1.0
            return h
        
        return heights

    def poly_objective(params, degree_p, degree_q):
        """Objective for polynomial coefficient optimization."""
        # Split params into real/imag parts of P and Q coefficients
        n_p = degree_p + 1
        n_q = degree_q + 1
        n_p_complex = 2 * n_p  # real and imag parts
        n_q_complex = 2 * n_q
        
        p_re = params[0:n_p]
        p_im = params[n_p:2*n_p]
        q_re = params[2*n_p:2*n_p + n_q]
        q_im = params[2*n_p + n_q:2*n_p + 2*n_q]
        
        coeffs_p = p_re + 1j * p_im
        coeffs_q = q_re + 1j * q_im
        
        heights = poly_to_heights(coeffs_p, coeffs_q, degree_p, degree_q)
        return smooth_negative_ratio(heights)

    # Phase 1: Quick evaluation of known good shapes to establish baseline
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Try single spike
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    s = compute_ratio(h_spike)
    if s > best_score:
        best_score = s
        best_h = h_spike.copy()
    
    # Try a few parametric shapes
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        x = np.arange(N) - N / 2
        h = np.exp(-x**2 / (2 * sigma**2))
        s = compute_ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 2: Optimize polynomial parameterization
    rng = np.random.default_rng(42)
    
    # Try different polynomial degrees
    for deg_p in [2, 3, 4]:
        for deg_q in [2, 3, 4]:
            if time.time() > deadline - 20:
                break
            
            n_params = 2 * (deg_p + 1) + 2 * (deg_q + 1)
            bounds = [(-10, 10)] * n_params
            
            # Multiple random initializations
            for init in range(3):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                p_re = rng.normal(0, 1, deg_p + 1)
                p_im = rng.normal(0, 1, deg_p + 1)
                q_re = rng.normal(0, 1, deg_q + 1)
                q_im = rng.normal(0, 1, deg_q + 1)
                
                # Ensure first coefficient is real and positive (phase invariance)
                p_re[0] = abs(p_re[0]) + 0.1
                p_im[0] = 0.0
                
                params0 = np.concatenate([p_re, p_im, q_re, q_im])
                
                try:
                    res = minimize(
                        poly_objective, params0,
                        args=(deg_p, deg_q),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    # Extract heights
                    n_p = deg_p + 1
                    n_q = deg_q + 1
                    p_re_opt = res.x[0:n_p]
                    p_im_opt = res.x[n_p:2*n_p]
                    q_re_opt = res.x[2*n_p:2*n_p + n_q]
                    q_im_opt = res.x[2*n_p + n_q:2*n_p + 2*n_q]
                    
                    coeffs_p = p_re_opt + 1j * p_im_opt
                    coeffs_q = q_re_opt + 1j * q_im_opt
                    
                    h_poly = poly_to_heights(coeffs_p, coeffs_q, deg_p, deg_q)
                    s = compute_ratio(h_poly)
                    if s > best_score:
                        best_score = s
                        best_h = h_poly.copy()
                except Exception:
                    pass
        
        if time.time() > deadline - 15:
            break
    
    # Phase 3: Local refinement on full 50-dim space
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        bounds = [(0, None)] * N
        
        try:
            res = minimize(
                smooth_negative_ratio, h_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.clip(res.x, 0, None)
            s = compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Final cleanup
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-320:
        best_h = np.zeros(N)
        best_h[N // 2] = 1e-300
    
    return list(best_h)

# EVOLVE-BLOCK-END
