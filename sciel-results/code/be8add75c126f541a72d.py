# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from scipy.integrate import fixed_quad

def _build_basis():
    """
    Build the 70x130 quadrature grid and the 21 basis function values Y_m(x,y,z)
    on the unit sphere. Returns:
        theta_pts, phi_pts: grid points
        Y: shape (21, 70, 130) with Y[i] = Y_{m=i-10}
        weights: shape (70, 130) for the surface measure dOmega = sin(theta)dtheta dphi
    """
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # x_gl = cos(theta), so sin(theta) dtheta = -dx (we integrate over x from -1 to 1)
    # Actually, the surface measure is dOmega = sin(theta) dtheta dphi.
    # Let u = cos(theta), then du = -sin(theta) dtheta.
    # Integral over sphere = integral_{phi=0}^{2pi} integral_{u=-1}^{1} f(u, phi) du dphi.
    # So the weight for u is just w_gl, and for phi it is 2pi/n_phi.
    # Total mass should be 4*pi. Sum(w_gl) = 2. Sum(phi_weights) = 2pi. Product = 4pi. Correct.

    phi_pts = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # u = cos(theta)
    u = x_gl
    theta = np.arccos(u)
    sin_theta = np.sqrt(np.maximum(0, 1 - u**2))

    # Weights for the quadrature
    w_phi = np.full(n_phi, 2 * np.pi / n_phi)
    weights = np.outer(w_gl, w_phi)  # shape (70, 130)

    # Build basis functions
    # Y_0 = N_0 P_10^0(u)
    # Y_m = sqrt(2) N_m P_10^m(u) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(u) sin(|m| phi) for m < 0
    # N_m = sqrt( (2*10+1)/2 * (10-m)!/(10+m)! ) / sqrt(4*pi) ?
    # The task says: N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Let's check normalization.
    # Standard orthonormal spherical harmonics Y_l^m have:
    # Y_l^m = N_l^m P_l^m(cos theta) e^{i m phi}
    # N_l^m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    # The real harmonics are:
    # Y_l^0 = N_l^0 P_l^0
    # Y_l^{m} = sqrt(2) N_l^m P_l^m cos(m phi) for m > 0
    # Y_l^{-m} = sqrt(2) N_l^m P_l^m sin(m phi) for m > 0
    # So the normalization is consistent with the task description.

    Y = np.zeros((21, n_theta, n_phi))
    for m in range(-10, 11):
        abs_m = abs(m)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - abs_m) / factorial(10 + abs_m))
        
        # Compute associated Legendre polynomial P_10^{abs_m}(u)
        # We can use scipy.special.lpmv
        from scipy.special import lpmv
        P_lm = lpmv(abs_m, 10, u)  # shape (70,)
        
        if m == 0:
            Y[10] = N_m * P_lm[:, np.newaxis]
        elif m > 0:
            Y[10 + m] = np.sqrt(2.0) * N_m * P_lm[:, np.newaxis] * np.cos(m * phi_pts)[np.newaxis, :]
        else:
            Y[10 + m] = np.sqrt(2.0) * N_m * P_lm[:, np.newaxis] * np.sin(abs_m * phi_pts)[np.newaxis, :]
    
    return u, phi_pts, Y, weights

def _compute_R(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2
    coeffs: shape (21,)
    Y: shape (21, 70, 130)
    weights: shape (70, 130)
    """
    f = np.tensordot(coeffs, Y, axes=([0], [0]))  # shape (70, 130)
    
    # ||f||_2^2 = sum f^2 * weights
    norm2_sq = np.sum(f**2 * weights)
    # ||f||_4^4 = sum f^4 * weights
    norm4_4 = np.sum(f**4 * weights)
    
    if norm2_sq < 1e-15:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def _compute_grad_R(coeffs, Y, weights):
    """
    Compute the gradient of R with respect to coeffs.
    R = (sum f^4 w)^{1/4} / (sum f^2 w)^{1/2}
    Let A = sum f^4 w, B = sum f^2 w.
    R = A^{1/4} B^{-1/2}
    dR/dc_i = (1/4) A^{-3/4} (dA/dc_i) B^{-1/2} + A^{1/4} (-1/2) B^{-3/2} (dB/dc_i)
    dA/dc_i = sum 4 f^3 (df/dc_i) w = 4 sum f^3 Y_i w
    dB/dc_i = sum 2 f (df/dc_i) w = 2 sum f Y_i w
    So dR/dc_i = A^{-3/4} B^{-1/2} sum f^3 Y_i w - (1/2) A^{1/4} B^{-3/2} sum f Y_i w
    """
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    A = np.sum(f**4 * weights)
    B = np.sum(f**2 * weights)
    
    if B < 1e-15:
        return np.zeros(21)
    
    # Precompute f^3 * w and f * w
    f3w = f**3 * weights
    fw = f * weights
    
    # sum f^3 Y_i w for each i
    grad_A = 4.0 * np.tensordot(f3w, Y, axes=([0, 1], [1, 2]))  # shape (21,)
    grad_B = 2.0 * np.tensordot(fw, Y, axes=([0, 1], [1, 2]))   # shape (21,)
    
    term1 = (A ** (-0.75)) * (B ** (-0.5)) * grad_A
    term2 = -0.5 * (A ** 0.25) * (B ** (-1.5)) * grad_B
    
    return term1 + term2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Build basis
    u, phi_pts, Y, weights = _build_basis()
    
    # Initial guess: all ones
    best_coeffs = np.ones(21, dtype=np.float64)
    best_R = _compute_R(best_coeffs, Y, weights)
    
    # Differential Evolution
    from scipy.optimize import differential_evolution
    
    def objective(coeffs):
        return _compute_R(coeffs, Y, weights)
    
    # Bounds for coefficients: [-10, 10]
    bounds = [(-10.0, 10.0)] * 21
    
    # Use DE with a reasonable population size
    # We need to be careful with the time limit.
    # DE can be slow. Let's use a smaller population and fewer iterations,
    # or use a custom loop to check time.
    
    # Let's try a manual DE loop to control time.
    # Or use scipy's DE with maxiter and a callback to stop early.
    
    # Actually, let's use scipy's differential_evolution with a tight maxiter
    # and see if it finishes in time. If not, we fall back to the best so far.
    
    # To be safe, let's run DE in a way that we can interrupt.
    # But scipy's DE doesn't have a time-based stop, only iteration-based.
    # We can estimate the time per iteration and set maxiter accordingly.
    
    # Let's do a few evaluations to estimate time.
    t0 = time.time()
    for _ in range(10):
        objective(np.random.uniform(-1, 1, 21))
    t1 = time.time()
    avg_time_per_eval = (t1 - t0) / 10.0
    
    # DE with popsize=15, maxiter=1000 would be ~15000 evaluations.
    # If avg_time_per_eval is 0.01s, that's 150s.
    # If it's 0.1s, that's 1500s, which is too long.
    
    # Let's set maxiter based on remaining time.
    remaining_time = deadline - time.time() - 30.0  # 30s safety margin
    if remaining_time <= 0:
        return best_coeffs
    
    # Estimate max iterations for DE
    # DE population size is typically 15 * len(bounds) = 315. That's too big.
    # Let's use popsize=10, which is 210 individuals per generation.
    popsize = 10
    individuals_per_gen = popsize * 21
    est_time_per_gen = individuals_per_gen * avg_time_per_eval
    max_iter = int(remaining_time / est_time_per_gen) if est_time_per_gen > 0 else 100
    max_iter = max(10, min(max_iter, 500))  # Cap at 500 to be safe
    
    # Run DE
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=popsize,
            seed=42,
            tol=1e-8,
            polish=True  # Polish with L-BFGS-B at the end
        )
        if result.success and result.fun < best_R:
            best_coeffs = result.x
            best_R = result.fun
    except Exception:
        pass
    
    # If we have time left, do a local polish from the best DE result
    remaining_time = deadline - time.time() - 10.0
    if remaining_time > 5.0:
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                jac=lambda c: _compute_grad_R(c, Y, weights),
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_R:
                best_coeffs = res.x
                best_R = res.fun
        except Exception:
            pass
    
    # Final check
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
