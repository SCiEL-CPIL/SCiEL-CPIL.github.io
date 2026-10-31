"""Optimized program for the first autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval width: total range is 1/2, so each interval is 1/(2*N)
dx = 1.0 / (2.0 * N)


def _compute_score(h):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h[i] on intervals of width dx:
    The discrete autoconvolution conv[k] = sum_i h[i]*h[k-i]
    Normalized max = max(conv) / N
    """
    conv = np.fft.irfft(np.fft.rfft(h, n=2 * N - 1) ** 2, n=2 * N - 1)
    return np.max(conv) / N


def _smooth_max(conv, beta):
    """Compute smooth max of conv using log-sum-exp trick."""
    m = np.max(conv)
    if m < 1e-15:
        return 0.0, np.zeros_like(conv)
    exp_terms = np.exp(beta * (conv - m) / N)
    log_sum = np.log(np.sum(exp_terms))
    smooth = (m + log_sum / beta) / N
    # Gradient with respect to conv: (1/N) * exp_terms / sum(exp_terms)
    grad_conv = exp_terms / (N * np.sum(exp_terms))
    return smooth, grad_conv


def _parametric_to_step(phases):
    """Convert phase parameters to a nonnegative step function.
    
    We represent h as |z|^2 where z is a complex sequence whose DFT 
    has constant amplitude and variable phase. This ensures h >= 0.
    
    Specifically: Z = DFT(z), |Z_k| = constant, phase(Z_k) = phases[k]
    Then z = IDFT(Z), and h = |z|^2 (real, nonnegative).
    """
    # Create constant-amplitude DFT with given phases
    # Use N-point DFT
    amp = np.sqrt(N)  # Constant amplitude
    Z = amp * np.exp(1j * phases)
    # Inverse DFT to get z
    z = np.fft.ifft(Z)
    # h = |z|^2
    h = np.abs(z) ** 2
    # Normalize so that sum(h) = N
    s = np.sum(h)
    if s > 1e-10:
        h = h / s * N
    return h


def _parametric_objective(phases, beta):
    """Compute the smooth surrogate objective for given phases."""
    h = _parametric_to_step(phases)
    conv = np.fft.irfft(np.fft.rfft(h, n=2 * N - 1) ** 2, n=2 * N - 1)
    smooth, grad_conv = _smooth_max(conv, beta)
    return smooth, grad_conv


def _local_refinement(h, deadline, beta_start=20.0, beta_end=100.0, n_anneal=5):
    """Annealed L-BFGS-B refinement with increasing beta for sharper max approximation."""
    best_h = h.copy()
    best_score = _compute_score(best_h)
    
    # Anneal beta from low to high
    for i in range(n_anneal):
        if time.time() > deadline - 10:
            break
        
        beta = beta_start + (beta_end - beta_start) * i / max(1, n_anneal - 1)
        
        def obj(params):
            h_local = np.maximum(0.0, params)
            s = np.sum(h_local)
            if s < 1e-10:
                return 1e10
            h_local = h_local / s * N
            conv = np.fft.irfft(np.fft.rfft(h_local, n=2 * N - 1) ** 2, n=2 * N - 1)
            sm, _ = _smooth_max(conv, beta)
            return sm
        
        try:
            result = minimize(
                obj,
                best_h.copy(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                bounds=[(0, None)] * N
            )
            
            h_opt = np.maximum(0.0, result.x)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt / s * N
                score = _compute_score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_score


def construct():
    """Construct a nonnegative step function with N intervals to minimize 
    the normalized maximum of its autoconvolution."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Start with a reasonable initial guess
    x = np.linspace(0, 1, N)
    
    # Generate multiple initializations
    initializations = []
    
    # Cosine bump
    h_cos = np.maximum(0.0, np.cos(np.pi * (x - 0.5)))
    h_cos = h_cos / np.sum(h_cos) * N
    initializations.append(h_cos)
    
    # Constant
    initializations.append(np.ones(N))
    
    # Quadratic bump
    h_quad = np.maximum(0.0, 1.0 - 4 * (x - 0.5)**2)
    h_quad = h_quad / np.sum(h_quad) * N
    initializations.append(h_quad)
    
    # Triangular
    h_tri = np.abs(1.0 - 2.0 * x)
    h_tri = h_tri / np.sum(h_tri) * N
    initializations.append(h_tri)
    
    # Two bumps
    h_two = np.maximum(0.0, np.cos(2 * np.pi * (x - 0.25))) + np.maximum(0.0, np.cos(2 * np.pi * (x - 0.75)))
    h_two = h_two / np.sum(h_two) * N
    initializations.append(h_two)
    
    # Random phases -> spectral flatness approach
    rng = np.random.default_rng(42)
    for _ in range(3):
        phases = rng.uniform(-np.pi, np.pi, N)
        h_rand = _parametric_to_step(phases)
        initializations.append(h_rand)
    
    # Track best
    best_h = initializations[0].copy()
    best_score = _compute_score(best_h)
    
    # Optimize each initialization
    for h_init in initializations:
        if time.time() > deadline - 15:
            break
        
        h_init = np.maximum(0.0, h_init)
        s = np.sum(h_init)
        if s < 1e-10:
            continue
        h_init = h_init / s * N
        
        # Local refinement with annealing
        h_opt, score = _local_refinement(h_init, deadline)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final polish with high beta
    if time.time() < deadline - 5:
        best_h, best_score = _local_refinement(
            best_h, deadline, 
            beta_start=50.0, beta_end=200.0, n_anneal=3
        )
    
    # Ensure nonnegativity
    best_h = np.maximum(0.0, best_h)
    return list(best_h)


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"Min: {min(h)}, Max: {max(h)}")
    print(f"Score: {_compute_score(np.array(h))}")
