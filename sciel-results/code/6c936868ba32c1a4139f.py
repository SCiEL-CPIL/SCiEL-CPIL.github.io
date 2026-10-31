"""Low-rank spline representation with differential evolution + gradient refinement."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Knot positions for piecewise-linear spline
    knots = np.linspace(-0.25, 0.25, N)
    
    # Low-rank basis: use a few smooth basis functions
    # Rank-3: flat-top, cosine taper, and a shape function
    def basis_functions(x):
        """Return (N, rank) basis matrix for low-rank representation."""
        rank = 3
        B = np.zeros((N, rank))
        # Component 1: flat-top (constant)
        B[:, 0] = 1.0
        # Component 2: cosine taper (allows edge shaping)
        B[:, 1] = np.cos(np.pi * x / 0.25)
        # Component 3: quadratic shape (allows parabolic variations)
        B[:, 2] = 1.0 - 4.0 * x**2
        return B
    
    B = basis_functions(knots)
    rank = B.shape[1]
    
    def heights_from_coeffs(coeffs):
        """Convert low-rank coefficients to nonnegative step heights."""
        h = B @ coeffs
        return np.maximum(h, 0.0)
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(conv) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total < 1e-15:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)
    
    def objective(coeffs):
        """Objective for differential evolution: normalized max of autoconvolution."""
        h = heights_from_coeffs(coeffs)
        return compute_metric(h)
    
    def gradient_refine(h_init, max_iters=500):
        """Local refinement using projected gradient descent."""
        h = h_init.copy()
        best_h = h.copy()
        best_m = compute_metric(h)
        lr = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak_idx = np.argmax(acorr)
            peak = acorr[peak_idx]
            total = np.sum(h)
            
            if peak < 1e-15:
                break
            
            k = peak_idx
            if k == 0:
                grad_acorr_k = 2.0 * h
            else:
                grad_acorr_k = np.zeros(N)
                if k < N:
                    grad_acorr_k[k:] += h[:N-k]
                    grad_acorr_k[:N-k] += h[k:]
            
            metric_grad = grad_acorr_k / (total * total) - 2.0 * acorr[k] / (total * total * total)
            
            h_new = h - lr * metric_grad
            h_new = np.maximum(h_new, 0.0)
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            if m_new < best_m * 0.999:
                lr = min(lr * 1.2, 0.05)
            else:
                lr = max(lr * 0.95, 0.001)
            
            h = h_new
        
        return best_h, best_m
    
    # Differential evolution bounds for low-rank coefficients
    # Allow coefficients in range that can produce nonnegative heights
    bounds = [(-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    # Run differential evolution with multiple seeds
    best_h = None
    best_m = float('inf')
    
    for seed in [42, 123, 456]:
        if time.time() > deadline:
            break
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                init='sobol',
                polish=False
            )
            
            h_de = heights_from_coeffs(result.x)
            m_de = compute_metric(h_de)
            
            # Local refinement from DE result
            h_refined, m_refined = gradient_refine(h_de, max_iters=300)
            
            if m_refined < best_m:
                best_m = m_refined
                best_h = h_refined
        except Exception:
            continue
    
    # Also try some hand-crafted low-rank initializations
    if time.time() < deadline:
        test_coeffs = [
            [1.0, 0.0, 0.0],      # Uniform
            [0.8, 0.5, 0.0],      # Cosine taper
            [0.9, 0.3, 0.2],      # Mixed
            [1.0, 0.0, -0.5],     # Flat with dip
        ]
        
        for coeffs in test_coeffs:
            if time.time() > deadline:
                break
            
            h_init = heights_from_coeffs(coeffs)
            h_refined, m_refined = gradient_refine(h_init, max_iters=400)
            
            if m_refined < best_m:
                best_m = m_refined
                best_h = h_refined
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)

# EVOLVE-BLOCK-END
