import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        corr = compute_corr(h)
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def h_from_params(params):
        """
        Parameterize h as a low-dimensional function:
        h[i] = 0.5 + sum_{m=1}^{M} (a_m * cos(2*pi*m*i/N + phi_m))
        """
        M = len(params) // 2
        h = np.full(N, 0.5)
        for m in range(M):
            a = params[2*m]
            phi = params[2*m + 1]
            h += a * np.cos(2 * np.pi * m * np.arange(N) / N + phi)
        return h

    def objective(params):
        """Objective function for differential evolution."""
        if time.time() > deadline - 5.0:
            return 1e6
        h = h_from_params(params)
        h = np.clip(h, 0, 1)
        h = project_to_polytope(h, N / 2.0)
        return max_overlap(h)

    # Define bounds for parameters
    M = 3
    bounds = [(-0.5, 0.5)] * M + [(0, 2*np.pi)] * M
    
    # Run differential evolution
    de_result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=False,
        workers=1
    )
    
    best_h = h_from_params(de_result.x)
    best_h = np.clip(best_h, 0, 1)
    best_h = project_to_polytope(best_h, N / 2.0)
    best_score = max_overlap(best_h)
    
    # Also try a 2-block structure as a baseline
    best_2block = None
    best_2block_score = float('inf')
    for k in range(1, N):
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        if a_min >= a_max:
            continue
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, 50)
        for a_val in a_vals:
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a_val
            h[k:] = b_val
            score = max_overlap(h)
            if score < best_2block_score:
                best_2block_score = score
                best_2block = h.copy()
    
    if best_2block is not None and best_2block_score < best_score:
        best_h = best_2block
        best_score = best_2block_score
    
    # Final polish with local optimization if time remains
    if time.time() < deadline - 10.0:
        def local_objective(x):
            h = np.clip(x, 0, 1)
            sum_val = h.sum()
            sum_penalty = 100.0 * (sum_val - N / 2.0) ** 2
            overlap = max_overlap(h)
            return overlap + sum_penalty
        
        def local_gradient(x):
            h = np.clip(x, 0, 1)
            sum_val = h.sum()
            sum_grad = 200.0 * (sum_val - N / 2.0)
            
            corr = compute_corr(h)
            k_min = np.argmin(corr)
            
            idx = np.arange(N)
            corr_grad = -(h[(idx + k_min) % N] + h[(idx - k_min) % N])
            
            return corr_grad + sum_grad
        
        try:
            res = minimize(
                local_objective,
                best_h,
                jac=local_gradient,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_opt = project_to_polytope(res.x, N / 2.0)
            score = max_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        return np.full(N, 0.5)
        
    # Final polish: ensure exact constraints
    best_h = project_to_polytope(best_h, N / 2.0)
    
    # Safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
