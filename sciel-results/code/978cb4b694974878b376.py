"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_i h[i] * (1 - h[(i-k) % N])."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    # For each shift k, overlap = dot(h, roll(one_minus_h, k))
    # We want max over k
    # Use FFT for efficiency
    # overlap_k = sum_i h[i] * (1 - h[(i-k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i-k)%N]
    # = N*mean(h) - autocorrelation of h at lag k
    # Since sum(h) = N/2, N*mean(h) = N/2
    # So overlap_k = N/2 - autocorr_h(k)
    # We want to minimize max_k overlap_k = N/2 - min_k autocorr_h(k)
    # So minimizing max overlap = maximizing min autocorrelation
    # autocorr_h(k) = sum_i h[i] * h[(i-k) % N]
    
    # Compute autocorrelation via FFT
    H = np.fft.fft(h)
    acf = np.real(np.fft.ifft(H * np.conj(H)))
    # acf[k] = sum_i h[i] * h[(i-k) % N] for k=0..N-1
    # But ifft gives circular autocorrelation
    min_acf = np.min(acf)
    max_overlap = N / 2.0 - min_acf
    return max_overlap

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Parametric model: h_i = min(ceiling, A * sin(2*pi*f*i/N + phi) + offset)
    # We have 4 parameters: ceiling, A, f, phi
    # offset is determined to make mean(h) = 0.5
    
    def param_to_h(params):
        ceiling, A, f, phi = params
        i = np.arange(N)
        wave = A * np.sin(2 * np.pi * f * i / N + phi)
        # Find offset so that mean(min(ceiling, wave + offset)) = 0.5
        # Use bisection on offset
        def mean_h(off):
            h = np.minimum(ceiling, wave + off)
            return np.mean(h)
        
        # Bisection to find offset
        lo, hi = -A - 1.0, A + 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if mean_h(mid) < 0.5:
                lo = mid
            else:
                hi = mid
        offset = (lo + hi) / 2.0
        h = np.minimum(ceiling, wave + offset)
        # Clip to [0, 1] for safety
        h = np.clip(h, 0.0, 1.0)
        # Adjust to ensure exact sum = N/2
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            # Re-adjust if clipping changed sum
            diff2 = N / 2.0 - np.sum(h)
            h = h + diff2 / N
        return h
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        h = param_to_h(params)
        return compute_max_overlap(h)
    
    # Initial guesses
    best_score = 1.0  # constant 0.5 baseline
    best_params = None
    
    # Try several initial points
    initial_guesses = [
        [0.5, 0.3, 1.0, 0.0],
        [0.5, 0.3, 2.0, 0.0],
        [0.5, 0.3, 0.5, 0.0],
        [0.5, 0.5, 1.0, np.pi/2],
        [0.5, 0.5, 2.0, np.pi/2],
        [0.7, 0.4, 1.0, 0.0],
        [0.3, 0.4, 1.0, 0.0],
        [0.5, 0.6, 3.0, 0.0],
        [0.5, 0.6, 0.5, np.pi/4],
        [0.5, 0.4, 4.0, 0.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        h = param_to_h(best_params)
    else:
        h = np.full(N, 0.5)
    
    # Final polish: try local refinement with SLSQP on the h variables directly
    if time.time() < deadline - 20:
        def h_objective(h_flat):
            h = h_flat
            return compute_max_overlap(h)
        
        def h_constraint(h_flat):
            return np.sum(h_flat) - N / 2.0
        
        constraints = [{'type': 'eq', 'fun': h_constraint}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            result = minimize(h_objective, h, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 500, 'ftol': 1e-10})
            if result.fun < best_score:
                h = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Ensure exact sum
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Final correction
        diff2 = N / 2.0 - np.sum(h)
        h = h + diff2 / N
    
    return h
# EVOLVE-BLOCK-END
