"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """
    Compute the maximum overlap of h with 1-h over all shifts.
    h: array of length N, values in [0, 1], sum(h) = N/2.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k)%N]) / N  (since integral is 1, width 2/N, sum h = N/2)
    Wait, let's re-verify the overlap definition.
    Integral of h(x) (1-h(x+k)) dx.
    h is step function. h(x) = h[i] for x in [2i/N, 2(i+1)/N).
    Integral = sum_i (2/N) h[i] (1 - h[(i+k)%N]).
    Sum_i h[i] = N/2.
    Integral = (2/N) [ sum_i h[i] - sum_i h[i] h[(i+k)%N] ]
             = (2/N) [ N/2 - autocorr[k] ]
             = 1 - (2/N) autocorr[k].
    To minimize max overlap, we need to maximize min autocorr.
    
    The inspirations used a different score function:
    shifted = np.zeros(n); shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))
    This computes max_i h[i](1-h[i+1]). This is NOT the same as the shift overlap defined above.
    The shift overlap involves h[i] and h[i+k].
    Let's stick to the correct definition from the task: "maximum overlap of h with shifted 1-h".
    Overlap(k) = integral h(x) (1-h(x+k)) dx.
    
    Let's implement the correct overlap calculation.
    """
    n = len(h)
    # autocorr[k] = sum_i h[i] h[(i+k)%n]
    # Using FFT for speed
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] corresponds to sum_i h[i] h[(i-k)%n].
    # Since h is real, autocorr is symmetric.
    # overlap[k] = 1 - (2/n) * autocorr[k]
    overlaps = 1.0 - (2.0 / n) * autocorr
    return np.max(overlaps)

def _construct_from_params(params, N):
    """
    Construct h from parameters: base 0.5 + sum of sinusoidal modes.
    params: [amp1, phase1, amp2, phase2, ...]
    """
    h = np.full(N, 0.5)
    n_modes = len(params) // 2
    x = np.arange(N)
    for i in range(n_modes):
        freq = i + 1
        amp = params[2*i]
        phase = params[2*i + 1]
        h += amp * np.sin(2 * np.pi * freq * x / N + phase)
    
    # Project to [0, 1] and sum = N/2
    # Simple iterative projection
    h = np.clip(h, 0.0, 1.0)
    for _ in range(20):
        s = h.sum()
        if s == 0:
            break
        # We want sum h = N/2
        # If sum is off, adjust. But clipping might break it.
        # A better projection:
        # 1. Clip to [0,1]
        # 2. If sum > N/2, subtract from largest elements
        # 3. If sum < N/2, add to smallest elements
        # For simplicity and speed in optimization, we can just scale and clip,
        # but that might not satisfy sum exactly.
        # Let's use a simple water-filling like approach or just accept small error?
        # The task says "sum is exactly len/2".
        # Let's do a proper projection.
        
        # Actually, the inspiration code did:
        # diff = N/2.0 - h.sum()
        # h += diff / N
        # h = np.clip(h, 0.0, 1.0)
        # This is a simple projection. Let's use it.
        diff = N / 2.0 - h.sum()
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Check convergence
        if abs(N/2.0 - h.sum()) < 1e-10:
            break
            
    return h

def construct():
    deadline = time.time() + 140.0
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try different numbers of modes
    n_modes_list = [2, 3, 4, 5, 6]
    
    for n_modes in n_modes_list:
        if time.time() > deadline - 15:
            break
        
        n_params = 2 * n_modes
        bounds = [(-0.5, 0.5)] * n_modes + [(0, 2*np.pi)] * n_modes
        
        # Multi-start
        n_starts = 5
        for start in range(n_starts):
            if time.time() > deadline - 5:
                break
            
            if start == 0:
                x0 = np.zeros(n_params)
                for i in range(n_modes):
                    x0[2*i] = 0.1 * (1 if i % 2 == 0 else -1)
                    x0[2*i+1] = 0.0
            else:
                np.random.seed(42 + start)
                x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            
            def objective(params):
                h = _construct_from_params(params, N)
                return _score(h)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                h = _construct_from_params(res.x, N)
                score = _score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            except Exception:
                pass
    
    return best_h
# EVOLVE-BLOCK-END
