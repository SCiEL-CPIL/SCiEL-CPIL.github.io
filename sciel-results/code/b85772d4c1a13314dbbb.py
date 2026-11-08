"""Improved program for Erdos minimum overlap task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 110.0  # seconds, with 10s safety margin before 120s limit

def _compute_max_overlap(h):
    """Compute the maximum of sum_{j} h[j] * (1 - h[(j+k)%N]) * (1/N) over all shifts k."""
    n = len(h)
    bin_width = 1.0 / n
    max_ov = 0.0
    # Use FFT for efficient cross-correlation
    # overlap(k) = sum_j h[j] * (1-h[(j+k)%n]) = sum_j h[j] - sum_j h[j]*h[(j+k)%n]
    # = 1 - sum_j h[j]*h[(j+k)%n]  (since sum h[j] = n/2 * bin_width = 1)
    # Wait: sum_j h[j] * bin_width = 1, so sum_j h[j] = n
    # overlap(k) = sum_j h[j]*(1-h[(j+k)%n]) * bin_width
    #            = (sum_j h[j] - sum_j h[j]*h[(j+k)%n]) * bin_width
    #            = (n - sum_j h[j]*h[(j+k)%n]) * bin_width
    #            = (n - autocorr[k]) * (1/n)
    # where autocorr[k] = sum_j h[j]*h[(j+k)%n]
    
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h)
    auto = np.fft.ifft(H * np.conj(H)).real
    # auto[k] = sum_j h[j]*h[(j-k)%n], so we need to index carefully
    # overlap for shift k: sum_j h[j]*h[(j+k)%n] = auto[n-k] if k>0, or auto[0] if k==0
    # Actually circular correlation: c[k] = sum_j h[j]*h[(j+k)%n] = ifft(fft(h)*conj(fft(h))) gives c[-k]
    # Let's just be safe and compute directly for small N or verify
    
    # For N=100, direct computation is fast enough
    for k in range(n):
        ov = np.dot(h, np.roll(h, -k)) * bin_width  # sum_j h[j]*h[(j+k)%n]
        overlap_val = (n - ov) * bin_width
        if overlap_val > max_ov:
            max_ov = overlap_val
    return max_ov

def _build_h_from_params(params):
    """Map parameters to valid step heights in [0,1] with sum = N/2."""
    h = 1.0 / (1.0 + np.exp(-params))
    # Normalize to sum = N/2
    target_sum = N / 2.0
    h = h * (target_sum / h.sum())
    # Clip to [0, 1] and renormalize
    h = np.clip(h, 0.0, 1.0)
    # After clipping, renormalize while keeping in [0,1]
    for _ in range(10):
        s = h.sum()
        if s > 0:
            h = h * (target_sum / s)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def _objective(params):
    h = _build_h_from_params(params)
    return _compute_max_overlap(h)

def construct():
    start_time = time.time()
    
    # Initial: uniform 0.5
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)  # should be 0.25
    
    # Try Nelder-Mead optimization
    x0 = np.zeros(N)  # sigmoid(0) = 0.5
    try:
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        elapsed = time.time() - start_time
        if elapsed < DEADLINE:
            h_opt = _build_h_from_params(res.x)
            score_opt = _compute_max_overlap(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    elapsed = time.time() - start_time
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        x0 = rng.normal(0, 1.0, N)
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            h_opt = _build_h_from_params(res.x)
            score_opt = _compute_max_overlap(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
        except Exception:
            pass
    
    return best_h
