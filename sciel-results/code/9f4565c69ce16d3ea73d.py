import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts."""
    # overlap at shift k = sum_i h[i] * (1 - h[(i+k) % N])
    # = sum h[i] - sum h[i]*h[(i+k)%N] = N/2 - autocorr(k)
    h = np.asarray(h)
    # Circular autocorrelation
    autocorr = np.fft.irfft(np.abs(np.fft.rfft(h, n=2*N))**2)[:N]
    # Actually, circular autocorrelation:
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT: autocorr = ifft(|fft(h)|^2) but circular
    H = np.fft.fft(h, n=2*N)
    autocorr = np.fft.ifft(H * np.conj(H)).real[:N]
    # Normalize: for circular autocorrelation, autocorr[k] = sum_i h[i]*h[(i+k)%N]
    # The sum of h is N/2, so autocorr[0] = sum h[i]^2
    overlaps = N / 2.0 - autocorr
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    # Ensure sum constraint approximately
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _objective(x):
    """Objective: maximize min autocorrelation = minimize max overlap.
    x is the parameter vector (h values)."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    # We want to minimize the max overlap
    # Use a smooth approximation: log-sum-exp
    # But for simplicity, use max with a small smoothing
    max_ov = np.max(overlaps)
    # Add small penalty for constraint violation
    sum_penalty = (np.sum(h) - N / 2.0) ** 2
    return max_ov + 0.01 * sum_penalty

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    # Adjust to meet sum constraint
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-10:
        # Simple: scale and clip, then adjust
        h = h * (target / current)
        h = np.clip(h, 0, 1)
        # Fine-tune: distribute remaining difference
        diff = target - np.sum(h)
        if diff > 0:
            # Need to increase some values
            mask = h < 1.0
            if np.sum(mask) > 0:
                per = diff / np.sum(mask)
                h[mask] = np.minimum(h[mask] + per, 1.0)
        else:
            mask = h > 0.0
            if np.sum(mask) > 0:
                per = -diff / np.sum(mask)
                h[mask] = np.maximum(h[mask] - per, 0.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with alternating pattern as initial guess
    h0 = np.zeros(N)
    h0[::2] = 0.6
    h0[1::2] = 0.4
    h0 = _project_to_feasible(h0)
    
    best_h = h0.copy()
    best_score = _max_overlap(best_h)
    
    # Try multiple random restarts with gradient-based optimization
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            x0 = h0.copy()
        else:
            # Random initial point
            x0 = rng.uniform(0.2, 0.8, size=N)
            x0 = _project_to_feasible(x0)
        
        # Use L-BFGS-B with bounds
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < best_score:
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_h = h_new
                best_score = score_new
        
        if time.time() > deadline - 10.0:
            break
    
    # Final polish: local refinement on the best solution
    if time.time() < deadline - 15.0:
        # Try a few more iterations with tighter tolerance
        result = minimize(
            _objective,
            best_h,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        h_polished = _project_to_feasible(result.x)
        score_polished = _max_overlap(h_polished)
        if score_polished < best_score:
            best_h = h_polished
            best_score = score_polished
    
    return best_h
