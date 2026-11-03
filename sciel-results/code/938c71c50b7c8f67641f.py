"""Rewritten program for Erdos minimum overlap task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_max_overlap(h):
    """Compute the maximum overlap (min correlation) and the argmin shift."""
    corrs = _compute_all_corrs_fft(h)
    min_idx = np.argmin(corrs)
    return corrs[min_idx], min_idx + 1, corrs

def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    def f(theta):
        w = np.clip(v - theta, 0, 1)
        return np.sum(w) - target_sum
    
    lo = np.min(v) - 1.0
    hi = np.max(v) + 1.0
    
    if f(lo) < 0:
        lo -= 10.0
    if f(hi) > 0:
        hi += 10.0
        
    try:
        theta = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        w = np.clip(v, 0, 1)
        for _ in range(50):
            diff = target_sum - np.sum(w)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = w < 1.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            else:
                mask = w > 0.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
        return w

    w = np.clip(v - theta, 0, 1)
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-12:
        mask = (w > 1e-9) & (w < 1.0 - 1e-9)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
    return w

def _subgradient_ascent(h, target_sum, max_iters=500, step_scale=0.05):
    """Projected subgradient ascent on the true max-overlap objective."""
    n = len(h)
    h = _project(h.copy(), target_sum)
    current_val, k_star, _ = _compute_max_overlap(h)
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Compute subgradient: direction to decrease the max overlap
        # The subgradient of max(R_k) is the unit vector in the direction of the active set.
        # Here we use the subgradient of the specific worst shift R_{k_star}.
        # g[i] = d R_{k_star} / d h[i]
        # R_k = (2/n) * sum_{i=0}^{n-k-1} h[i] h[i+k]
        # d R_k / d h[i] = (2/n) * (h[i+k] if i < n-k else 0) + (2/n) * (h[i-k] if i >= k else 0)
        
        g = np.zeros(n)
        k = k_star
        if k < n:
            g[:n-k] += (2.0/n) * h[k:]
            g[k:] += (2.0/n) * h[:n-k]
        
        # We want to minimize the max overlap, so we ascend in the negative gradient direction?
        # Wait, the objective is to MINIMIZE the max overlap.
        # So we want to decrease R_k.
        # Gradient ascent on -f(x) is gradient descent on f(x).
        # Let's do gradient descent on f(x) = max R_k.
        # Step: h <- h - alpha * g
        
        # Adaptive step size
        alpha = step_scale / (1.0 + it * 0.01)
        
        h_trial = h - alpha * g
        h_trial = _project(h_trial, target_sum)
        
        trial_val, _, _ = _compute_max_overlap(h_trial)
        
        if trial_val < current_val:
            h = h_trial
            current_val = trial_val
        else:
            # Try smaller step
            alpha *= 0.5
            h_trial = h - alpha * g
            h_trial = _project(h_trial, target_sum)
            trial_val, _, _ = _compute_max_overlap(h_trial)
            if trial_val < current_val:
                h = h_trial
                current_val = trial_val
            else:
                break
                
    return h, current_val

def _coordinate_descent(h, target_sum, max_iters=100):
    """Coordinate descent polish: adjust each coordinate to minimize max overlap."""
    n = len(h)
    h = _project(h.copy(), target_sum)
    current_val, _, _ = _compute_max_overlap(h)
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for i in range(n):
            # Try to adjust h[i]
            # Keep sum constant: if we change h[i] by delta, we must adjust another coordinate.
            # Simpler: project after change.
            # But coordinate descent on simplex/box is tricky.
            # Let's try a small perturbation in h[i] and project.
            
            # Try increasing h[i]
            h_trial = h.copy()
            h_trial[i] += 0.01
            h_trial = _project(h_trial, target_sum)
            val_inc, _, _ = _compute_max_overlap(h_trial)
            
            # Try decreasing h[i]
            h_trial = h.copy()
            h_trial[i] -= 0.01
            h_trial = _project(h_trial, target_sum)
            val_dec, _, _ = _compute_max_overlap(h_trial)
            
            best_val = min(val_inc, val_dec, current_val)
            
            if best_val < current_val - 1e-8:
                if val_inc < val_dec:
                    h = h_trial # h_trial from inc
                else:
                    h_trial = h.copy()
                    h_trial[i] -= 0.01
                    h = _project(h_trial, target_sum)
                current_val = best_val
                improved = True
        
        if not improved:
            break
            
    return h, current_val

def _construct_diverse_initializations(n):
    """Generate diverse initializations."""
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            val = (n/2.0) / (n-k)
            if 0 <= val <= 1:
                restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, val)]))
            val = (n/2.0) / k
            if 0 <= val <= 1:
                restarts.append(np.concatenate([np.full(k, val), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            restarts.append(0.5 + 0.35*np.sin(freq*x + phase))
    
    return restarts

def construct():
    """Construct step heights minimizing maximum overlap using subgradient and coordinate descent."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_val, _, _ = _compute_max_overlap(best_x)
    
    restarts = _construct_diverse_initializations(n)
    
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        
        x = _project(x0, target_sum)
        
        # Phase 1: Subgradient ascent
        try:
            h_trial, val_trial = _subgradient_ascent(x, target_sum, max_iters=300, step_scale=0.05)
            if val_trial < best_val:
                best_val = val_trial
                best_x = h_trial
            x = h_trial
        except Exception:
            pass
        
        # Phase 2: Coordinate descent polish
        if _time_remaining() > 5.0:
            try:
                h_trial, val_trial = _coordinate_descent(x, target_sum, max_iters=50)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = h_trial
            except Exception:
                pass
    
    # Final polish on the best solution found
    if _time_remaining() > 10.0:
        try:
            h_trial, val_trial = _subgradient_ascent(best_x, target_sum, max_iters=200, step_scale=0.02)
            if val_trial < best_val:
                best_val = val_trial
                best_x = h_trial
        except Exception:
            pass
        
        if _time_remaining() > 5.0:
            try:
                h_trial, val_trial = _coordinate_descent(best_x, target_sum, max_iters=100)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = h_trial
            except Exception:
                pass
    
    # Ensure exact sum and bounds
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
