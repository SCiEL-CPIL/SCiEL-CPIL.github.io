import numpy as np
import time

N = 400
TIME_LIMIT = 110  # seconds, safety margin before 120s kill

def _compute_score(f):
    """Compute max |autoconv| / (sum f)^2."""
    # Use FFT for fast autoconvolution
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * np.conj(F), size)[:n]
    max_ac = np.max(np.abs(ac))
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    
    # Start with all +1s (known to give 1/N)
    f = np.ones(N, dtype=np.float64)
    best_score = _compute_score(f)
    best_f = f.copy()
    
    # Local search: try flipping individual elements
    # For ±1 signals, flipping changes sign
    # But let's allow continuous values to explore more
    
    # Actually, for continuous values, the problem is different.
    # Let's try a gradient-free local search with continuous heights.
    
    rng = np.random.default_rng(42)
    
    # Try different initializations and local improvements
    attempts = 0
    while time.time() - start_time < TIME_LIMIT:
        attempts += 1
        
        # Create a perturbation
        if attempts == 1:
            candidate = best_f.copy()
        else:
            # Mix: sometimes start from best, sometimes random
            if rng.random() < 0.5:
                candidate = best_f.copy()
            else:
                candidate = rng.choice([-1.0, 1.0], size=N)
            
            # Add small perturbations to some elements
            n_perturb = rng.integers(1, min(20, N // 10) + 1)
            indices = rng.choice(N, size=n_perturb, replace=False)
            candidate[indices] += rng.normal(0, 0.5, size=n_perturb)
            # Clamp to reasonable range
            candidate = np.clip(candidate, -10, 10)
        
        # Local improvement: coordinate descent
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT:
            improved = False
            score_before = _compute_score(candidate)
            if score_before >= best_score:
                break
            
            # Try adjusting each element slightly
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT:
                    break
                old_val = candidate[i]
                # Try small steps
                for delta in [-0.1, -0.5, -1.0, 0.1, 0.5, 1.0]:
                    candidate[i] = old_val + delta
                    s = _compute_score(candidate)
                    if s < score_before:
                        score_before = s
                        improved = True
                    else:
                        candidate[i] = old_val
                
                if score_before < best_score:
                    best_score = score_before
                    best_f = candidate.copy()
        
        if time.time() - start_time > TIME_LIMIT - 5:
            break
    
    # Ensure we return a valid result
    # If best_score is inf or very large, fall back to all ones
    if not np.isfinite(best_score) or best_score > 1.0:
        return np.ones(N, dtype=np.float64)
    
    return best_f
