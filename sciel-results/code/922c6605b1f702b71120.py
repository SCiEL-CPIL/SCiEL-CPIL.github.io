"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    Overlap at shift k is sum_i h[i] * (1-h[(i+k) % N])
    We want to minimize the maximum over all non-zero shifts k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Compute all circular cross-correlations of h with (1-h)
    # Using FFT for speed
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    # Circular correlation: sum_i h[i] * (1-h)[(i+k) % n]
    # This is the circular cross-correlation of h and (1-h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    # corr[k] = sum_i h[i] * (1-h)[(i-k) % n]  ... need to be careful
    # Actually let's think: np.fft.ifft(fft(a) * conj(fft(b))) gives circular cross-correlation
    # c[k] = sum_i a[i+k] * b[i] (circular)
    # We want overlap[k] = sum_i h[i] * (1-h)[(i+k) % n]
    # Let a = h, b = 1-h. Then c[k] = sum_i h[i+k] * (1-h)[i]
    # overlap[k] = sum_i h[i] * (1-h)[i+k] = c[-k] = c[n-k]
    # So max over k != 0 of overlap[k] = max over k != 0 of c[n-k] = max over k != 0 of c[k]
    # (since as k ranges over 1..n-1, n-k also ranges over 1..n-1)
    
    # Exclude k=0 (which is sum h[i]*(1-h[i]) = sum h[i] - sum h[i]^2)
    max_overlap = np.max(np.abs(corr[1:]))  # actually we want the actual value, not abs
    # Wait, overlap can be negative? No, h[i] in [0,1], 1-h[j] in [0,1], so product is in [0,1]
    # So corr values should be non-negative. Let's just take max.
    max_overlap = np.max(corr[1:])
    return max_overlap

def _objective_and_grad(h):
    """Objective: maximize the minimum... no, minimize the maximum overlap.
    We use a smooth approximation: soft-max of the overlaps.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    
    # Compute all overlaps via FFT
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    
    # We want to minimize max(corr[1:])
    # Use log-sum-exp for smooth max
    c = corr[1:]
    # Add small epsilon to avoid issues
    c = np.clip(c, 1e-12, None)
    beta = 20.0  # temperature parameter for soft-max
    log_sum_exp = np.log(np.sum(np.exp(beta * c))) / beta
    return log_sum_exp

def _objective(h):
    return _objective_and_grad(h)

def _construct_with_optimization():
    """Try to find a good h via optimization."""
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    # Multi-start optimization
    np.random.seed(42)
    
    n_starts = 50
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        if start_idx == 0:
            h0 = np.full(N, 0.5)
        else:
            # Random perturbation with some structure
            h0 = np.random.rand(N)
            # Project to satisfy sum = N/2
            h0 = h0 - np.mean(h0) + 0.5
            h0 = np.clip(h0, 0.0, 1.0)
            # Re-project to satisfy sum constraint
            # Simple iterative projection
            for _ in range(100):
                h0 = np.clip(h0, 0.0, 1.0)
                h0 = h0 + (N/2.0 - np.sum(h0)) / N
            h0 = np.clip(h0, 0.0, 1.0)
            # Final adjustment
            diff = N/2.0 - np.sum(h0)
            if abs(diff) > 1e-10:
                # Distribute difference
                mask = (h0 > 1e-6) & (h0 < 1.0 - 1e-6)
                if np.any(mask):
                    h0[mask] += diff / np.sum(mask)
                else:
                    h0 += diff / N
                    h0 = np.clip(h0, 0.0, 1.0)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            h_opt = result.x
            # Ensure sum constraint
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-10:
                mask = (h_opt > 1e-6) & (h_opt < 1.0 - 1e-6)
                if np.any(mask):
                    h_opt[mask] += diff / np.sum(mask)
                else:
                    h_opt += diff / N
                    h_opt = np.clip(h_opt, 0.0, 1.0)
            
            # Clip to [0,1]
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _compute_score(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    return best_h

def _construct_binary():
    """Try a binary sequence approach."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    # For binary sequences with N/2 ones, try different patterns
    # Start with simple patterns and use local search
    
    np.random.seed(123)
    
    # Try random binary sequences
    n_attempts = 200
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Generate random binary sequence with exactly N/2 ones
        indices = np.random.choice(N, size=N//2, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Local search: try flipping pairs (one 1 to 0, one 0 to 1)
        for _ in range(20):
            if time.time() > deadline:
                break
            
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            
            if len(ones) == 0 or len(zeros) == 0:
                break
            
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(zeros))]
            
            h[i] = 0.0
            h[j] = 1.0
            
            new_score = _compute_score(h)
            if new_score < score:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            else:
                h[i] = 1.0
                h[j] = 0.0
    
    return best_h

def _refine(h):
    """Refine a given h using local optimization."""
    start_time = time.time()
    deadline = start_time + 30.0
    
    h = np.array(h, dtype=np.float64)
    
    # Use L-BFGS-B with tighter tolerances
    try:
        result = minimize(
            _objective,
            h,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        
        h_opt = result.x
        diff = N/2.0 - np.sum(h_opt)
        if abs(diff) > 1e-10:
            mask = (h_opt > 1e-6) & (h_opt < 1.0 - 1e-6)
            if np.any(mask):
                h_opt[mask] += diff / np.sum(mask)
            else:
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
        
        h_opt = np.clip(h_opt, 0.0, 1.0)
        
        if _compute_score(h_opt) <= _compute_score(h) + 1e-9:
            return h_opt
    except Exception:
        pass
    
    return h

def construct():
    """Main entry point."""
    start_time = time.time()
    total_deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Strategy 1: Optimization-based approach
    h_opt = _construct_with_optimization()
    score_opt = _compute_score(h_opt)
    
    if time.time() > total_deadline:
        return h_opt
    
    # Strategy 2: Binary sequence approach
    h_bin = _construct_binary()
    score_bin = _compute_score(h_bin)
    
    if time.time() > total_deadline:
        return h_opt if score_opt <= score_bin else h_bin
    
    # Pick the better one and refine
    if score_opt <= score_bin:
        h_best = h_opt
    else:
        h_best = h_bin
    
    if time.time() < total_deadline - 5.0:
        h_best = _refine(h_best)
    
    # Final validation
    h_best = np.clip(h_best, 0.0, 1.0)
    diff = N/2.0 - np.sum(h_best)
    if abs(diff) > 1e-8:
        mask = (h_best > 1e-6) & (h_best < 1.0 - 1e-6)
        if np.any(mask):
            h_best[mask] += diff / np.sum(mask)
        else:
            h_best += diff / N
            h_best = np.clip(h_best, 0.0, 1.0)
    
    return h_best
# EVOLVE-BLOCK-END
