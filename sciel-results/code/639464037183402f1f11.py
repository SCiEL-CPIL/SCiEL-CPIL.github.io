import numpy as np
import time
import sys

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    
    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    
    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner
            
    return P

def optimize_weights_for_fixed_support(s, time_limit):
    """
    Optimizes weights for a fixed support s using Random Restarted Hill Climbing.
    Since s is fixed, we can precompute the 'winning' probability matrix for each (k,l) pair.
    Let A[k,l] be the matrix of size n x n where A[k,l,i,j] = 1 if s_i+s_j < 2*s_l - s_k.
    Then P = sum_{k,l} p_k p_l (sum_{i,j} A[k,l,i,j] p_i p_j).
    This is equivalent to P = sum_{k,l} p_k p_l Q_{k,l}(p) where Q is a quadratic form.
    Actually, we can just use the fast evaluate_measure, but we restrict search to weights.
    """
    n = len(s)
    best_w = np.ones(n) / n
    best_score = evaluate_measure(s, best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Pre-compute the mask for faster evaluation? 
    # The current evaluate_measure is O(n^2 log n). For small n (<=20), it's very fast.
    # We will perform many steps.
    
    max_steps = 50000
    step_size = 0.1
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        w = rng.dirichlet(np.ones(n))
        current_score = evaluate_measure(s, w)
        
        if current_score > best_score:
            best_score = current_score
            best_w = w.copy()
            
        # Hill Climbing
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return best_w, best_score
            
            # Pick two random indices to swap weight
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 == i2:
                continue
                
            # Determine direction
            if w[i1] < 1e-6:
                continue
                
            # Move some mass from i1 to i2
            # Adaptive step size
            move = min(w[i1] * 0.1, 0.05) * (1 if rng.random() < 0.5 else -1)
            # Ensure we don't go negative
            if move > 0 and w[i1] < move:
                move = w[i1] * 0.5
            if move < 0 and w[i2] < -move:
                move = w[i2] * 0.5
                
            if move == 0:
                continue
                
            w_trial = w.copy()
            w_trial[i1] -= move
            w_trial[i2] += move
            
            # Check validity
            if np.any(w_trial < -1e-9):
                continue
                
            score_trial = evaluate_measure(s, w_trial)
            
            if score_trial > current_score:
                w = w_trial
                current_score = score_trial
                if current_score > best_score:
                    best_score = current_score
                    best_w = w.copy()
                # Shrink step size slightly on success
                step_size *= 0.95
            else:
                # Grow step size slightly on failure
                step_size *= 1.05
                if step_size > 0.5:
                    step_size = 0.5
                if step_size < 0.001:
                    break # Local minimum found, restart

    return best_w, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Strategy: Test a few different support structures.
    # 1. Powers of 2: 0, 1, 2, 4, ...
    # 2. Powers of 3: 0, 1, 3, 9, ...
    # 3. Linear: 0, 1, 2, ..., n-1
    # 4. Exponential with base 1.5
    
    supports_to_test = []
    
    # Powers of 2
    for n in [10, 15, 20, 30]:
        s = np.array([0.0] + [2.0 ** i for i in range(1, n)])
        supports_to_test.append(s)
        
    # Powers of 3
    for n in [10, 15, 20]:
        s = np.array([0.0] + [3.0 ** i for i in range(1, n)])
        supports_to_test.append(s)
        
    # Linear
    for n in [10, 20, 50]:
        s = np.arange(n, dtype=np.float64)
        supports_to_test.append(s)
        
    # Mixed: 0, then powers of 1.5
    for n in [15, 25]:
        s = np.array([0.0] + [1.5 ** i for i in range(1, n)])
        supports_to_test.append(s)
        
    # Sort supports by initial score with uniform weights to prioritize
    scored_supports = []
    for s in supports_to_test:
        n = len(s)
        w_init = np.ones(n) / n
        score = evaluate_measure(s, w_init)
        scored_supports.append((score, s))
        
    scored_supports.sort(key=lambda x: -x[0])
    
    # Allocate time to top supports
    num_supports = min(len(scored_supports), 10)
    
    for i in range(num_supports):
        if time.time() + 5.0 > deadline - safety_margin:
            break
            
        score_seed, s = scored_supports[i]
        
        # Update best if seed is better
        if score_seed > best_score:
            best_score = score_seed
            best_support = s.copy()
            best_weights = np.ones(len(s)) / len(s)
            
        # Allocate time for optimization
        remaining = deadline - safety_margin - time.time()
        if remaining < 5.0:
            break
            
        # Give more time to earlier (better) supports
        # Exponential decay of time budget
        time_alloc = remaining * (0.5 ** i)
        time_alloc = max(2.0, min(100.0, time_alloc))
        
        try:
            w_opt, score_opt = optimize_weights_for_fixed_support(s, time_alloc)
            
            if score_opt > best_score:
                best_score = score_opt
                best_support = s.copy()
                best_weights = w_opt.copy()
                
        except Exception:
            pass
            
    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n
        
    # Final validation and cleanup
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        n = len(best_support)
        best_weights = np.ones(n) / n
        
    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
