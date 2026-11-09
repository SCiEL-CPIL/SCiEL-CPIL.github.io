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

def precompute_support(s):
    """
    Precompute structures for a fixed support to speed up repeated evaluations.
    """
    n = len(s)
    s = np.maximum(s, 0.0)
    
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    return {
        'n': n,
        's': s,
        'ss_sorted': ss_sorted,
        'sort_idx': sort_idx
    }

def eval_with_precomp(precomp, p):
    """
    Fast evaluation using precomputed support structure.
    p must be normalized (sums to 1).
    """
    n = precomp['n']
    s = precomp['s']
    ss_sorted = precomp['ss_sorted']
    sort_idx = precomp['sort_idx']
    
    # Build pp in the sorted order
    pp = p[:, None] * p[None, :]
    pp_flat = pp.flatten()
    pp_sorted = pp_flat[sort_idx]
    
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

def optimize_weights_for_fixed_support(s, time_limit, seed=42):
    """
    Optimizes weights for a fixed support s using Random Restarted Hill Climbing
    with precomputed support structure for speed.
    """
    n = len(s)
    if n == 0:
        return np.array([1.0]), 0.0
    
    precomp = precompute_support(s)
    
    best_w = np.ones(n) / n
    best_score = eval_with_precomp(precomp, best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    max_restarts = 100
    max_steps_per_restart = 20000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization with more peaked distributions
        w = rng.dirichlet(np.ones(n) * 0.5)
        current_score = eval_with_precomp(precomp, w)
        
        if current_score > best_score:
            best_score = current_score
            best_w = w.copy()
        
        # Hill Climbing with adaptive step
        step_size = 0.1
        stuck_count = 0
        
        for step in range(max_steps_per_restart):
            if time.time() - start_time > time_limit:
                return best_w, best_score
            
            # Pick two random indices
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 == i2:
                continue
                
            if w[i1] < 1e-7:
                continue
                
            # Move mass from i1 to i2
            max_move = w[i1] * 0.3
            move = max_move * (1 if rng.random() < 0.5 else -1)
            
            if move > 0 and w[i1] < move:
                move = w[i1] * 0.5
            if move < 0 and w[i2] < -move:
                move = w[i2] * 0.5
                
            if abs(move) < 1e-9:
                continue
                
            w_trial = w.copy()
            w_trial[i1] -= move
            w_trial[i2] += move
            
            if np.any(w_trial < -1e-9):
                continue
                
            score_trial = eval_with_precomp(precomp, w_trial)
            
            if score_trial > current_score + 1e-12:
                w = w_trial
                current_score = score_trial
                if current_score > best_score:
                    best_score = current_score
                    best_w = w.copy()
                stuck_count = 0
                step_size *= 0.97
            else:
                stuck_count += 1
                step_size *= 1.03
                if step_size > 0.3:
                    step_size = 0.3
                if step_size < 0.005:
                    break  # Local minimum, restart
        
    return best_w, best_score

def generate_support_candidates():
    """
    Generate a diverse set of support structures.
    """
    candidates = []
    
    # Powers of 2
    for n in [4, 5, 6, 8, 10, 12, 15]:
        s = np.array([0.0] + [2.0 ** i for i in range(1, n)])
        candidates.append(s)
        
    # Powers of 3
    for n in [4, 5, 6, 8, 10]:
        s = np.array([0.0] + [3.0 ** i for i in range(1, n)])
        candidates.append(s)
        
    # Linear
    for n in [4, 5, 6, 8, 10, 12, 15]:
        s = np.arange(n, dtype=np.float64)
        candidates.append(s)
        
    # Mixed: 0, then powers of 1.5
    for n in [6, 8, 10, 12, 15]:
        s = np.array([0.0] + [1.5 ** i for i in range(1, n)])
        candidates.append(s)
        
    # Mixed: 0, then powers of 1.2
    for n in [10, 15, 20]:
        s = np.array([0.0] + [1.2 ** i for i in range(1, n)])
        candidates.append(s)
        
    # Two point: {0, a}
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        s = np.array([0.0, float(a)])
        candidates.append(s)
        
    # Three point: {0, a, b}
    for a in [1, 2, 4, 8, 16]:
        for b in [a*1.5, a*2, a*3, a*5, a*10]:
            if b > 0:
                s = np.array([0.0, float(a), float(b)])
                candidates.append(s)
                
    # Four point: {0, a, b, c}
    for a in [1, 2, 4]:
        for b in [a*2, a*3, a*5]:
            for c in [b*2, b*3, b*5]:
                s = np.array([0.0, float(a), float(b), float(c)])
                candidates.append(s)
    
    # Geometric progression
    for n in [5, 8, 10, 12]:
        for r in [1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
            s = r ** np.arange(n)
            s[0] = 0.0
            candidates.append(s)
            
    # Uniform grid
    for n in [5, 8, 10, 12, 15]:
        for max_val in [1, 2, 5, 10]:
            s = np.linspace(0, max_val, n)
            candidates.append(s)
            
    # Bimodal: cluster near 0 and cluster at high values
    for n1 in [2, 3, 4]:
        for n2 in [2, 3, 4]:
            for gap in [5, 10, 20, 50]:
                s1 = np.linspace(0, 1, n1)
                s2 = np.linspace(gap, gap + 1, n2)
                s = np.concatenate([s1, s2])
                candidates.append(s)
    
    # Tri-modal
    for n1 in [2, 3]:
        for n2 in [2, 3]:
            for n3 in [2, 3]:
                for gap1 in [5, 10]:
                    for gap2 in [10, 20]:
                        s1 = np.linspace(0, 1, n1)
                        s2 = np.linspace(gap1, gap1 + 1, n2)
                        s3 = np.linspace(gap1 + gap2, gap1 + gap2 + 1, n3)
                        s = np.concatenate([s1, s2, s3])
                        candidates.append(s)
                        
    return candidates

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 20.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Generate support candidates
    supports = generate_support_candidates()
    
    # Score all supports with uniform weights to prioritize
    scored_supports = []
    for s in supports:
        n = len(s)
        w_init = np.ones(n) / n
        score = evaluate_measure(s, w_init)
        scored_supports.append((score, s))
        
    scored_supports.sort(key=lambda x: -x[0])
    
    # Process top supports with weight optimization
    num_supports = min(len(scored_supports), 40)
    
    # Time allocation with exponential decay
    total_time_budget = deadline - safety_margin - time.time()
    
    time_weights = []
    for i in range(num_supports):
        tw = 0.6 ** i
        time_weights.append(tw)
    tw_sum = sum(time_weights)
    time_weights = [tw / tw_sum * total_time_budget for tw in time_weights]
    time_weights = [max(2.0, tw) for tw in time_weights]
    
    for i in range(num_supports):
        if time.time() + 5.0 > deadline - safety_margin:
            break
            
        score_seed, s = scored_supports[i]
        
        if score_seed > best_score:
            best_score = score_seed
            best_support = s.copy()
            best_weights = np.ones(len(s)) / len(s)
        
        time_limit = time_weights[i]
        
        remaining = deadline - safety_margin - time.time()
        if remaining < 3.0:
            break
        time_limit = min(time_limit, remaining - 3.0)
        
        try:
            w_opt, score_opt = optimize_weights_for_fixed_support(s, time_limit, seed=i)
            
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
