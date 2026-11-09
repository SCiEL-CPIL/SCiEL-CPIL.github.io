import numpy as np
import time
import sys
from scipy.optimize import minimize

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
    Vectorized over (k,l) pairs.
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
    
    # Vectorized: thresholds for all (k, l) pairs
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    # C[idx-1] where idx>0, else 0
    prob_inner = np.where(idx > 0, C[np.maximum(idx - 1, 0)], 0.0)
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
    
    return float(P)

def optimize_weights_for_fixed_support(s, time_limit, seed=42):
    """
    Optimizes weights for a fixed support s using a combination of
    scipy SLSQP and fast in-place hill climbing.
    """
    n = len(s)
    if n == 0:
        return np.array([1.0]), 0.0
    
    precomp = precompute_support(s)
    
    best_w = np.ones(n) / n
    best_score = eval_with_precomp(precomp, best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    def neg_score(w):
        return -eval_with_precomp(precomp, w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-10, 1.0)] * n
    
    # Phase 1: scipy SLSQP from multiple starts
    n_scipy_starts = min(20, max(5, int(time_limit / 2.0)))
    for i in range(n_scipy_starts):
        if time.time() - start_time > time_limit * 0.5:
            break
        if i == 0:
            w0 = np.ones(n) / n
        else:
            w0 = rng.dirichlet(np.ones(n) * 0.3)
        
        try:
            res = minimize(neg_score, w0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            w_opt = np.maximum(res.x, 0.0)
            ws = np.sum(w_opt)
            if ws > 1e-9:
                w_opt = w_opt / ws
            sc = eval_with_precomp(precomp, w_opt)
            if sc > best_score:
                best_score = sc
                best_w = w_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Fast in-place hill climbing
    remaining = time_limit - (time.time() - start_time)
    if remaining < 1.0:
        return best_w, best_score
    
    # Use best_w as starting point for hill climbing
    w = best_w.copy()
    current_score = best_score
    step_size = 0.05
    max_steps = int(remaining * 5000)  # estimate steps we can do
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        i1 = rng.integers(0, n)
        i2 = rng.integers(0, n)
        if i1 == i2:
            continue
        if w[i1] < 1e-7:
            continue
        
        max_move = w[i1] * 0.4
        move = max_move * (1.0 if rng.random() < 0.5 else -1.0)
        if move > 0 and w[i1] < move:
            move = w[i1] * 0.5
        if move < 0 and w[i2] < -move:
            move = w[i2] * 0.5
        if abs(move) < 1e-9:
            continue
        
        # In-place mutation
        w[i1] -= move
        w[i2] += move
        
        if w[i1] < -1e-9 or w[i2] < -1e-9:
            w[i1] += move
            w[i2] -= move
            continue
        
        score_trial = eval_with_precomp(precomp, w)
        
        if score_trial > current_score + 1e-12:
            current_score = score_trial
            if current_score > best_score:
                best_score = current_score
                best_w = w.copy()
            step_size *= 0.98
        else:
            w[i1] += move
            w[i2] -= move
            step_size *= 1.04
            if step_size > 0.3:
                step_size = 0.3
            if step_size < 0.003:
                # Restart from best
                w = best_w.copy()
                current_score = best_score
                step_size = 0.05
    
    return best_w, best_score

def generate_support_candidates():
    """
    Generate a diverse set of support structures, emphasizing
    geometric sequences that favor the inequality.
    """
    candidates = []
    
    # Two point: {0, a} - simple but effective
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
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
    
    # Five point: {0, a, b, c, d}
    for a in [1, 2]:
        for b in [a*2, a*3]:
            for c in [b*2, b*3]:
                for d in [c*2, c*3]:
                    s = np.array([0.0, float(a), float(b), float(c), float(d)])
                    candidates.append(s)
    
    # Geometric sequences with 0 prepended (key structure)
    for n in [4, 5, 6, 7, 8, 10, 12]:
        for r in [1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
            s = r ** np.arange(n)
            s[0] = 0.0
            candidates.append(s)
    
    # Pure geometric (no 0)
    for n in [4, 5, 6, 8, 10]:
        for r in [1.5, 2.0, 2.5, 3.0]:
            s = r ** np.arange(n)
            candidates.append(s)
    
    # Linear
    for n in [4, 5, 6, 8, 10, 12, 15]:
        s = np.arange(n, dtype=np.float64)
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
    
    # "Staircase" supports: 0, then widely spaced values
    for n in [5, 6, 8, 10]:
        s = np.array([0.0] + [float(2 ** i * 3) for i in range(1, n)])
        candidates.append(s)
    
    # "Exponential with offset": 0, 1, then rapid growth
    for n in [5, 6, 8, 10]:
        s = np.array([0.0, 1.0] + [1.0 + 2.0 ** i for i in range(1, n-1)])
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
    num_supports = min(len(scored_supports), 50)
    
    # Time allocation: give more time to top candidates
    total_time_budget = deadline - safety_margin - time.time()
    
    time_weights = []
    for i in range(num_supports):
        tw = 0.55 ** i
        time_weights.append(tw)
    tw_sum = sum(time_weights)
    time_weights = [tw / tw_sum * total_time_budget for tw in time_weights]
    time_weights = [max(3.0, tw) for tw in time_weights]
    
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
    
    # Remove any zero-weight points to keep it clean (optional)
    # Actually keep them - they're valid
    
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
