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

def optimize_weights(s, time_limit, rng):
    """
    Optimize weights for fixed support using coordinate descent + random restarts.
    """
    n = len(s)
    best_w = np.ones(n) / n
    best_score = evaluate_measure(s, best_w)
    
    start = time.time()
    
    for restart in range(50):
        if time.time() - start > time_limit:
            break
        
        # Random Dirichlet initialization
        alpha = rng.dirichlet(np.ones(n) * 0.5)
        w = alpha.copy()
        cur_score = evaluate_measure(s, w)
        
        if cur_score > best_score:
            best_score = cur_score
            best_w = w.copy()
        
        # Coordinate descent: iterate over each coordinate
        for epoch in range(200):
            if time.time() - start > time_limit:
                break
            
            improved = False
            for i in range(n):
                if w[i] < 1e-12:
                    continue
                
                # Try moving mass from i to each other coordinate
                best_move = 0.0
                best_j = -1
                best_frac = 0.0
                
                for j in range(n):
                    if j == i:
                        continue
                    # Try moving fraction f of w[i] to w[j]
                    for frac in [0.05, 0.1, 0.2, 0.3, 0.5]:
                        move = w[i] * frac
                        if move < 1e-15:
                            continue
                        w_trial = w.copy()
                        w_trial[i] -= move
                        w_trial[j] += move
                        if np.any(w_trial < -1e-9):
                            continue
                        sc = evaluate_measure(s, w_trial)
                        if sc > cur_score + 1e-12:
                            if sc - cur_score > best_move:
                                best_move = sc - cur_score
                                best_j = j
                                best_frac = frac
                
                if best_j >= 0:
                    move = w[i] * best_frac
                    w[i] -= move
                    w[best_j] += move
                    cur_score += best_move
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_w = w.copy()
            
            if not improved:
                break
    
    return best_w, best_score

def optimize_support_and_weights(s_init, w_init, time_limit, rng):
    """
    Jointly optimize support and weights.
    """
    n = len(s_init)
    best_s = s_init.copy()
    best_w = w_init.copy()
    best_score = evaluate_measure(s_init, w_init)
    
    start = time.time()
    
    for iteration in range(30):
        if time.time() - start > time_limit:
            break
        
        # First optimize weights for current support
        w_opt, score_w = optimize_weights(best_s, (time_limit - (time.time() - start)) * 0.4, rng)
        if score_w > best_score:
            best_score = score_w
            best_w = w_opt.copy()
        
        if time.time() - start > time_limit:
            break
        
        # Then perturb support points
        s_trial = best_s.copy()
        for i in range(n):
            if rng.random() < 0.7:
                # Small perturbation
                s_trial[i] = max(0.0, s_trial[i] * (1.0 + rng.normal(0, 0.1)))
        
        # Ensure support is sorted and non-negative
        s_trial = np.sort(np.maximum(s_trial, 0.0))
        
        # Optimize weights for perturbed support
        w_trial, score_trial = optimize_weights(s_trial, (time_limit - (time.time() - start)) * 0.3, rng)
        
        if score_trial > best_score:
            best_score = score_trial
            best_s = s_trial.copy()
            best_w = w_trial.copy()
    
    return best_s, best_w, best_score

def generate_two_cluster_support(n, gap, rng):
    """
    Generate a support with two clusters: low values near 0, high values near gap.
    """
    low_count = n // 2
    high_count = n - low_count
    
    low_vals = np.sort(rng.uniform(0, 0.1, size=low_count))
    high_vals = np.sort(gap + rng.uniform(0, 0.1, size=high_count))
    
    s = np.concatenate([low_vals, high_vals])
    return np.sort(s)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 15.0
    rng = np.random.default_rng(42)
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Strategy 1: Two-cluster supports with various gaps
    # The key idea: if X4 is from the high cluster with prob p_high,
    # and X1,X2,X3 are from the low cluster with prob p_low,
    # then P[X1+X2+X3 < 2*X4] is high when 2*high_val >> 3*max_low_val
    
    gaps_to_try = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    n_values = [4, 6, 8, 10, 12, 16]
    
    candidates = []
    
    for gap in gaps_to_try:
        for n in n_values:
            s = generate_two_cluster_support(n, gap, rng)
            # Initial weights: put more mass on high cluster
            w = np.ones(n)
            low_count = n // 2
            w[:low_count] = 0.3
            w[low_count:] = 0.7
            w = w / np.sum(w)
            candidates.append((s, w))
    
    # Strategy 2: Geometric supports
    for base in [1.5, 2.0, 3.0]:
        for n in [8, 12, 16]:
            s = np.array([0.0] + [base ** i for i in range(1, n)])
            # Weight toward extremes
            w = np.ones(n)
            w[0] = 0.3
            w[-1] = 0.3
            w[1:-1] = 0.4 / (n - 2) if n > 2 else 0.4
            w = w / np.sum(w)
            candidates.append((s, w))
    
    # Strategy 3: Linear with bias
    for n in [8, 12, 16]:
        s = np.arange(n, dtype=np.float64)
        # Biased weights: more mass on high end
        w = np.linspace(0.5, 2.0, n)
        w = w / np.sum(w)
        candidates.append((s, w))
    
    # Strategy 4: Single point + spread
    for n in [6, 10, 14]:
        s = np.zeros(n)
        s[0] = 0.0
        s[1:] = np.linspace(1.0, 5.0, n-1)
        w = np.zeros(n)
        w[0] = 0.5
        w[1:] = 0.5 / (n-1)
        candidates.append((s, w))
    
    # Evaluate all candidates with uniform-ish weights first to rank them
    scored_candidates = []
    for s, w in candidates:
        score = evaluate_measure(s, w)
        scored_candidates.append((score, s, w))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Update best from top candidates
    for score, s, w in scored_candidates[:10]:
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w.copy()
    
    # Allocate time to optimize top candidates
    remaining = deadline - safety_margin - time.time()
    if remaining < 5.0:
        # Not enough time, return what we have
        pass
    else:
        # Top 5 candidates get most time
        top_k = min(5, len(scored_candidates))
        for i in range(top_k):
            if time.time() > deadline - safety_margin:
                break
            
            score_seed, s, w = scored_candidates[i]
            
            # Allocate time: more for better seeds
            frac = 0.4 ** i
            time_alloc = max(2.0, min(remaining * frac, remaining * 0.5))
            
            try:
                s_opt, w_opt, score_opt = optimize_support_and_weights(
                    s, w, time_alloc, rng
                )
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_support = s_opt.copy()
                    best_weights = w_opt.copy()
                    
            except Exception:
                pass
    
    # Final validation
    if best_support is None:
        n = 8
        best_support = np.array([0.0, 0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0])
        best_weights = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
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
    print(f"Support: {s}")
    print(f"Weights: {w}")
