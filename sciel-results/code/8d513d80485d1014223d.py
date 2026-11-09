import numpy as np
import time
import scipy.optimize
import sys

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    Uses sorted pair sums + cumulative sum + binary search for O(n^2 log n) per eval.
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

def objective(params, n):
    """
    Objective function for scipy.optimize.
    params: array of length 2n. First n are support points, last n are weights.
    Returns -P to minimize.
    """
    s = params[:n]
    w = params[n:]
    
    if np.any(np.isnan(s)) or np.any(np.isnan(w)):
        return 1e10
        
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    
    if np.sum(w) < 1e-9:
        return 1e10
        
    val = evaluate_measure(s, w)
    return -val

def generate_diverse_candidates():
    """
    Generate a diverse set of candidate distributions as starting points.
    """
    candidates = []
    
    # Two-point distributions: mass at 0 and at 'a'
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 16.0, 25.0, 40.0, 64.0]:
        for mass_0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            support = np.array([0.0, float(a)])
            weights = np.array([mass_0, 1.0 - mass_0])
            candidates.append((support.copy(), weights.copy()))
    
    # Three-point distributions: {0, a, b}
    for a in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for b in [a + 1, a * 2, a * 3, a * 4]:
            if b <= 0:
                continue
            for m0 in [0.1, 0.3, 0.5]:
                for ma in [0.1, 0.3, 0.5]:
                    mb = 1.0 - m0 - ma
                    if mb <= 0:
                        continue
                    support = np.array([0.0, float(a), float(b)])
                    weights = np.array([m0, ma, mb])
                    candidates.append((support.copy(), weights.copy()))
    
    # Geometric-like: support {0, 1, ..., n-1}, weights decay
    for n in [4, 6, 8, 12, 16]:
        for r in [0.5, 0.7, 0.8, 0.9]:
            support = np.arange(n, dtype=np.float64)
            weights = r ** np.arange(n)
            candidates.append((support.copy(), weights.copy()))
    
    # Exponential support: {1, 2, 4, 8, ...}
    for n in [3, 4, 5, 6]:
        support = 2.0 ** np.arange(n)
        for r in [0.5, 0.7, 0.8]:
            weights = r ** np.arange(n)
            candidates.append((support.copy(), weights.copy()))
    
    # Bimodal: mass at 0 and at large value
    for gap in [2, 4, 8, 16, 32, 64]:
        for mass_0 in [0.2, 0.4, 0.6, 0.8]:
            support = np.array([0.0, float(gap)])
            weights = np.array([mass_0, 1.0 - mass_0])
            candidates.append((support.copy(), weights.copy()))
    
    # Uniform on small grid
    for n in [4, 6, 8, 10, 12, 16, 20]:
        support = np.linspace(0, 1, n)
        weights = np.ones(n) / n
        candidates.append((support.copy(), weights.copy()))
    
    # Triangular weights on grid
    for n in [6, 8, 12, 16, 20]:
        support = np.arange(n, dtype=np.float64)
        mid = (n - 1) / 2.0
        weights = 1.0 - np.abs(np.arange(n) - mid) / (mid + 0.5)
        weights = np.maximum(weights, 1e-6)
        candidates.append((support.copy(), weights.copy()))
    
    return candidates

def optimize_from_seed(s0, w0, n_target, time_limit=30.0):
    """
    Run Nelder-Mead optimization starting from a seed distribution.
    If n_target != len(s0), we either keep the seed's n or adjust.
    """
    n = len(s0)
    
    # Ensure weights are normalized and non-negative
    w0 = np.maximum(w0, 0.0)
    w_sum = np.sum(w0)
    if w_sum <= 1e-9:
        w0 = np.ones(n) / n
    else:
        w0 = w0 / w_sum
    
    s0 = np.maximum(s0, 0.0)
    
    x0 = np.concatenate([s0, w0])
    
    best_x = x0.copy()
    best_val = -objective(x0, n)
    
    start_time = time.time()
    
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            args=(n,), 
            method='Nelder-Mead', 
            options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
        )
        
        val = -res.fun
        if val > best_val:
            best_val = val
            best_x = res.x
        
    except Exception:
        pass
    
    # Local perturbation refinement
    s_best = np.maximum(best_x[:n], 0.0)
    w_best = np.maximum(best_x[n:], 0.0)
    w_sum = np.sum(w_best)
    if w_sum > 1e-9:
        w_best = w_best / w_sum
    
    # A few rounds of random perturbation
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb weights
        if n >= 2:
            w_trial = w_best.copy()
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            delta = np.random.uniform(0.01, 0.15) * np.max(w_best)
            if np.random.random() < 0.5:
                w_trial[i1] += delta
                w_trial[i2] -= delta
            else:
                w_trial[i2] += delta
                w_trial[i1] -= delta
            
            if np.all(w_trial >= -1e-9) and np.sum(w_trial) > 1e-9:
                w_trial = np.maximum(w_trial, 0.0)
                score = evaluate_measure(s_best, w_trial)
                if score > best_val + 1e-10:
                    w_best = w_trial
                    best_val = score
        
        # Perturb support (only for small n to keep eval fast)
        if n >= 3 and n <= 16:
            s_trial = s_best.copy()
            idx = np.random.randint(n)
            delta = np.random.uniform(-0.3, 0.3)
            s_trial[idx] = max(0.0, s_trial[idx] + delta)
            
            score = evaluate_measure(s_trial, w_best)
            if score > best_val + 1e-10:
                s_best = s_trial
                best_val = score
    
    return s_best, w_best, best_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 15.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Generate diverse candidates
    candidates = generate_diverse_candidates()
    
    # Sort by initial score to prioritize better seeds
    scored_candidates = []
    for s, w in candidates:
        try:
            score = evaluate_measure(s, w)
            scored_candidates.append((score, s, w))
        except Exception:
            pass
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Process top candidates with optimization
    # Each candidate gets a time budget
    num_candidates = min(len(scored_candidates), 40)
    
    for i in range(num_candidates):
        if time.time() + 5.0 > deadline - safety_margin:
            break
        
        score_seed, s_seed, w_seed = scored_candidates[i]
        
        if score_seed > best_score:
            best_score = score_seed
            best_support = s_seed.copy()
            best_weights = w_seed.copy()
        
        # Allocate time for optimization based on remaining time
        remaining = deadline - safety_margin - time.time()
        if remaining < 3.0:
            break
        
        # Give more time to better seeds
        time_limit = min(30.0, remaining / max(1, num_candidates - i))
        time_limit = max(3.0, time_limit)
        
        try:
            s_opt, w_opt, score_opt = optimize_from_seed(s_seed, w_seed, len(s_seed), time_limit=time_limit)
            
            if score_opt > best_score:
                best_score = score_opt
                best_support = s_opt
                best_weights = w_opt
                
        except Exception:
            pass
    
    # Also try a couple of larger-n optimizations if time permits
    if time.time() + 10.0 < deadline - safety_margin:
        for n_large in [10, 15, 20]:
            if time.time() + 5.0 > deadline - safety_margin:
                break
            
            s_init = np.linspace(0, 1, n_large)
            w_init = np.ones(n_large) / n_large
            
            remaining = deadline - safety_margin - time.time()
            if remaining < 5.0:
                break
            
            time_limit = min(25.0, remaining)
            
            try:
                s_opt, w_opt, score_opt = optimize_from_seed(s_init, w_init, n_large, time_limit=time_limit)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_support = s_opt
                    best_weights = w_opt
                    
            except Exception:
                pass
    
    # Fallback
    if best_support is None:
        n = 10
        best_support = np.linspace(0, 1, n)
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
