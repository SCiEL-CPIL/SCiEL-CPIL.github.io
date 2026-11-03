import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _optimize_weights_on_support(indices, deadline):
    """
    Optimizes the weights of h on the given indices, assuming other entries are 0.
    Uses Nelder-Mead on the normalized weights.
    """
    k = len(indices)
    if k == 0:
        return None, 0.0
    if k == 1:
        h = np.zeros(N)
        h[indices[0]] = 1.0
        return h, _score_raw(h)

    # We optimize the unnormalized weights, then normalize.
    # Since the score is homogeneous, we can fix sum(w) = 1 and optimize w.
    # Let w be the vector of weights.
    
    def obj(w):
        h = np.zeros(N)
        h[indices] = w
        s = _score_raw(h)
        # We want to maximize s, so minimize -s
        return -s

    # Initial guess: uniform
    w0 = np.ones(k) / k
    
    # Bounds: w_i >= 0. Nelder-Mead doesn't handle bounds well, 
    # but if we start positive and the landscape is nice, it stays positive.
    # If it goes negative, we clip in the callback or just accept slight invalidity 
    # and clip at the end, but better to use a transformation or just careful steps.
    # Given the small dimension (k <= 6), we can just run it.
    
    try:
        res = opt.minimize(
            obj, 
            w0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        w_opt = res.x
        w_opt = np.clip(w_opt, 0, None)
        if np.sum(w_opt) < 1e-15:
            return None, 0.0
        w_opt /= np.sum(w_opt)
        
        h = np.zeros(N)
        h[indices] = w_opt
        score = _score_raw(h)
        return h, score
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Strategy: Search for optimal sparse supports
    # 1. Try all pairs (Sidon-like structures are good for minimizing max conv)
    # 2. Try random small sets and expand greedily
    
    indices_all = np.arange(N)
    
    # Phase 1: Pairs
    # For each pair (i, j), the support is {i, j}.
    # We can analytically or quickly optimize the weight ratio.
    # Let's just use the generic optimizer for k=2.
    
    # To save time, we don't check ALL pairs, but a good subset.
    # Actually, N=50, pairs are 1225. Optimization is fast for k=2.
    # Let's do all pairs.
    
    for i in range(N):
        if time.time() > deadline - 60:
            break
        for j in range(i + 1, N):
            if time.time() > deadline - 60:
                break
            h_cand, s_cand = _optimize_weights_on_support([i, j], deadline)
            if h_cand is not None and s_cand > best_score:
                best_score = s_cand
                best_h = h_cand.copy()
                
    # Phase 2: Triplets and Quadruplets via Greedy Expansion
    # Start from best pairs, try adding a 3rd index.
    
    # Get top 20 pairs based on score? Or just iterate all and keep top few?
    # Let's keep a list of top candidates.
    top_candidates = []
    
    # Re-evaluate pairs to get top ones? Or just use the best_h found so far as a seed?
    # Better: generate random supports of size 3 and 4 and optimize.
    # Also, try adding indices to the current best_h support.
    
    # Let's do a random search for supports of size 3 to 6.
    np.random.seed(123)
    
    max_support_size = 6
    
    # Random restarts
    for _ in range(200):
        if time.time() > deadline - 20:
            break
            
        k = np.random.randint(3, max_support_size + 1)
        # Ensure distinct indices
        idxs = np.random.choice(N, k, replace=False)
        
        h_cand, s_cand = _optimize_weights_on_support(idxs, deadline)
        if h_cand is not None and s_cand > best_score:
            best_score = s_cand
            best_h = h_cand.copy()
            
    # Phase 3: Local Search on Support
    # Take the best_h, identify its support (non-zero indices).
    # Try swapping one index in the support with one outside.
    
    if time.time() < deadline - 10:
        support = np.where(best_h > 1e-6)[0]
        if len(support) < 1:
            support = np.array([0])
            
        # Try to expand or swap
        # We will try to replace one index in support with a random index not in support
        # and re-optimize weights.
        
        attempts = 0
        improved = True
        while improved and attempts < 50 and time.time() < deadline - 5:
            improved = False
            attempts += 1
            
            current_support = set(support.tolist())
            # Pick a random index to remove (if size > 1)
            if len(support) > 1:
                remove_idx = np.random.choice(support)
            else:
                continue
                
            # Pick a random index to add
            candidates = [i for i in range(N) if i not in current_support]
            if not candidates:
                break
            add_idx = np.random.choice(candidates)
            
            new_support_list = list((current_support - {remove_idx}) | {add_idx})
            np.random.shuffle(new_support_list)
            
            h_cand, s_cand = _optimize_weights_on_support(new_support_list, deadline)
            if h_cand is not None and s_cand > best_score + 1e-10:
                best_score = s_cand
                best_h = h_cand.copy()
                support = np.where(best_h > 1e-6)[0]
                improved = True
                
    # Phase 4: Polish the best solution found
    # Run a dense optimization on the active support one more time with tighter tolerance?
    # Nelder-Mead is already quite good. Let's just ensure normalization and non-negativity.
    
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) > 1e-15:
        best_h /= np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return list(best_h)
