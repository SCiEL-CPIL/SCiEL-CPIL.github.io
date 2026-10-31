import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = fftconvolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _generate_sidon_candidates():
    """
    Generate a list of candidate Sidon sets of various sizes.
    Combines known good structures with random generation.
    """
    candidates = []
    
    # Known good base structures (shifted/modified to fit [0, 50))
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and _is_sidon_set(s):
            candidates.append(list(s))
    
    # Random generation of Sidon sets
    np.random.seed(123)
    existing = [frozenset(c) for c in candidates]
    for _ in range(2000):
        k = np.random.randint(4, 12)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if _is_sidon_set(indices):
            fs = frozenset(indices)
            if fs not in existing:
                candidates.append(indices)
                existing.append(fs)
                if len(candidates) > 60:
                    break
    return candidates

def _optimize_weights(support, deadline):
    """
    Optimize weights on a fixed support set to maximize the ratio.
    Uses smooth max surrogate for the convolution peak.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial weights: equal
    w_init = np.ones(k) / k
    h_init = np.zeros(N)
    h_init[support] = w_init
    best_score = _exact_ratio(h_init)
    best_w = w_init.copy()
    
    # Try multiple betas for the smooth max
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 2:
            break
        
        def _neg_ratio_smooth(params):
            # params are log weights
            w_local = np.exp(params)
            s = np.sum(w_local)
            if s < 1e-15:
                return 0.0
            w_local = w_local / s
            
            h_local = np.zeros(N)
            h_local[support] = w_local
            
            l2_sq = np.sum(h_local**2)
            conv = fftconvolve(h_local, h_local, mode='full')
            
            # Smooth max
            m = np.max(conv)
            if m < 0: m = 0
            shifted = conv - m
            # Avoid overflow
            shifted = np.clip(shifted, -500, 500)
            smooth_max = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            
            if smooth_max < 1e-15:
                return 0.0
            return -l2_sq / smooth_max

        try:
            # Use L-BFGS-B
            res = minimize(
                _neg_ratio_smooth,
                np.log(w_init + 1e-10),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            w_res = np.exp(res.x)
            s = np.sum(w_res)
            if s > 1e-15:
                w_res = w_res / s
            else:
                w_res = w_init
            
            h_res = np.zeros(N)
            h_res[support] = w_res
            
            score = _exact_ratio(h_res)
            if score > best_score:
                best_score = score
                best_w = w_res.copy()
                w_init = w_res.copy() # Use best result as init for next beta
        except Exception:
            pass
            
    h_best = np.zeros(N)
    h_best[support] = best_w
    return h_best, best_score

def _greedy_sidon_search(deadline):
    """
    Greedy search for a support set with low autocorrelation peak.
    Starts with small sets and greedily adds elements that minimize the peak.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try multiple random starts
    np.random.seed(42)
    
    max_k = 15 # Max number of non-zeros to try
    num_starts = 15
    
    for start_id in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Random initial set of size 3-5
        init_size = 3 + (start_id % 3)
        current_indices = np.random.choice(N, size=init_size, replace=False)
        current_indices.sort()
        
        # Ensure it's a Sidon set? Not strictly required for the greedy step, 
        # but starting with one might help.
        # If not Sidon, it's still a valid start.
        
        # Optimize weights on this initial set to get a baseline
        h_curr, curr_score = _optimize_weights(list(current_indices), deadline)
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()
            
        # Greedy expansion
        for _ in range(max_k - len(current_indices)):
            if time.time() > deadline - 5:
                break
            
            best_new_index = -1
            best_new_score = curr_score
            best_new_h = h_curr.copy()
            
            # Try adding each possible new index
            candidates = [i for i in range(N) if i not in current_indices]
            
            # Quick metric: minimize max autocorrelation of the set (with unit weights)
            base_h = np.zeros(N)
            base_h[current_indices] = 1.0
            base_conv = fftconvolve(base_h, base_h, mode='full')
            base_max = np.max(base_conv)
            
            # Find index j that minimizes the peak increase
            best_candidate_score = float('inf')
            best_candidate_idx = -1
            
            for j in candidates:
                h_test = base_h.copy()
                h_test[j] = 1.0
                conv_test = fftconvolve(h_test, h_test, mode='full')
                max_test = np.max(conv_test)
                # We prefer lower max autocorrelation
                if max_test < best_candidate_score:
                    best_candidate_score = max_test
                    best_candidate_idx = j
            
            if best_candidate_idx == -1:
                break
                
            # Add the best candidate
            new_indices = list(current_indices) + [best_candidate_idx]
            new_indices.sort()
            
            # Optimize weights on the new set
            h_optimized, new_score = _optimize_weights(new_indices, deadline)
            
            if new_score > best_new_score:
                curr_score = new_score
                current_indices = new_indices
                h_curr = h_optimized
                
            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Evaluate pre-generated Sidon sets
    sidon_candidates = _generate_sidon_candidates()
    
    for support in sidon_candidates:
        if time.time() > deadline - 10:
            break
        h_opt, score = _optimize_weights(support, deadline)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Phase 2: Greedy search for potentially better supports
    h_greedy, score_greedy = _greedy_sidon_search(deadline)
    if score_greedy > best_score:
        best_score = score_greedy
        best_h = h_greedy
        
    # Fallback if everything failed
    if best_score < 0.5:
        h_fallback = np.ones(N) / N
        score_fallback = _exact_ratio(h_fallback)
        if score_fallback > best_score:
            best_score = score_fallback
            best_h = h_fallback
            
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
