import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _exact_ratio(h):
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _local_polish(h, deadline):
    """
    Coordinate descent polish: iterate over each coordinate, find optimal value
    given all others fixed, using the exact ratio.
    """
    h = np.array(h, dtype=np.float64).copy()
    n_iter = 0
    while time.time() < deadline - 2 and n_iter < 20:
        n_iter += 1
        improved = False
        for i in range(N):
            if time.time() > deadline - 2:
                break
            h_old = h[i]
            # Try a grid of values for coordinate i
            base = h.copy()
            base[i] = 0.0
            l1_base = np.sum(base)
            l2_base = np.sum(base**2)
            
            if l1_base < 1e-15:
                continue
                
            best_val = h_old
            best_score = _exact_ratio(h)
            
            # Grid search around current value
            step = max(1e-4, h_old * 0.1) if h_old > 0 else 1e-4
            for delta in np.linspace(-2*step, 2*step, 11):
                new_val = max(0.0, h_old + delta)
                h[i] = new_val
                s = _exact_ratio(h)
                if s > best_score:
                    best_score = s
                    best_val = new_val
            
            h[i] = best_val
            if abs(best_val - h_old) > 1e-8:
                improved = True
        if not improved:
            break
    return h

def _optimize_weights(fixed_indices, h_full, deadline):
    """
    Optimize weights on a fixed set of indices.
    Returns the best h_full found.
    """
    k = len(fixed_indices)
    if k == 0:
        return h_full
    
    # Initial weights: equal
    w = np.ones(k) / k
    
    def _neg_ratio_smooth(params, beta=50.0):
        # params are log weights (unconstrained)
        w_local = np.exp(params)
        w_sum = np.sum(w_local)
        if w_sum < 1e-15:
            return 0.0
        w_local = w_local / w_sum
        
        h_local = np.zeros(N)
        h_local[fixed_indices] = w_local
        
        l2_sq = np.sum(h_local**2)
        conv = np.convolve(h_local, h_local, mode='full')
        
        # Proper soft-max: log-sum-exp for numerical stability
        m = np.max(conv)
        if m < 1e-15:
            return 0.0
        # soft_max = m + (1/beta) * log(sum(exp(beta*(conv - m))))
        # This is the smooth approximation to max(conv)
        exp_terms = np.exp(beta * (conv - m))
        soft_max = m + np.log(np.sum(exp_terms)) / beta
        
        if soft_max < 1e-15:
            return 0.0
        return -l2_sq / soft_max

    best_score = _exact_ratio(h_full)
    best_w = w.copy()
    
    # Try multiple initializations and betas
    init_list = [np.ones(k) / k]
    # Add a few random initializations
    for _ in range(3):
        r = np.random.dirichlet(np.ones(k))
        init_list.append(r)
    
    for beta in [30.0, 60.0, 100.0]:
        if time.time() > deadline - 5:
            break
        for w_init in init_list:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _neg_ratio_smooth,
                    np.log(w_init + 1e-10),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                w_res = np.exp(res.x)
                w_res = w_res / np.sum(w_res)
                
                h_res = np.zeros(N)
                h_res[fixed_indices] = w_res
                
                score = _exact_ratio(h_res)
                if score > best_score:
                    best_score = score
                    best_w = w_res.copy()
            except Exception:
                pass
            
    h_best = np.zeros(N)
    h_best[fixed_indices] = best_w
    
    # Local polish on the sparse support
    if time.time() < deadline - 3:
        h_best = _local_polish(h_best, deadline)
    
    return h_best

def _greedy_sidon_search(deadline):
    """
    Greedy search for a support set with low autocorrelation peak.
    We start with a random small set and greedily add elements that minimize the peak autocorrelation.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try multiple random starts
    np.random.seed(42)
    
    max_k = 12 # Max number of non-zeros to try
    num_starts = 8
    
    for start_id in range(num_starts):
        if time.time() > deadline - 15:
            break
            
        # Random initial set of size 3-5
        init_size = 3 + (start_id % 3)
        current_indices = np.random.choice(N, size=init_size, replace=False)
        current_indices.sort()
        
        # Optimize weights on this initial set to get a baseline
        h_curr = np.zeros(N)
        h_curr[current_indices] = 1.0
        h_curr = _optimize_weights(list(current_indices), h_curr, deadline)
        curr_score = _exact_ratio(h_curr)
        
        # Greedy expansion
        for _ in range(max_k - len(current_indices)):
            if time.time() > deadline - 15:
                break
            
            # Try adding each possible new index
            candidates = [i for i in range(N) if i not in current_indices]
            
            # Quick metric: minimize max autocorrelation when adding unit weight at j
            base_h = np.zeros(N)
            base_h[current_indices] = 1.0
            
            best_candidate_score = float('inf')
            best_candidate_idx = -1
            second_best_score = float('inf')
            second_best_idx = -1
            
            for j in candidates:
                h_test = base_h.copy()
                h_test[j] = 1.0
                conv_test = np.convolve(h_test, h_test, mode='full')
                max_test = np.max(conv_test)
                # We prefer lower max autocorrelation
                if max_test < best_candidate_score:
                    second_best_score = best_candidate_score
                    second_best_idx = best_candidate_idx
                    best_candidate_score = max_test
                    best_candidate_idx = j
                elif max_test < second_best_score:
                    second_best_score = max_test
                    second_best_idx = j
            
            if best_candidate_idx == -1:
                break
                
            # Try the best and second-best candidates
            for cand_idx in [best_candidate_idx, second_best_idx]:
                if cand_idx == -1 or time.time() > deadline - 15:
                    continue
                
                new_indices = list(current_indices) + [cand_idx]
                new_indices.sort()
                
                # Optimize weights on the new set
                h_test = np.zeros(N)
                h_test[new_indices] = 1.0
                h_optimized = _optimize_weights(new_indices, h_test, deadline)
                new_score = _exact_ratio(h_optimized)
                
                if new_score > curr_score:
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
    
    best_h, best_score = _greedy_sidon_search(deadline)
    
    # If greedy search failed or score is bad, fallback to a simple dense initialization
    if best_score < 0.5:
        h_fallback = np.ones(N) / N
        score_fallback = _exact_ratio(h_fallback)
        if score_fallback > best_score:
            best_h = h_fallback
            best_score = score_fallback
            
    # Random restarts with different seeds
    for seed in [1, 2, 3, 7, 13]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        h_try, score_try = _greedy_sidon_search(deadline)
        if score_try > best_score:
            best_score = score_try
            best_h = h_try
            
    # Final local polish on the best solution found
    if time.time() < deadline - 3:
        best_h = _local_polish(best_h, deadline)
        final_score = _exact_ratio(best_h)
        if final_score > best_score:
            best_score = final_score
            
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
