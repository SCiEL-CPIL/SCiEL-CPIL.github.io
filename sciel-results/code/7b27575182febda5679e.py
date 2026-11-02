import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def get_max_mult(S):
        if not S:
            return 0
        sums = {}
        S_list = list(S)
        for i in S_list:
            for j in S_list:
                s = i + j
                sums[s] = sums.get(s, 0) + 1
        return max(sums.values())

    def greedy_build(seed_set, max_size=12):
        S = set(seed_set)
        for _ in range(max_size):
            best_x = -1
            best_max_mult = float('inf')
            
            for x in range(N):
                if x in S:
                    continue
                S_trial = S | {x}
                m = get_max_mult(S_trial)
                # We want to minimize M, but also increase K.
                # Score is roughly K/M for equal weights.
                # Greedy: minimize M. If tie, pick smallest x? Or maybe largest K/M.
                # Let's stick to minimizing M first, then K/M.
                if m < best_max_mult:
                    best_max_mult = m
                    best_x = x
                elif m == best_max_mult:
                    # If M is same, adding x increases K, so K/M increases.
                    # So any x that doesn't increase M is good.
                    # Pick smallest x to keep numbers small? Or just first found.
                    if best_x == -1:
                        best_x = x
            
            if best_x != -1:
                S.add(best_x)
            else:
                break
        return S

    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate diverse initial candidates using Sidon-like sets
    seeds = [
        [0],
        [0, 1],
        [0, 1, 4],
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23, 30],
        [1, 2, 5, 11, 21, 34],
        [0, 1, 3, 7, 12, 20, 31], # Another Sidon-ish
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        S = greedy_build(seed)
        if not S:
            continue
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0 / len(S)
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Also try random sparse sets to diversify
    np.random.seed(42)
    for _ in range(1000):
        if time.time() > deadline - 20:
            break
        k = np.random.randint(3, 15)
        S = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[S] = 1.0 / k
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 2. Local Search: Set Swaps
    # Extract support from best_h
    S_best = set(np.nonzero(best_h)[0])
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try swapping each element in S_best with an element outside
        for i in list(S_best):
            if time.time() > deadline - 10:
                break
            for j in range(N):
                if j in S_best:
                    continue
                S_trial = S_best.copy()
                S_trial.remove(i)
                S_trial.add(j)
                
                h_trial = np.zeros(N)
                for k in S_trial:
                    h_trial[k] = 1.0 / len(S_trial)
                
                s_trial = score(h_trial)
                if s_trial > current_score + 1e-9:
                    S_best = S_trial
                    best_h = h_trial
                    current_score = s_trial
                    improved = True
                    # Restart inner loop logic? Or just continue? 
                    # Since S_best changed, the 'i' we are iterating over might be invalid now if it was removed.
                    # But 'i' is from list(S_best) which is a snapshot. 
                    # If i was removed, it's not in S_best anymore, but we are just checking if j is in S_best.
                    # It's safer to break out and restart the outer loop to be safe with indices.
                    break
            if improved:
                break # Break outer loop as well to restart with new S_best

    # 3. Continuous Optimization: Gradient Ascent on Amplitudes
    # We now have a good support set S_best.
    # Let's optimize the amplitudes h_i for i in S_best.
    # We can use a simple iterative method or scipy if available, but let's stick to numpy for speed and reliability.
    # Actually, scipy is allowed. Let's use scipy.optimize.minimize on the negative log ratio.
    
    try:
        from scipy.optimize import minimize
        
        support = list(S_best)
        k = len(support)
        if k == 0:
            return list(best_h)
            
        # Initial amplitudes: uniform
        x0 = np.ones(k) / k
        
        def neg_log_ratio(x):
            # x are amplitudes for the support indices
            # Enforce non-negativity? minimize might go negative. 
            # We can use bounds.
            h_full = np.zeros(N)
            h_full[support] = x
            h_full = np.maximum(h_full, 0) # Clip negatives, but gradient won't handle this well.
            # Better to use softplus or just rely on bounds.
            
            l1 = np.sum(h_full)
            if l1 < 1e-10:
                return 0.0
            l2_sq = np.sum(h_full**2)
            conv = np.convolve(h_full, h_full, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-10:
                return 0.0
            
            ratio = l2_sq / (l1 * max_conv)
            if ratio <= 0:
                return 0.0
            return -np.log(ratio)
        
        def jac_neg_log_ratio(x):
            # Numerical gradient is safer and N is small (50)
            eps = 1e-6
            grad = np.zeros(k)
            f0 = neg_log_ratio(x)
            for i in range(k):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = neg_log_ratio(x_plus)
                grad[i] = (f_plus - f0) / eps
            return grad
        
        # Run optimization
        # We want to maximize ratio, so minimize -log(ratio)
        # Bounds: amplitudes > 0
        bounds = [(1e-8, None) for _ in range(k)]
        
        # Use L-BFGS-B
        res = minimize(neg_log_ratio, x0, method='L-BFGS-B', jac=jac_neg_log_ratio, 
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success or res.fun < neg_log_ratio(x0):
            x_opt = res.x
            h_opt = np.zeros(N)
            h_opt[support] = x_opt
            h_opt = np.maximum(h_opt, 0)
            
            s_opt = score(h_opt)
            if s_opt > current_score:
                best_h = h_opt
                current_score = s_opt
                
    except Exception as e:
        pass # Fallback to discrete solution if scipy fails or times out

    # 4. Final Local Search on Amplitudes (Coordinate Ascent)
    # If scipy didn't improve or isn't available, do manual coordinate ascent
    h = best_h.copy()
    support = list(np.nonzero(h)[0])
    
    # Normalize to L1=1 to simplify? No, ratio is scale invariant.
    # Just optimize as is.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_score = score(h)
        for idx in support:
            if time.time() > deadline - 2:
                break
            # Try scaling h[idx]
            # Optimal h[idx] given others?
            # Let h = h_others + h_idx * e_idx
            # This is complex. Let's just do a simple line search.
            h_old = h[idx]
            if h_old < 1e-10:
                continue
            
            # Try multiplying by factors
            for factor in [0.5, 0.8, 1.2, 1.5, 2.0, 0.1]:
                h_trial = h.copy()
                h_trial[idx] = h_old * factor
                s_trial = score(h_trial)
                if s_trial > current_score + 1e-9:
                    h = h_trial
                    current_score = s_trial
                    improved = True
                    break # Move to next index
            
    return list(h)
