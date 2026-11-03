import numpy as np
import scipy.optimize as opt
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def autoconv(h):
        return np.convolve(h, h)

    def calculate_ratio(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        c = autoconv(h_norm)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / peak

    def find_sidon_set(K, N, attempts=100):
        # Try to find a Sidon set of size K in [0, N)
        for _ in range(attempts):
            S = set()
            sums = set()
            # Greedy addition
            # Randomize order of candidates
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for cand in candidates:
                if len(S) == K:
                    break
                # Check if cand can be added
                conflict = False
                new_sums = set()
                
                # Check cand + cand
                if cand + cand in sums:
                    conflict = True
                else:
                    new_sums.add(cand + cand)
                    
                # Check cand + s for s in S
                if not conflict:
                    for s in S:
                        val = cand + s
                        if val in sums:
                            conflict = True
                            break
                        new_sums.add(val)
                
                if not conflict:
                    S.add(cand)
                    sums.update(new_sums)
            
            if len(S) == K:
                return list(S)
        return None

    def optimize_for_support(support):
        K = len(support)
        if K == 0:
            return None, 0.0
        
        def objective(x):
            # x are the weights for the support
            # Ensure non-negativity is handled by bounds
            h_full = np.zeros(N)
            for i, idx in enumerate(support):
                h_full[idx] = x[i]
            
            l1 = np.sum(h_full)
            if l1 < 1e-10:
                return 0.0 # Minimize -ratio, so return large positive? No, maximize ratio.
            
            h_norm = h_full / l1
            l2_sq = np.sum(h_norm**2)
            c = autoconv(h_norm)
            peak = np.max(c)
            if peak < 1e-10:
                return 0.0
            
            ratio = l2_sq / peak
            return -ratio # Minimize negative ratio

        def objective_grad(x):
            # Finite difference is probably fine for small K
            return None

        # Initial guess: uniform
        x0 = np.ones(K) / K
        
        # Bounds: x_i >= 0
        bounds = [(0, None)] * K
        
        # Constraint: sum(x) == 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 
                 'jac': lambda x: np.ones(K)})
        
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success:
                x_opt = res.x
                # Re-normalize to be safe
                s = np.sum(x_opt)
                if s > 0:
                    x_opt = x_opt / s
                else:
                    return None, 0.0
                
                h_full = np.zeros(N)
                for i, idx in enumerate(support):
                    h_full[idx] = x_opt[i]
                
                ratio = calculate_ratio(h_full)
                return h_full, ratio
        except Exception:
            pass
        
        return None, 0.0

    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_ratio = calculate_ratio(best_h)

    # Try different sizes of Sidon sets
    # Max size is roughly sqrt(N)
    max_K = int(np.sqrt(N)) + 2
    
    # Generate a pool of Sidon sets
    sidon_sets = []
    for K in range(2, max_K + 1):
        # Try to find multiple sets for each K
        found = 0
        attempts = 0
        while found < 5 and attempts < 200:
            S = find_sidon_set(K, N, attempts=1)
            if S is not None:
                # Check if already in pool (simple check)
                S_tuple = tuple(sorted(S))
                if S_tuple not in [tuple(sorted(s)) for s in sidon_sets]:
                    sidon_sets.append(S)
                    found += 1
            attempts += 1
            if time.time() > deadline - 20.0:
                break
        if time.time() > deadline - 20.0:
            break

    # Optimize for each Sidon set
    for S in sidon_sets:
        if time.time() > deadline - 10.0:
            break
            
        h_opt, ratio = optimize_for_support(S)
        if h_opt is not None and ratio > best_ratio:
            best_ratio = ratio
            best_h = h_opt

    # Also try a few random sparse supports (not necessarily Sidon)
    # In case Sidon restriction is too tight
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        K = random.randint(3, 10)
        S = random.sample(range(N), K)
        h_opt, ratio = optimize_for_support(S)
        if h_opt is not None and ratio > best_ratio:
            best_ratio = ratio
            best_h = h_opt

    # Final polish: If we have a good candidate, run a high-precision optimization on the full vector
    # restricted to the support of best_h
    support_best = np.where(best_h > 1e-6)[0]
    if len(support_best) > 0:
        def objective_full(x):
            h_full = np.zeros(N)
            for i, idx in enumerate(support_best):
                h_full[idx] = x[i]
            l1 = np.sum(h_full)
            if l1 < 1e-10:
                return 0.0
            h_norm = h_full / l1
            l2_sq = np.sum(h_norm**2)
            c = autoconv(h_norm)
            peak = np.max(c)
            if peak < 1e-10:
                return 0.0
            return -(l2_sq / peak)

        x0 = np.array([best_h[idx] for idx in support_best])
        s = np.sum(x0)
        x0 = x0 / s
        
        bounds = [(0, None)] * len(support_best)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 
                 'jac': lambda x: np.ones(len(support_best))})
        
        try:
            res = opt.minimize(objective_full, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 500, 'ftol': 1e-14})
            if res.success:
                x_opt = res.x
                s = np.sum(x_opt)
                if s > 0:
                    x_opt = x_opt / s
                h_full = np.zeros(N)
                for i, idx in enumerate(support_best):
                    h_full[idx] = x_opt[i]
                ratio = calculate_ratio(h_full)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_full
        except Exception:
            pass

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
