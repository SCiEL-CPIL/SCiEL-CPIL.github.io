import numpy as np
import scipy.optimize as opt
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit

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
        for _ in range(attempts):
            S = set()
            sums = set()
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for cand in candidates:
                if len(S) == K:
                    break
                conflict = False
                new_sums = set()
                
                if cand + cand in sums:
                    conflict = True
                else:
                    new_sums.add(cand + cand)
                    
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
            h_full = np.zeros(N)
            for i, idx in enumerate(support):
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
            
            ratio = l2_sq / peak
            return -ratio

        x0 = np.ones(K) / K
        bounds = [(0, None)] * K
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 
                 'jac': lambda x: np.ones(K)})
        
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success:
                x_opt = res.x
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

    max_K = int(np.sqrt(N)) + 3 # Slightly larger
    
    sidon_sets = []
    for K in range(2, max_K + 1):
        found = 0
        attempts = 0
        while found < 5 and attempts < 200:
            S = find_sidon_set(K, N, attempts=1)
            if S is not None:
                S_tuple = tuple(sorted(S))
                if S_tuple not in [tuple(sorted(s)) for s in sidon_sets]:
                    sidon_sets.append(S)
                    found += 1
            attempts += 1
            if time.time() > deadline - 20.0:
                break
        if time.time() > deadline - 20.0:
            break

    for S in sidon_sets:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Evaluate uniform distribution on this support
        h_uniform = np.zeros(N)
        for idx in S:
            h_uniform[idx] = 1.0 / len(S)
        
        ratio_uniform = calculate_ratio(h_uniform)
        if ratio_uniform > best_ratio:
            best_ratio = ratio_uniform
            best_h = h_uniform

        # 2. Try optimization
        h_opt, ratio_opt = optimize_for_support(S)
        if h_opt is not None and ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_h = h_opt

    # Final polish: Full vector optimization
    if time.time() < deadline - 5.0:
        def objective_full(h):
            l1 = np.sum(h)
            if l1 < 1e-10:
                return 0.0
            h_norm = h / l1
            l2_sq = np.sum(h_norm**2)
            c = autoconv(h_norm)
            peak = np.max(c)
            if peak < 1e-10:
                return 0.0
            return -(l2_sq / peak)

        cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 
                 'jac': lambda h: np.ones(N)})
        
        try:
            res = opt.minimize(objective_full, best_h, method='SLSQP', bounds=[(0, None)]*N, constraints=cons,
                               options={'maxiter': 100, 'ftol': 1e-10})
            if res.success:
                h_final = res.x
                h_final = np.clip(h_final, 0, None)
                s = np.sum(h_final)
                if s > 0:
                    h_final = h_final / s
                    ratio_final = calculate_ratio(h_final)
                    if ratio_final > best_ratio:
                        best_ratio = ratio_final
                        best_h = h_final
        except Exception:
            pass

    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
