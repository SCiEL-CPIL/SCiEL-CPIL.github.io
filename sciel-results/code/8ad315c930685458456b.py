import numpy as np
import scipy.optimize as opt
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def calculate_ratio(h):
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        c = np.convolve(h_norm, h_norm)
        peak = np.max(c)
        if peak < 1e-10:
            return 0.0
        return l2_sq / peak

    def find_max_sidon_set(N, max_attempts=1000):
        best_set = set()
        for _ in range(max_attempts):
            S = set()
            sums = set()
            # Greedy with randomization
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for cand in candidates:
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
            
            if len(S) > len(best_set):
                best_set = S
                
        return list(best_set)

    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_ratio = calculate_ratio(best_h)

    # Find the largest Sidon set
    sidon_set = find_max_sidon_set(N, max_attempts=2000)
    
    if len(sidon_set) > 0:
        h_sidon = np.zeros(N)
        for i in sidon_set:
            h_sidon[i] = 1.0 / len(sidon_set)
        
        ratio_sidon = calculate_ratio(h_sidon)
        if ratio_sidon > best_ratio:
            best_ratio = ratio_sidon
            best_h = h_sidon

        # Optimize weights on the Sidon set
        K = len(sidon_set)
        if K > 0:
            def objective(x):
                h_full = np.zeros(N)
                for i, idx in enumerate(sidon_set):
                    h_full[idx] = x[i]
                return -calculate_ratio(h_full)

            x0 = np.ones(K) / K
            bounds = [(1e-6, None)] * K
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 
                     'jac': lambda x: np.ones(K)})
            
            try:
                res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 500, 'ftol': 1e-12})
                if res.success:
                    x_opt = res.x
                    s = np.sum(x_opt)
                    if s > 0:
                        x_opt = x_opt / s
                    h_full = np.zeros(N)
                    for i, idx in enumerate(sidon_set):
                        h_full[idx] = x_opt[i]
                    ratio_opt = calculate_ratio(h_full)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_h = h_full
            except Exception:
                pass

    # Try to improve by checking if we can add more points or change the set
    # The Sidon set found might not be optimal in terms of ratio if weights are adjusted
    # But uniform Sidon is usually very good.

    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
