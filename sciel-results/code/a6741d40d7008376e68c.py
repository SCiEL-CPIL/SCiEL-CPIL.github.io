import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def score(h):
        h = np.asarray(h, dtype=float)
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

    # Pre-compute sum multiplicity helper for fast Sidon check
    # A set S has max sum multiplicity M if the max count of pairs (i,j) with i+j=s is M.
    # We want to minimize M.

    def max_sum_mult(S):
        if len(S) == 0:
            return 0
        counts = {}
        S_list = list(S)
        for i in S_list:
            for j in S_list:
                s = i + j
                counts[s] = counts.get(s, 0) + 1
        return max(counts.values())

    def local_search_support(S_init, max_iters=2000):
        """
        Simulated annealing / hill climbing to minimize max_sum_mult.
        """
        S = set(S_init)
        k = len(S)
        current_m = max_sum_mult(S)
        best_m = current_m
        best_S = S.copy()
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Pick a random element to remove
            i = random.choice(list(S))
            # Pick a random element to add
            candidates = [x for x in range(N) if x not in S]
            if not candidates:
                break
            j = random.choice(candidates)
            
            S_new = S.copy()
            S_new.remove(i)
            S_new.add(j)
            
            m_new = max_sum_mult(S_new)
            
            # Accept if better, or with probability exp(-(delta)/T)
            delta = m_new - current_m
            if delta < 0 or (random.random() < np.exp(-delta / T)):
                S = S_new
                current_m = m_new
                if current_m < best_m:
                    best_m = current_m
                    best_S = S.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_S

    def optimize_amplitudes(S, time_limit=5.0):
        """
        Optimize amplitudes for a given support S using coordinate descent
        on the objective: maximize L2^2 / (L1 * max_conv).
        Since L1 and max_conv are non-linear, we use a penalty/iterative approach.
        Actually, let's just use scipy.optimize.minimize on the ratio directly,
        but handle the non-smoothness by using a smoothed max or just relying on
        the local nature of the max.
        """
        if not S:
            return np.zeros(N), 0.0
        
        support = list(S)
        k = len(support)
        
        # Initialize with uniform
        x0 = np.ones(k) / k
        
        def objective(x):
            h_full = np.zeros(N)
            h_full[support] = x
            return -score(h_full)
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', bounds=[(1e-8, None)]*k, 
                           options={'maxiter': 50, 'ftol': 1e-8})
            if res.fun < objective(x0):
                x_opt = np.maximum(res.x, 0)
                h_opt = np.zeros(N)
                h_opt[support] = x_opt
                return h_opt, score(h_opt)
        except Exception:
            pass
        
        # Fallback: uniform
        h_uniform = np.zeros(N)
        for i in support:
            h_uniform[i] = 1.0 / k
        return h_uniform, score(h_uniform)

    best_h = np.zeros(N)
    best_score = 0.0

    # 1. Generate diverse initial supports
    np.random.seed(42)
    random.seed(42)

    candidate_supports = []

    # Structured Sidon-like sets (Golomb rulers / modular)
    # Small k: k=5..15
    for k in range(5, 16):
        # Generate random Sidon-like sets
        for _ in range(5):
            S = set()
            while len(S) < k:
                S.add(np.random.randint(0, N))
            candidate_supports.append(list(S))
    
    # Also try some structured ones
    # Arithmetic progression is bad (high multiplicity)
    # Random is okay
    # Let's add some specific known good sets if possible, or just random
    
    # 2. Process candidates
    processed = []
    
    for S_init in candidate_supports:
        if time.time() > deadline - 40:
            break
        
        # Local search to improve support
        S_improved = local_search_support(set(S_init), max_iters=500)
        
        # Score uniform on improved support
        h_uniform = np.zeros(N)
        for i in S_improved:
            h_uniform[i] = 1.0 / len(S_improved)
        s_uniform = score(h_uniform)
        
        processed.append((s_uniform, S_improved))
        
    # Sort by uniform score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Optimize amplitudes for top supports
    top_k = 15
    for s_proxy, S in processed[:top_k]:
        if time.time() > deadline - 10:
            break
        h_opt, s_opt = optimize_amplitudes(S, time_limit=3.0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # If nothing better found, return single spike
    if best_score <= 1.0:
        h = [0.0] * N
        h[N // 2] = 1.0
        return h

    return list(best_h)
