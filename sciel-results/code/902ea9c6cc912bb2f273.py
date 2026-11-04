import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initial Solution: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def get_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
    
    best_cost = get_cost(p)
    best_p = p.copy()
    
    def local_search(p):
        current_cost = get_cost(p)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                ai = a[i]
                ui = p[i]
                for j in range(i + 1, n):
                    aj = a[j]
                    uj = p[j]
                    
                    diff_b = b[uj] - b[ui]
                    delta = np.dot(ai - aj, diff_b[p])
                    
                    term1 = (ai[i] - aj[i]) * (b[uj, ui] - b[ui, ui])
                    term2 = (ai[j] - aj[j]) * (b[uj, uj] - b[ui, uj])
                    
                    delta -= term1
                    delta -= term2
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                
            if time.time() > deadline:
                break
                
        return p, current_cost

    p, best_cost = local_search(p)
    best_p = p.copy()
    
    while time.time() < deadline:
        p_current = best_p.copy()
        
        k = np.random.randint(5, 15)
        for _ in range(k):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_current[i], p_current[j] = p_current[j], p_current[i]
        
        p_new, cost_new = local_search(p_current)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            
        if time.time() > deadline:
            break
            
    return best_p.tolist()
