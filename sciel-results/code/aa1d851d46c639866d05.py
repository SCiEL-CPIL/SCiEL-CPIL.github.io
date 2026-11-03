import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute B_p for fast access in 2-opt
    B_p = b[np.ix_(p, p)]
    
    def two_opt(p, cost, B_p, max_passes=100):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline:
                return p, cost, B_p
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost, B_p
                for j in range(i + 1, n):
                    # Vectorized delta calculation using B_p
                    a_ij = a[i, :] - a[j, :]
                    a_ki = a[:, i] - a[:, j]
                    
                    s1 = np.dot(a_ij, B_p[j, :] - B_p[i, :])
                    s2 = np.dot(a_ki, B_p[:, j] - B_p[:, i])
                    
                    # Subtractions
                    s1 -= (a[i, i] - a[j, i]) * (B_p[j, i] - B_p[i, i])
                    s1 -= (a[i, j] - a[j, j]) * (B_p[j, j] - B_p[i, j])
                    
                    s2 -= (a[i, i] - a[i, j]) * (B_p[i, j] - B_p[i, i])
                    s2 -= (a[j, i] - a[j, j]) * (B_p[j, j] - B_p[j, i])
                    
                    # Cross
                    cross = (a[i, j] - a[j, i]) * (B_p[j, i] - B_p[i, j])
                    
                    delta = s1 + s2 + cross
                    
                    if delta < 0:
                        # Swap i, j in p
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        # Update B_p
                        B_p[i, :], B_p[j, :] = B_p[j, :], B_p[i, :].copy()
                        B_p[:, i], B_p[:, j] = B_p[:, j].copy(), B_p[:, i]
                        improved = True
                        break
                if improved:
                    break
        return p, cost, B_p

    # ILS Loop
    while time.time() < deadline:
        p, current_cost, B_p = two_opt(p, current_cost, B_p, max_passes=50)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() > deadline:
            break
        
        # Perturbation: 10 random swaps
        for _ in range(10):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Update cost and B_p
        current_cost = compute_cost(p)
        B_p = b[np.ix_(p, p)]
        
    return best_p.tolist()
