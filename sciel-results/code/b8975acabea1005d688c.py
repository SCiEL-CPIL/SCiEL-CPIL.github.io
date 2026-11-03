import numpy as np
import time
import math

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank matching
    # Heavy flows to short distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # Cost = sum(a[i,j] * b[p[i], p[j]])
    # We can use np.einsum or just loops if n is small, but n=150 is small enough for vectorized ops?
    # B_perm = b[np.ix_(p, p)]
    # cost = np.sum(a * B_perm)
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    # 2. Simulated Annealing
    T = 100000.0  # Initial temperature
    T_min = 1.0
    alpha = 0.999  # Cooling factor
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Precompute nothing? Just use a and b directly.
    # To speed up dot product, we can extract rows.
    
    while time.time() - start_time < time_limit:
        # Pick two random distinct indices
        i = np.random.randint(n)
        k = np.random.randint(n)
        if i == k:
            continue
        
        u = p[i]
        v = p[k]
        
        # Calculate Delta C
        # S = dot(A[i, :] - A[k, :], B[v, :] - B[u, :])
        diff_a = a[i, :] - a[k, :]
        diff_b = b[v, :] - b[u, :]
        S = np.dot(diff_a, diff_b)
        
        # Correction terms
        # Term 1: - (A[i,i] - A[k,i]) * (B[v,u] - B[u,u])
        # Term 2: - (A[i,k] - A[k,k]) * (B[v,v] - B[u,v])
        # Term 3: + (A[i,k] - A[k,i]) * (B[v,u] - B[u,v])
        # Term 4: + (A[i,i] - A[k,k]) * (B[v,v] - B[u,u])
        
        t1 = -(a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
        t2 = -(a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
        t3 = (a[i, k] - a[k, i]) * (b[v, u] - b[u, v])
        t4 = (a[i, i] - a[k, k]) * (b[v, v] - b[u, u])
        
        delta_c = S + t1 + t2 + t3 + t4
        
        # Acceptance
        if delta_c < 0 or np.random.random() < math.exp(-delta_c / T):
            p[i], p[k] = p[k], p[i]
            cost += delta_c
        
        # Cool
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart or continue with low T
            # If we hit T_min, we are in local search phase.
            # We can keep going until time limit.
            
    return p.tolist()
