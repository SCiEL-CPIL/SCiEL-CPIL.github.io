import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # Helper to calculate cost
    def calc_cost(p):
        # p is a list of locations for each facility
        # cost = sum a[i,j] b[p[i], p[j]]
        # Using matrix multiplication: sum_i sum_j a[i,j] b[p[i], p[j]]
        # This is equivalent to sum over i of (a[i, :] * b[p[i], p]) dot 1? No.
        # It is sum_i sum_j a[i,j] b[p[i], p[j]]
        # We can do: for each i, sum_j a[i,j] b[p[i], p[j]]
        # Vectorized: 
        # Let B_p be the matrix where row i is b[p[i], :]
        # Then cost = sum(a * B_p)
        # B_p = b[np.array(p), :]
        return np.sum(a * b[np.array(p), :])

    # Helper to calculate swap delta
    def swap_delta(p, i, j):
        # p is numpy array
        u = p[i]
        v = p[j]
        
        # If u == v, delta is 0 (should not happen in valid permutation)
        if u == v:
            return 0.0
            
        diff_uv = b[u, :] - b[v, :]
        
        # Sum over k != i, j of a[i,k] * (b[u, p[k]] - b[v, p[k]])
        # We compute full sum and subtract k=i and k=j terms
        # Full sum: dot(a[i, :], diff_uv[p])
        # Term k=i: a[i,i] * diff_uv[p[i]] = a[i,i] * (b[u, u] - b[v, u])
        # Term k=j: a[i,j] * diff_uv[p[j]] = a[i,j] * (b[u, v] - b[v, v])
        
        sum_i = np.dot(a[i, :], diff_uv[p])
        sum_i -= a[i, i] * diff_uv[u]
        sum_i -= a[i, j] * diff_uv[v]
        
        # Sum over k != i, j of a[j,k] * (b[v, p[k]] - b[u, p[k]])
        # = - Sum over k != i, j of a[j,k] * (b[u, p[k]] - b[v, p[k]])
        # Full sum: dot(a[j, :], -diff_uv[p])
        # Term k=i: a[j,i] * (-diff_uv[p[i]]) = -a[j,i] * diff_uv[u]
        # Term k=j: a[j,j] * (-diff_uv[p[j]]) = -a[j,j] * diff_uv[v]
        
        sum_j = np.dot(a[j, :], -diff_uv[p])
        sum_j -= a[j, i] * (-diff_uv[u])
        sum_j -= a[j, j] * (-diff_uv[v])
        
        # Cross terms for i, j pair
        # Original: a[i,j] b[u,v] + a[j,i] b[v,u]
        # New: a[i,j] b[v,u] + a[j,i] b[u,v]
        # Delta = a[i,j](b[v,u] - b[u,v]) + a[j,i](b[u,v] - b[v,u])
        cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        return sum_i + sum_j + cross

    # Initial Solution: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # Parallel Tempering Setup
    num_replicas = 4
    # Temperatures: Geometric progression
    # T_min should be low enough to converge, T_max high enough to explore
    # Estimate initial energy
    E0 = calc_cost(p_init.tolist())
    # Heuristic T range
    T_min = 10.0
    T_max = 1000.0
    ratios = (T_max / T_min) ** (1.0 / (num_replicas - 1))
    temps = np.array([T_min * (ratios ** i) for i in range(num_replicas)])
    
    # Initialize replicas
    # Replica 0: Best initial guess
    # Others: Random permutations or perturbed
    replicas_p = [None] * num_replicas
    replicas_E = [0.0] * num_replicas
    
    replicas_p[0] = p_init.copy()
    replicas_E[0] = E0
    
    for r in range(1, num_replicas):
        # Random permutation
        pr = np.random.permutation(n)
        replicas_p[r] = pr
        replicas_E[r] = calc_cost(pr.tolist())
        
    best_p = replicas_p[0].copy()
    best_E = replicas_E[0]
    
    # Main Loop
    while time.time() - start_time < time_limit:
        # 1. Metropolis Step for each replica
        for r in range(num_replicas):
            p = replicas_p[r]
            E = replicas_E[r]
            T = temps[r]
            
            # Pick random i, j
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = swap_delta(p, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                p[i], p[j] = p[j], p[i]
                E += delta
                replicas_p[r] = p
                replicas_E[r] = E
                
                if E < best_E:
                    best_E = E
                    best_p = p.copy()
                    
        # 2. Exchange Step
        # Try to swap between adjacent replicas (r, r+1)
        # Probability = min(1, exp(-(E_r - E_{r+1})/T_r + (E_r - E_{r+1})/T_{r+1}))
        # Actually, standard formula:
        # P = min(1, exp( - (E_high - E_low) * (1/T_high - 1/T_low) ))
        # where T_high > T_low
        for r in range(num_replicas - 1):
            r_low = r
            r_high = r + 1
            T_low = temps[r_low]
            T_high = temps[r_high]
            E_low = replicas_E[r_low]
            E_high = replicas_E[r_high]
            
            # Ensure T_high > T_low (which is true by construction)
            # If E_high > E_low, the exponent is negative, prob < 1
            # If E_high < E_low, the exponent is positive, prob = 1
            
            # dE = E_high - E_low
            dE = E_high - E_low
            # Exponent = -dE * (1/T_high - 1/T_low)
            # Note: 1/T_high < 1/T_low, so (1/T_high - 1/T_low) is negative.
            # If dE > 0 (high temp has higher energy), exponent is positive -> Prob = 1?
            # Wait.
            # Standard PT:
            # Swap if:
            # E_low + dE (at high T) < E_high (at low T) -> Always swap
            # Else: Prob = exp( -(E_high - E_low)/T_high + (E_high - E_low)/T_low )
            # = exp( (E_low - E_high) (1/T_low - 1/T_high) )? No.
            # Let's use the rigorous form:
            # Ratio = (Z_high / Z_low) ... 
            # Acceptance probability A = min(1, exp( - (E_high - E_low) / T_high + (E_high - E_low) / T_low ))
            # = min(1, exp( (E_low - E_high) / T_low + (E_high - E_low) / T_high ))
            # = min(1, exp( (E_low - E_high) * (1/T_low - 1/T_high) ))
            
            if dE > 0:
                # E_high > E_low
                # Term = (E_low - E_high) * (1/T_low - 1/T_high)
                # E_low - E_high is negative.
                # 1/T_low - 1/T_high is positive.
                # Product is negative. Exp is < 1.
                exponent = (E_low - E_high) * (1.0/T_low - 1.0/T_high)
                prob = np.exp(exponent)
            else:
                # E_high <= E_low
                # Term is positive or zero. Exp >= 1.
                prob = 1.0
                
            if np.random.random() < prob:
                # Swap permutations and energies
                replicas_p[r_low], replicas_p[r_high] = replicas_p[r_high], replicas_p[r_low]
                replicas_E[r_low], replicas_E[r_high] = replicas_E[r_high], replicas_E[r_low]
                
                # Update best if swapped replica is now better
                if replicas_E[r_low] < best_E:
                    best_E = replicas_E[r_low]
                    best_p = replicas_p[r_low].copy()
                if replicas_E[r_high] < best_E:
                    best_E = replicas_E[r_high]
                    best_p = replicas_p[r_high].copy()

    return best_p.tolist()
