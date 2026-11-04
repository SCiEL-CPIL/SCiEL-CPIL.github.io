import numpy as np
import time
import random

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initial Solution: Rank Matching
    # Sort facilities by sum of row/col in A (heavy flow)
    # Sort locations by sum of row/col in B (distance centrality)
    # Heavy flows should go to central (low distance) locations.
    
    a_sums = a.sum(axis=1) + a.sum(axis=0)
    b_sums = b.sum(axis=1) + b.sum(axis=0)
    
    # Facilities with highest flow sums
    heavy_facs = np.argsort(-a_sums, kind='stable')
    # Locations with lowest distance sums (most central)
    central_locs = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facs] = central_locs
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # Efficiently: sum over i,j
    # We can compute this as trace(A @ B_p) where B_p is permuted B?
    # B_p[i,j] = B[p[i], p[j]]
    # This is equivalent to sum_{i,j} A[i,j] B[p[i], p[j]]
    
    def calc_cost(p):
        # p is permutation of locations for facilities
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as:
        # Let B_p[i,j] = b[p[i], p[j]]
        # Cost = np.sum(a * B_p)
        # Constructing B_p is O(n^2).
        # For n=100, this is 10,000 ops. Very fast.
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for fast delta calculation
    # We will use the vectorized delta formula derived above.
    
    def get_delta(i, j, p, a, b):
        # Swap facilities i and j
        u = p[i]
        v = p[j]
        
        # FullSum = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Note: b[v, p[k]] means row v of b, columns p[k]
        # p is array of locations.
        
        # Vectorized:
        # diff_a = a[i, :] - a[j, :]
        # diff_b = b[v, :] - b[u, :]
        # But diff_b needs to be indexed by p[k].
        # So we need b[v, p] and b[u, p].
        
        row_v_p = b[v, p]
        row_u_p = b[u, p]
        
        diff_a = a[i, :] - a[j, :]
        diff_b = row_v_p - row_u_p
        
        full_sum = np.dot(diff_a, diff_b)
        
        # Subtract terms for k=i and k=j
        # Term_i = (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Term_j = (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        d_sum = full_sum - term_i - term_j
        
        # Interaction term D_ij
        d_ij = a[i, j]*b[v, u] + a[j, i]*b[u, v] - a[i, j]*b[u, v] - a[j, i]*b[v, u]
        
        return d_sum + d_ij

    # Simulated Annealing
    # Parameters
    T = 10000.0
    T_min = 1.0
    alpha = 0.999
    steps_per_temp = 500
    
    while time.time() < deadline - 10.0:  # Leave 10s for local search
        for _ in range(steps_per_temp):
            if time.time() > deadline - 10.0:
                break
            
            # Pick random i, j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = get_delta(i, j, p, a, b)
            
            if delta < 0:
                # Accept
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # Accept with probability exp(-delta / T)
                if T > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            break

    # Local Search Polish
    # 2-opt style: try all swaps, accept if improves.
    # Repeat until no improvement or time runs out.
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                delta = get_delta(i, j, p, a, b)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    # Restart loops? Or just continue?
                    # Continuing is faster, but might miss better swaps.
                    # For QAP, 2-opt with restart is standard.
                    # However, given time constraints, a single pass with acceptance
                    # is a "steepest descent" or "first improvement" depending on impl.
                    # Let's stick to first improvement found in the scan, then restart scan.
                    break
            if improved:
                break # Restart outer loop
    
    return best_p.tolist()
