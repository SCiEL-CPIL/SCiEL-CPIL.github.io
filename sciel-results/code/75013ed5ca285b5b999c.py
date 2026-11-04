import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initial solution: Rank matching
    # Heavy flows (high row sum in a) should go to short distances (low row sum in b)
    # Note: b is distance matrix, lower is better.
    # a is flow matrix, higher is worse if distance is high.
    # So match high a-row-sum with low b-row-sum.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind='stable')
    # Sort locations by ascending distance sum (most central)
    central = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # Cost = sum a[i, j] * b[p[i], p[j]]
    # Vectorized: np.sum(a * b[np.ix_(p, p)])
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for safety
    
    T = 1000.0  # Initial temperature
    cooling = 0.999  # Cooling rate
    
    # Precompute for speed? 
    # We will compute delta on the fly.
    
    while time.time() - start_time < time_limit:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Calculate Delta Cost
        # Terms for k != i, j:
        # sum_k (a[i, k] * (b[v, p[k]] - b[u, p[k]]) + a[k, i] * (b[p[k], v] - b[p[k], u]))
        
        # b[v, p] is the row v of b, indexed by p
        # b[u, p] is the row u of b, indexed by p
        diff_b_row = b[v, p] - b[u, p]
        
        # Sum a[i, k] * diff_b_row[k] for k != i, j
        # np.dot(a[i, :], diff_b_row) gives sum for all k
        # Subtract k=i and k=j terms
        sum_ai = np.dot(a[i, :], diff_b_row)
        sum_ai -= a[i, i] * diff_b_row[i]
        sum_ai -= a[i, j] * diff_b_row[j]
        
        # Sum a[k, i] * (b[p[k], v] - b[p[k], u])
        # b[:, v] is column v of b
        # b[:, u] is column u of b
        diff_b_col = b[p, v] - b[p, u]
        
        sum_ki = np.dot(a[:, i], diff_b_col)
        sum_ki -= a[i, i] * diff_b_col[i]
        sum_ki -= a[j, i] * diff_b_col[j]
        
        # Terms for i, j directly
        # Old: a[i, j] * b[u, v] + a[j, i] * b[v, u] + a[i, i] * b[u, u] + a[j, j] * b[v, v]
        # New: a[i, j] * b[v, u] + a[j, i] * b[u, v] + a[i, i] * b[v, v] + a[j, j] * b[u, u]
        
        delta_ij = a[i, j] * (b[v, u] - b[u, v])
        delta_ji = a[j, i] * (b[u, v] - b[v, u])
        delta_ii = a[i, i] * (b[v, v] - b[u, u])
        delta_jj = a[j, j] * (b[u, u] - b[v, v])
        
        delta = sum_ai + sum_ki + delta_ij + delta_ji + delta_ii + delta_jj
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept move
            p[i], p[j] = v, u
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        if T < 1e-6:
            T = 1e-6

    return best_p.tolist()
