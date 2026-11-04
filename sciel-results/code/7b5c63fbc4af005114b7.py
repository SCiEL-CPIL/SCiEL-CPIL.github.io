import numpy as np
from scipy.optimize import linear_sum_assignment
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Initialization using Hungarian Algorithm on a surrogate cost matrix
    # Surrogate: Row sums of A (facility strength) vs Row sums of B (location centrality)
    # We want high strength facilities at low distance locations.
    row_sum_a = a.sum(axis=1)
    row_sum_b = b.sum(axis=1)
    
    # Cost matrix C[i, j] = row_sum_a[i] * row_sum_b[j]
    # We minimize this, so high row_sum_a should go with low row_sum_b.
    # linear_sum_assignment minimizes the sum of C[row, col].
    # So if row_sum_a[i] is large, we want row_sum_b[j] to be small.
    # This matches the logic.
    C = np.outer(row_sum_a, row_sum_b)
    
    # linear_sum_assignment returns (row_ind, col_ind)
    # row_ind is usually 0..n-1
    # col_ind[k] is the column assigned to row row_ind[k]
    # Since row_ind is 0..n-1, p = col_ind gives p[i] = location of facility i
    _, p = linear_sum_assignment(C)
    p = p.astype(np.int64)
    
    # Calculate initial cost
    def calc_cost(perm):
        # sum_{i,j} a[i,j] * b[perm[i], perm[j]]
        # Vectorized:
        # b_perm = b[np.ix_(perm, perm)]
        # return np.sum(a * b_perm)
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    # Estimate temperature
    # Max possible cost is roughly n^2 * max(a) * max(b)
    # Let's use a heuristic T_start
    T = np.std(a) * np.std(b) * n * 0.1
    if T == 0:
        T = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # Precompute some values for speed? 
    # a and b are fixed.
    
    while time.time() < deadline:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Calculate Delta
        # A_diff = a[i] - a[j]
        # B_diff = b[v] - b[u]
        # V = B_diff[p]
        # dot = np.dot(a[i] - a[j], V)
        # correction = -(a[i,i]-a[j,i])*(b[u,v]-b[u,u]) - (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        
        A_diff = a[i] - a[j]
        B_diff = b[v] - b[u]
        V = B_diff[p]
        
        dot = np.dot(A_diff, V)
        
        # Correction terms for k=i and k=j
        # term_i = -A_diff[i] * (b[v, p[i]] - b[u, p[i]]) = -A_diff[i] * (b[v, u] - b[u, u])
        # term_j = -A_diff[j] * (b[v, p[j]] - b[u, p[j]]) = -A_diff[j] * (b[v, v] - b[u, v])
        
        correction = -A_diff[i] * (b[v, u] - b[u, u]) - A_diff[j] * (b[v, v] - b[u, v])
        
        delta = dot + correction
        
        # Acceptance criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Swap i and j in p
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_p.tolist()
