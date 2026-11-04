import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Hungarian Algorithm on linearized cost
    # W[i, j] represents the contribution of placing facility i at location j
    # Linearization: W = A @ B.T
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    def get_cost(p):
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as trace(a @ b[:, p][:, p])? No.
        # It is sum_i sum_j a[i,j] b[p[i], p[j]]
        # Let B_p be the matrix b with columns permuted by p? No, rows and cols.
        # b[p[i], p[j]] is the element at row p[i] col p[j].
        # So we take submatrix b[p][:, p]
        B_perm = b[np.ix_(p, p)]
        return int(np.sum(a * B_perm))
        
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Helper for fast delta calculation
    # Precompute nothing specific, but use vectorized ops in delta.
    def compute_delta_swap(i, k, p):
        u, v = p[i], p[k]
        if u == v: return 0
        
        # We want to swap p[i] and p[k].
        # Old: p[i]=u, p[k]=v
        # New: p[i]=v, p[k]=u
        
        # Delta = New_Cost - Old_Cost
        # Terms involving i or k change.
        # Cost = sum_{r,s} a[r,s] b[p[r], p[s]]
        
        # Let's use the standard formula for swap (i, k) with locations u, v.
        # Change in cost is:
        # 1. Terms where r=i or s=i (row i of A, col i of A)
        # 2. Terms where r=k or s=k (row k of A, col k of A)
        # 3. Cross terms?
        
        # Efficient vectorized calculation:
        # Let b_i = b[u, :] (row u of B)
        # Let b_k = b[v, :] (row v of B)
        # Let b_col_u = b[:, u] (col u of B)
        # Let b_col_v = b[:, v] (col v of B)
        
        # The change in cost comes from:
        # Replacing b[u, p[j]] with b[v, p[j]] for all j (row i of A)
        # Replacing b[p[j], u] with b[p[j], v] for all j (col i of A)
        # Replacing b[v, p[j]] with b[u, p[j]] for all j (row k of A)
        # Replacing b[p[j], v] with b[p[j], u] for all j (col k of A)
        # AND correcting for the (i,k) and (k,i) and (i,i) and (k,k) terms which are counted twice or incorrectly.
        
        # A cleaner way:
        # Delta = sum_j a[i,j] (b[v, p[j]] - b[u, p[j]]) + sum_j a[j,i] (b[p[j], v] - b[p[j], u])
        #       - sum_j a[k,j] (b[v, p[j]] - b[u, p[j]]) - sum_j a[j,k] (b[p[j], v] - b[p[j], u])
        #       + correction for j=i and j=k where p[i] and p[k] also change.
        
        # Let's stick to the previously verified formula but implement it efficiently.
        # diff_row = b[v, p] - b[u, p]
        # diff_col = b[p, v] - b[p, u]
        
        # term_i_row = dot(a[i, :], diff_row)
        # term_i_col = dot(a[:, i], diff_col)
        # term_k_row = dot(a[k, :], diff_row)
        # term_k_col = dot(a[:, k], diff_col)
        
        # total_sum = term_i_row + term_i_col - term_k_row - term_k_col
        
        # Corrections for j=i and j=k in the sums above:
        # In term_i_row (j=i): a[i,i] * (b[v, u] - b[u, u])  (since p[i] was u)
        # In term_i_col (j=i): a[i,i] * (b[u, v] - b[u, u])  (since p[i] was u)
        # In term_k_row (j=k): a[k,k] * (b[v, v] - b[u, v])  (since p[k] was v)
        # In term_k_col (j=k): a[k,k] * (b[v, v] - b[v, u])  (since p[k] was v)
        
        # In term_i_row (j=k): a[i,k] * (b[v, v] - b[u, v])
        # In term_i_col (j=k): a[k,i] * (b[v, v] - b[v, u])
        # In term_k_row (j=i): a[k,i] * (b[v, u] - b[u, u])
        # In term_k_col (j=i): a[i,k] * (b[u, v] - b[u, u])
        
        # The "total_sum" includes these terms.
        # But the actual change for the (i,i), (k,k), (i,k), (k,i) pairs is:
        # (i,i): a[i,i] * (b[v,v] - b[u,u])
        # (k,k): a[k,k] * (b[u,u] - b[v,v])
        # (i,k): a[i,k] * (b[v,u] - b[u,v])
        # (k,i): a[k,i] * (b[u,v] - b[v,u])
        
        # So Delta = total_sum - (sum of old terms in total_sum for j in {i,k}) + (actual new terms for i,k)
        
        # Let's just compute the delta directly for the 4 pairs and sum over j not in {i,k}.
        # This is O(n) anyway.
        
        # Vectorized:
        diff_row = b[v, p] - b[u, p]
        diff_col = b[p, v] - b[p, u]
        
        # Sum over all j
        sum_i_row = np.dot(a[i, :], diff_row)
        sum_i_col = np.dot(a[:, i], diff_col)
        sum_k_row = np.dot(a[k, :], diff_row)
        sum_k_col = np.dot(a[:, k], diff_col)
        
        total = sum_i_row + sum_i_col - sum_k_row - sum_k_col
        
        # Subtract the contributions of j=i and j=k from 'total'
        # j=i:
        # sum_i_row included: a[i,i] * (b[v, u] - b[u, u])
        # sum_i_col included: a[i,i] * (b[u, v] - b[u, u])
        # sum_k_row included: a[k,i] * (b[v, u] - b[u, u])
        # sum_k_col included: a[i,k] * (b[u, v] - b[u, u])
        
        # j=k:
        # sum_i_row included: a[i,k] * (b[v, v] - b[u, v])
        # sum_i_col included: a[k,i] * (b[v, v] - b[v, u])
        # sum_k_row included: a[k,k] * (b[v, v] - b[u, v])
        # sum_k_col included: a[k,k] * (b[v, v] - b[v, u])
        
        # Add the actual changes for the 4 pairs (i,i), (i,k), (k,i), (k,k)
        # (i,i): a[i,i] * (b[v,v] - b[u,u])
        # (i,k): a[i,k] * (b[v,u] - b[u,v])
        # (k,i): a[k,i] * (b[u,v] - b[v,u])
        # (k,k): a[k,k] * (b[u,u] - b[v,v])
        
        # This is getting complicated to verify by hand. 
        # The previous code's formula was:
        # diag_ii = a[i,i] * (b[v,v] - b[u,u])
        # diag_kk = a[k,k] * (b[u,u] - b[v,v])
        # cross_ik = a[i,k] * (b[v,u] - b[u,v])
        # cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        # off_diag = total - term_i_val - term_k_val
        # where term_i_val was the sum of the 4 terms involving j=i in the 'total' expression.
        # term_k_val was the sum of the 4 terms involving j=k in the 'total' expression.
        
        # Let's reuse the logic from the current program which was working, but clean it up.
        
        term_i_val = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        term_k_val = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        off_diag = total - term_i_val - term_k_val
        
        diag_ii = a[i,i] * (b[v,v] - b[u,u])
        diag_kk = a[k,k] * (b[u,u] - b[v,v])
        cross_ik = a[i,k] * (b[v,u] - b[u,v])
        cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        delta = diag_ii + diag_kk + cross_ik + cross_ki + off_diag
        return delta

    # Local Search: First Improvement 2-opt
    def local_search(p, current_cost):
        improved = True
        # Limit iterations to avoid infinite loops or excessive time in one pass
        max_iter = 1000 
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Shuffle order to avoid bias? Or just sweep?
            # For speed, a simple sweep is good.
            for i in range(n):
                for k in range(i+1, n):
                    delta = compute_delta_swap(i, k, p)
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
        return p, current_cost

    # Simulated Annealing
    def simulated_annealing(p, current_cost, time_limit):
        # Estimate T
        sample_deltas = []
        for _ in range(50):
            i, k = np.random.choice(n, 2, replace=False)
            d = compute_delta_swap(i, k, p)
            sample_deltas.append(abs(d))
        T = np.mean(sample_deltas) * 2.0 if sample_deltas else 10000.0
        T_min = 1.0
        alpha = 0.9998
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            i, k = np.random.choice(n, 2, replace=False)
            delta = compute_delta_swap(i, k, p)
            
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return p, current_cost

    # Main Loop
    start_time = time.time()
    total_time_limit = 110.0  # Leave 10s buffer
    
    # 1. Initial Local Search
    p, current_cost = local_search(p, current_cost)
    
    # 2. Alternating SA and Local Search
    # We will run SA for a chunk of time, then Local Search.
    sa_time_chunk = 10.0
    
    while time.time() - start_time < total_time_limit - sa_time_chunk:
        # Run SA
        p, current_cost = simulated_annealing(p, current_cost, sa_time_chunk)
        
        # Run Local Search
        p, current_cost = local_search(p, current_cost)
        
        # Check time
        if time.time() - start_time > total_time_limit:
            break

    # Final Local Search if time permits
    if time.time() - start_time < total_time_limit - 1.0:
        p, current_cost = local_search(p, current_cost)

    return best_p.tolist()
