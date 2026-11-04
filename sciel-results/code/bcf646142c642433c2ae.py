import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Hungarian Algorithm on linearized cost
    # W[i, j] represents the contribution of placing facility i at location j
    # Linearization: sum_j a[i,j] * b[p[i], p[j]] is not linear.
    # Standard proxy: W = A @ B.T
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    def get_cost(p):
        B_perm = b[:, p]
        S = a @ B_perm
        return int(np.sum(S[np.arange(n), p]))
        
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for fast delta calculation
    # For swap i, k:
    # We need sum_j a[i,j] * (b[v, p[j]] - b[u, p[j]]) etc.
    
    def compute_delta_swap(i, k, p):
        u, v = p[i], p[k]
        if u == v: return 0
        
        # Vector differences
        # b[v, p] is row v of b, indexed by p
        # b[u, p] is row u of b, indexed by p
        diff_row = b[v, p] - b[u, p]
        diff_col = b[p, v] - b[p, u]
        
        # Main terms for j != i, k
        # sum_j a[i,j] * diff_row[j] - sum_j a[k,j] * diff_row[j]
        # + sum_j a[j,i] * diff_col[j] - sum_j a[j,k] * diff_col[j]
        
        # We compute for all j, then correct for j=i and j=k
        term_i_row = np.dot(a[i, :], diff_row)
        term_k_row = np.dot(a[k, :], diff_row)
        term_i_col = np.dot(a[:, i], diff_col)
        term_k_col = np.dot(a[:, k], diff_col)
        
        total = (term_i_row - term_k_row) + (term_i_col - term_k_col)
        
        # Corrections for j=i and j=k
        # When j=i: p[i]=u. 
        # In total, we used b[v, p[i]] = b[v, u] and b[p[i], v] = b[u, v].
        # Correct terms should use p[i]=v (new) and p[k]=u (new)? No, the formula is New - Old.
        # Let's stick to the proven formula from the previous code which was verified.
        
        # Previous formula:
        # diag_ii = a[i,i] * (b[v,v] - b[u,u])
        # diag_kk = a[k,k] * (b[u,u] - b[v,v])
        # cross_ik = a[i,k] * (b[v,u] - b[u,v])
        # cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        # Off-diag part:
        # sum_{j!=i,k} (a[i,j]-a[k,j])*(b[v,p[j]]-b[u,p[j]]) + (a[j,i]-a[j,k])*(b[p[j],v]-b[p[j],u])
        
        # Let's re-verify the "total" calculation above.
        # total = sum_j (a[i,j]-a[k,j]) * diff_row[j] + sum_j (a[j,i]-a[j,k]) * diff_col[j]
        
        # For j=i:
        # (a[i,i]-a[k,i]) * (b[v, p[i]] - b[u, p[i]]) + (a[i,i]-a[i,k]) * (b[p[i], v] - b[p[i], u])
        # p[i] is currently u.
        # = (a[i,i]-a[k,i]) * (b[v, u] - b[u, u]) + (a[i,i]-a[i,k]) * (b[u, v] - b[u, u])
        
        # For j=k:
        # (a[i,k]-a[k,k]) * (b[v, p[k]] - b[u, p[k]]) + (a[k,i]-a[k,k]) * (b[p[k], v] - b[p[k], u])
        # p[k] is currently v.
        # = (a[i,k]-a[k,k]) * (b[v, v] - b[u, v]) + (a[k,i]-a[k,k]) * (b[v, v] - b[v, u])
        
        # The sum over j != i, k is total - term_i - term_k.
        
        term_i_val = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        term_k_val = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        off_diag = total - term_i_val - term_k_val
        
        diag_ii = a[i,i] * (b[v,v] - b[u,u])
        diag_kk = a[k,k] * (b[u,u] - b[v,v])
        cross_ik = a[i,k] * (b[v,u] - b[u,v])
        cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        delta = diag_ii + diag_kk + cross_ik + cross_ki + off_diag
        return delta

    def compute_delta_3opt(i, j, k, p):
        # Cyclic shift: i->j, j->k, k->i
        # p[i] -> p[j], p[j] -> p[k], p[k] -> p[i]
        # This is complex. For now, stick to swaps for reliability, 
        # but maybe implement a simple 3-opt if time permits.
        # Given the 120s limit, swaps are safer.
        return 0 # Placeholder

    # 2. Simulated Annealing
    # Estimate T
    sample_deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        d = compute_delta_swap(i, k, p)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 2.0 if sample_deltas else 10000.0
    T_min = 1.0
    alpha = 0.9998
    
    start_time = time.time()
    time_limit = 60.0
    
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
            
    # 3. Fast Local Search (2-opt Descent)
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    # Limit passes to avoid timeout
    max_passes = 10
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
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
                        
    # 4. Second SA Round
    p = best_p.copy()
    current_cost = best_cost
    
    # Re-estimate T
    sample_deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        d = compute_delta_swap(i, k, p)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 1.5 if sample_deltas else 5000.0
    T_min = 1.0
    alpha = 0.9999
    
    start_time = time.time()
    time_limit = 20.0
    
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

    # 5. Final Local Search
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    max_passes = 5
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
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

    return best_p.tolist()
