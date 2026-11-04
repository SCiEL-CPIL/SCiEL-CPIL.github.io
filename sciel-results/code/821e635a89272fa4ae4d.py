import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Hungarian Algorithm
    # The directive suggests a matching formulation. 
    # Standard linearization for QAP initialization is W = A @ B.T
    # We maximize sum W[i, p[i]], which is equivalent to minimizing -W.
    # linear_sum_assignment minimizes cost, so we pass -W.
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    def get_cost(p):
        B_perm = b[:, p]
        S = a @ B_perm
        return int(np.sum(S[np.arange(n), p]))
        
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute row sums of A and B? 
    # For fast delta, we precompute vectors.
    
    def compute_delta_swap(i, k, p):
        u, v = p[i], p[k]
        if u == v: return 0
        
        # Vector differences
        diff_row = b[v, p] - b[u, p]
        diff_col = b[p, v] - b[p, u]
        
        term_i_row = np.dot(a[i, :], diff_row)
        term_k_row = np.dot(a[k, :], diff_row)
        term_i_col = np.dot(a[:, i], diff_col)
        term_k_col = np.dot(a[:, k], diff_col)
        
        total = (term_i_row - term_k_row) + (term_i_col - term_k_col)
        
        # Corrections for j=i and j=k
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
        # Old: p[i]=u, p[j]=v, p[k]=w
        # New: p[i]=v, p[j]=w, p[k]=u
        u, v, w = p[i], p[j], p[k]
        if u == v or v == w or u == w: return 0
        
        # We compute the change in cost.
        # Cost = sum_{r,s} a[r,s] b[p[r], p[s]]
        # Only terms involving i, j, k change.
        # Let's use the general formula:
        # Delta = Cost(New) - Cost(Old)
        # This is O(n) if we sum over all other nodes r.
        
        # Terms involving i:
        # Old: sum_s a[i,s] b[u, p[s]] + sum_r a[r,i] b[p[r], u]
        # New: sum_s a[i,s] b[v, p[s]] + sum_r a[r,i] b[p[r], v]
        # Note: p[s] changes for s in {i,j,k}.
        
        # It is safer to compute the full contribution of i, j, k to the cost in both states.
        # Contribution of a set of nodes S to the total cost is:
        # C(S) = sum_{r in S, s in S} a[r,s] b[p[r], p[s]] + sum_{r in S, s not in S} a[r,s] b[p[r], p[s]] + sum_{r not in S, s in S} a[r,s] b[p[r], p[s]]
        
        # Let's calculate the difference.
        # Nodes: i, j, k.
        # Old positions: u, v, w.
        # New positions: v, w, u.
        
        # 1. Internal terms (r,s in {i,j,k})
        # Old:
        # a[i,i]b[u,u] + a[i,j]b[u,v] + a[i,k]b[u,w]
        # a[j,i]b[v,u] + a[j,j]b[v,v] + a[j,k]b[v,w]
        # a[k,i]b[w,u] + a[k,j]b[w,v] + a[k,k]b[w,w]
        
        # New:
        # a[i,i]b[v,v] + a[i,j]b[v,w] + a[i,k]b[v,u]
        # a[j,i]b[w,v] + a[j,j]b[w,w] + a[j,k]b[w,u]
        # a[k,i]b[u,w] + a[k,j]b[u,v] + a[k,k]b[u,u]
        
        old_internal = (a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[i,k]*b[u,w] +
                        a[j,i]*b[v,u] + a[j,j]*b[v,v] + a[j,k]*b[v,w] +
                        a[k,i]*b[w,u] + a[k,j]*b[w,v] + a[k,k]*b[w,w])
        
        new_internal = (a[i,i]*b[v,v] + a[i,j]*b[v,w] + a[i,k]*b[v,u] +
                        a[j,i]*b[w,v] + a[j,j]*b[w,w] + a[j,k]*b[w,u] +
                        a[k,i]*b[u,w] + a[k,j]*b[u,v] + a[k,k]*b[u,u])
        
        # 2. External terms (one in S, one not in S)
        # For r in S, s not in S: a[r,s] (b[p_new[r], p[s]] - b[p_old[r], p[s]])
        # For r not in S, s in S: a[r,s] (b[p[r], p_new[s]] - b[p[r], p_old[s]])
        
        # Sum over s not in {i,j,k}
        # Let's create masks or just loop? n=150, loop is fine but vectorized is faster.
        
        # Vectorized approach:
        # diff_i_row = b[v, p] - b[u, p]  (Change in row i's contribution to other columns)
        # diff_j_row = b[w, p] - b[v, p]
        # diff_k_row = b[u, p] - b[w, p]
        
        # diff_i_col = b[p, v] - b[p, u]
        # diff_j_col = b[p, w] - b[p, v]
        # diff_k_col = b[p, u] - b[p, w]
        
        # Sum a[i, s] * diff_i_row[s] for s not in {i,j,k}
        # Sum a[s, i] * diff_i_col[s] for s not in {i,j,k}
        
        # We can sum over all s and subtract the internal ones.
        
        total_i_row = np.dot(a[i, :], b[v, p] - b[u, p])
        total_j_row = np.dot(a[j, :], b[w, p] - b[v, p])
        total_k_row = np.dot(a[k, :], b[u, p] - b[w, p])
        
        total_i_col = np.dot(a[:, i], b[p, v] - b[p, u])
        total_j_col = np.dot(a[:, j], b[p, w] - b[p, v])
        total_k_col = np.dot(a[:, k], b[p, u] - b[p, w])
        
        # The sums above include s in {i,j,k}. We need to subtract those internal contributions
        # because we already accounted for the internal change in `new_internal - old_internal`.
        
        # Let's verify:
        # np.dot(a[i, :], diff_i_row) = sum_s a[i,s] (b[v, p[s]] - b[u, p[s]])
        # This includes s=i, s=j, s=k.
        # The terms for s in {i,j,k} are:
        # s=i: a[i,i](b[v,u]-b[u,u])
        # s=j: a[i,j](b[v,v]-b[u,v])
        # s=k: a[i,k](b[v,w]-b[u,w])
        
        # In `new_internal - old_internal`, the terms involving row i are:
        # New: a[i,i]b[v,v] + a[i,j]b[v,w] + a[i,k]b[v,u]
        # Old: a[i,i]b[u,u] + a[i,j]b[u,v] + a[i,k]b[u,w]
        # Diff (Row i part): a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,w]-b[u,v]) + a[i,k](b[v,u]-b[u,w])
        
        # Notice the difference in indices for b:
        # Dot product term s=j: a[i,j](b[v, p[j]] - b[u, p[j]]) = a[i,j](b[v,v] - b[u,v]). Matches.
        # Dot product term s=k: a[i,k](b[v, p[k]] - b[u, p[k]]) = a[i,k](b[v,w] - b[u,w]). Matches.
        # Dot product term s=i: a[i,i](b[v, p[i]] - b[u, p[i]]) = a[i,i](b[v,u] - b[u,u]). 
        # Internal diff term s=i: a[i,i](b[v,v] - b[u,u]). Mismatch! p[i] changes from u to v.
        # In the dot product, p[i] is still u (old state) because we are computing the difference based on the *change* in position of i, but the *other* positions are fixed at old state?
        # No, `b[v, p]` uses the current p.
        # So `b[v, p[i]]` is `b[v, u]`.
        # But in the new state, p[i] becomes v. So the term is `a[i,i] b[v, v]`.
        # The internal calculation used `b[v,v]`.
        # The dot product calculation used `b[v,u]`.
        # So we must subtract the dot product internal terms and add the correct internal terms?
        # Actually, it's easier to just:
        # External = (Sum_all - Sum_internal_dot)
        # Total Delta = (New_Internal - Old_Internal) + External_i + External_j + External_k
        
        # Let's extract internal dot terms:
        # For i:
        # s=i: a[i,i](b[v,u]-b[u,u])
        # s=j: a[i,j](b[v,v]-b[u,v])
        # s=k: a[i,k](b[v,w]-b[u,w])
        # Sum_i_dot_int = a[i,i](b[v,u]-b[u,u]) + a[i,j](b[v,v]-b[u,v]) + a[i,k](b[v,w]-b[u,w])
        
        # For j:
        # s=i: a[j,i](b[w,u]-b[v,u])
        # s=j: a[j,j](b[w,v]-b[v,v])
        # s=k: a[j,k](b[w,w]-b[v,w])
        # Sum_j_dot_int = a[j,i](b[w,u]-b[v,u]) + a[j,j](b[w,v]-b[v,v]) + a[j,k](b[w,w]-b[v,w])
        
        # For k:
        # s=i: a[k,i](b[u,u]-b[w,u])
        # s=j: a[k,j](b[u,v]-b[w,v])
        # s=k: a[k,k](b[u,w]-b[w,w])
        # Sum_k_dot_int = a[k,i](b[u,u]-b[w,u]) + a[k,j](b[u,v]-b[w,v]) + a[k,k](b[u,w]-b[w,w])
        
        ext_i_row = total_i_row - (a[i,i]*(b[v,u]-b[u,u]) + a[i,j]*(b[v,v]-b[u,v]) + a[i,k]*(b[v,w]-b[u,w]))
        ext_j_row = total_j_row - (a[j,i]*(b[w,u]-b[v,u]) + a[j,j]*(b[w,v]-b[v,v]) + a[j,k]*(b[w,w]-b[v,w]))
        ext_k_row = total_k_row - (a[k,i]*(b[u,u]-b[w,u]) + a[k,j]*(b[u,v]-b[w,v]) + a[k,k]*(b[u,w]-b[w,w]))
        
        # Similarly for columns
        # For i (col i changes from u to v):
        # s=i: a[i,i](b[u,v]-b[u,u])  # Wait, col i is at pos u->v. 
        # diff_i_col[s] = b[p[s], v] - b[p[s], u]
        # s=i: b[u, v] - b[u, u]
        # s=j: b[v, v] - b[v, u]
        # s=k: b[w, v] - b[w, u]
        # Sum_i_col_int = a[i,i](b[u,v]-b[u,u]) + a[j,i](b[v,v]-b[v,u]) + a[k,i](b[w,v]-b[w,u])
        
        ext_i_col = total_i_col - (a[i,i]*(b[u,v]-b[u,u]) + a[j,i]*(b[v,v]-b[v,u]) + a[k,i]*(b[w,v]-b[w,u]))
        
        # For j (col j changes from v to w):
        # diff_j_col[s] = b[p[s], w] - b[p[s], v]
        # s=i: b[u, w] - b[u, v]
        # s=j: b[v, w] - b[v, v]
        # s=k: b[w, w] - b[w, v]
        # Sum_j_col_int = a[i,j](b[u,w]-b[u,v]) + a[j,j](b[v,w]-b[v,v]) + a[k,j](b[w,w]-b[w,v])
        
        ext_j_col = total_j_col - (a[i,j]*(b[u,w]-b[u,v]) + a[j,j](b[v,w]-b[v,v]) + a[k,j](b[w,w]-b[w,v]))
        
        # For k (col k changes from w to u):
        # diff_k_col[s] = b[p[s], u] - b[p[s], w]
        # s=i: b[u, u] - b[u, w]
        # s=j: b[v, u] - b[v, w]
        # s=k: b[w, u] - b[w, w]
        # Sum_k_col_int = a[i,k](b[u,u]-b[u,w]) + a[j,k](b[v,u]-b[v,w]) + a[k,k](b[w,u]-b[w,w])
        
        ext_k_col = total_k_col - (a[i,k](b[u,u]-b[u,w]) + a[j,k](b[v,u]-b[v,w]) + a[k,k](b[w,u]-b[w,w]))
        
        delta = (new_internal - old_internal) + (ext_i_row + ext_j_row + ext_k_row) + (ext_i_col + ext_j_col + ext_k_col)
        return delta

    # 2. Simulated Annealing with 2-Opt and 3-Opt
    # Estimate T
    sample_deltas = []
    for _ in range(200):
        if np.random.rand() < 0.5:
            i, k = np.random.choice(n, 2, replace=False)
            d = compute_delta_swap(i, k, p)
        else:
            i, j, k = np.random.choice(n, 3, replace=False)
            d = compute_delta_3opt(i, j, k, p)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 2.5 if sample_deltas else 10000.0
    T_min = 1.0
    alpha = 0.9995
    
    start_time = time.time()
    time_limit = 50.0
    
    while time.time() - start_time < time_limit:
        if np.random.rand() < 0.5:
            i, k = np.random.choice(n, 2, replace=False)
            delta = compute_delta_swap(i, k, p)
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        else:
            i, j, k = np.random.choice(n, 3, replace=False)
            delta = compute_delta_3opt(i, j, k, p)
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                # Apply cyclic shift
                p[i], p[j], p[k] = p[j], p[k], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
        T *= alpha
        if T < T_min:
            T = T_min
            
    # 3. Fast Local Search (2-Opt Descent)
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    max_passes = 20
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
                        
    # 4. Second SA Round with 3-Opt
    p = best_p.copy()
    current_cost = best_cost
    
    sample_deltas = []
    for _ in range(200):
        if np.random.rand() < 0.5:
            i, k = np.random.choice(n, 2, replace=False)
            d = compute_delta_swap(i, k, p)
        else:
            i, j, k = np.random.choice(n, 3, replace=False)
            d = compute_delta_3opt(i, j, k, p)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 1.5 if sample_deltas else 5000.0
    T_min = 1.0
    alpha = 0.9998
    
    start_time = time.time()
    time_limit = 40.0
    
    while time.time() - start_time < time_limit:
        if np.random.rand() < 0.5:
            i, k = np.random.choice(n, 2, replace=False)
            delta = compute_delta_swap(i, k, p)
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        else:
            i, j, k = np.random.choice(n, 3, replace=False)
            delta = compute_delta_3opt(i, j, k, p)
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p[i], p[j], p[k] = p[j], p[k], p[i]
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

    return best_p.tolist()
