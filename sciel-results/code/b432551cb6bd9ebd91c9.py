"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Initial Heuristic: Spectral Embedding
    # Model QAP as graph isomorphism. Project facility and location graphs into shared Euclidean space.
    
    # Compute Laplacians
    # For flow matrix A, degree d_i = sum_j A[i,j]. L = D - A.
    # For distance matrix B, degree d_i = sum_j B[i,j]. L = D - B.
    
    flow_degrees = a.sum(axis=1)
    dist_degrees = b.sum(axis=1)
    
    L_a = np.diag(flow_degrees) - a
    L_b = np.diag(dist_degrees) - b
    
    # Compute eigenvectors
    # We want the smallest non-zero eigenvalues (Fiedler vectors) to capture structure.
    # For n=100, full eigendecomposition is fast.
    
    try:
        # Symmetric matrices
        evals_a, evecs_a = np.linalg.eigh(L_a)
        evals_b, evecs_b = np.linalg.eigh(L_b)
        
        # Select first K eigenvectors (excluding the trivial zero eigenvalue if present, 
        # but usually we just take the first K smallest non-negative ones).
        # For connected graphs, the first eigenvalue is 0. We skip it.
        # However, for safety, let's just take the first K eigenvectors corresponding to smallest eigenvalues.
        K = 10 # Dimension of embedding
        
        # If the graph is disconnected, there might be multiple zero eigenvalues.
        # Let's just take the first K eigenvectors.
        V_a = evecs_a[:, :K]
        V_b = evecs_b[:, :K]
        
        # Center the embeddings? Not strictly necessary for matching if we use Hungarian.
        # But centering can help.
        V_a = V_a - V_a.mean(axis=0)
        V_b = V_b - V_b.mean(axis=0)
        
        # Match facilities to locations based on similarity in embedding space.
        # We want to minimize sum ||V_a[i] - V_b[p[i]]||^2 ?
        # Or maximize similarity?
        # Since we are minimizing QAP cost, and QAP is related to graph isomorphism,
        # similar neighborhoods should be matched.
        
        # Cost matrix for assignment: C[i, j] = -dot(V_a[i], V_b[j]) ?
        # Or C[i, j] = ||V_a[i] - V_b[j]||^2
        # Let's use squared distance.
        # C[i, j] = ||V_a[i]||^2 + ||V_b[j]||^2 - 2 V_a[i] dot V_b[j]
        
        # Use Hungarian algorithm for optimal assignment.
        from scipy.optimize import linear_sum_assignment
        
        # Construct cost matrix
        # To avoid large numbers, we can just use -dot product or distance.
        # Distance is safer.
        # V_a shape (n, K), V_b shape (n, K)
        # Dist^2 = sum_k (V_a[i,k] - V_b[j,k])^2
        # = sum_k V_a[i,k]^2 + sum_k V_b[j,k]^2 - 2 sum_k V_a[i,k] V_b[j,k]
        
        aa = np.sum(V_a**2, axis=1)
        bb = np.sum(V_b**2, axis=1)
        ab = V_a @ V_b.T
        
        cost_assign = aa[:, None] + bb[None, :] - 2 * ab
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_assign)
        
        # row_ind is facilities, col_ind is locations
        # p[i] = location for facility i
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
    except Exception as e:
        # Fallback to Rank Matching if spectral fails
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations

    # 2. Local Search / Tabu Search with First-Improvement and Perturbation
    
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Precompute for fast delta evaluation
    # We maintain row_sums[i] = sum_j a[i,j] b[p[i],p[j]]
    # and col_sums[j] = sum_i a[i,j] b[p[i],p[j]]
    # cost = sum_i row_sums[i]
    
    def build_sums(p):
        M = b[np.ix_(p, p)]
        row_sums = np.einsum('ij,ij->i', a, M)
        col_sums = np.einsum('ij,ij->j', a, M)
        return row_sums, col_sums, int(np.sum(row_sums))

    row_sums, col_sums, current_cost = build_sums(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu List
    tabu_len = 50
    tabu_list = [] # List of (i, j, expiry_iter)
    current_iter = 0
    
    # Simulated Annealing parameters for perturbation
    T = 500.0
    T_min = 10.0
    alpha = 0.9995
    
    # Helper to calculate delta for swap (i, j)
    # x = p[i], y = p[j]
    # Delta = (sum_k a[i,k] (b[y,p[k]] - b[x,p[k]]) + sum_k a[k,i] (b[p[k],y] - b[p[k],x]))
    #       + (sum_k a[j,k] (b[x,p[k]] - b[y,p[k]]) + sum_k a[k,j] (b[p[k],x] - b[p[k],y]))
    #       + (a[i,j] (b[y,x] - b[x,y]) + a[j,i] (b[x,y] - b[y,x]))
    #       - (a[i,j] (b[x,x] + b[y,y]) + a[j,i] (b[x,x] + b[y,y])) # Correction for self terms?
    
    # Let's re-derive carefully.
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Swap p[i] and p[j].
    # Terms not involving i or j are unchanged.
    # Terms involving i:
    #   u=i: sum_v a[i,v] b[p[i], p[v]] -> sum_v a[i,v] b[p[j], p[v]]
    #   v=i: sum_u a[u,i] b[p[u], p[i]] -> sum_u a[u,i] b[p[u], p[j]]
    # Terms involving j:
    #   u=j: sum_v a[j,v] b[p[j], p[v]] -> sum_v a[j,v] b[p[i], p[v]]
    #   v=j: sum_u a[u,j] b[p[u], p[j]] -> sum_u a[u,j] b[p[u], p[i]]
    
    # Let x = p[i], y = p[j].
    # Old:
    # R_i = sum_v a[i,v] b[x, p[v]]
    # C_i = sum_u a[u,i] b[p[u], x]
    # R_j = sum_v a[j,v] b[y, p[v]]
    # C_j = sum_u a[u,j] b[p[u], y]
    
    # New:
    # R_i' = sum_v a[i,v] b[y, p[v]]
    # C_i' = sum_u a[u,i] b[p[u], y]
    # R_j' = sum_v a[j,v] b[x, p[v]]
    # C_j' = sum_u a[u,j] b[p[u], x]
    
    # Note that in R_i, the term v=j is a[i,j] b[x, y].
    # In C_j, the term u=i is a[i,j] b[x, y].
    # In R_j, the term v=i is a[j,i] b[y, x].
    # In C_i, the term u=j is a[j,i] b[y, x].
    
    # So Old Cost contribution from i,j interactions:
    # Old = R_i + C_i + R_j + C_j
    # But wait, R_i includes v=i (a[i,i] b[x,x]) and v=j (a[i,j] b[x,y]).
    # C_i includes u=i (a[i,i] b[x,x]) and u=j (a[j,i] b[y,x]).
    # So a[i,i] b[x,x] is counted twice in R_i + C_i.
    # And a[i,j] b[x,y] is in R_i. a[j,i] b[y,x] is in C_i.
    
    # Let's just compute the change directly using vectors.
    # Delta = (R_i' + C_i' + R_j' + C_j') - (R_i + C_i + R_j + C_j)
    
    # R_i' - R_i = sum_v a[i,v] (b[y, p[v]] - b[x, p[v]])
    # C_i' - C_i = sum_u a[u,i] (b[p[u], y] - b[p[u], x])
    # R_j' - R_j = sum_v a[j,v] (b[x, p[v]] - b[y, p[v]])
    # C_j' - C_j = sum_u a[u,j] (b[p[u], x] - b[p[u], y])
    
    # Let diff_row = b[y, p] - b[x, p] (vector of length n)
    # Let diff_col = b[p, y] - b[p, x] (vector of length n)
    
    # Delta_row_i = dot(a[i, :], diff_row)
    # Delta_col_i = dot(a[:, i], diff_col)
    # Delta_row_j = dot(a[j, :], -diff_row)
    # Delta_col_j = dot(a[:, j], -diff_col)
    
    # Total Delta = Delta_row_i + Delta_col_i + Delta_row_j + Delta_col_j
    
    # Let's verify with the cross terms.
    # In Delta_row_i, term v=j: a[i,j] (b[y, y] - b[x, y])
    # In Delta_col_j, term u=i: a[i,j] (b[x, x] - b[x, y])
    # Sum: a[i,j] (b[y,y] + b[x,x] - 2b[x,y])
    
    # In Delta_row_j, term v=i: a[j,i] (b[x, x] - b[y, x])
    # In Delta_col_i, term u=j: a[j,i] (b[y, y] - b[y, x])
    # Sum: a[j,i] (b[x,x] + b[y,y] - 2b[y,x])
    
    # Total cross change:
    # a[i,j] (b[y,y] + b[x,x] - 2b[x,y]) + a[j,i] (b[x,x] + b[y,y] - 2b[y,x])
    
    # Actual change in cost for pair (i,j):
    # Old: a[i,j] b[x,y] + a[j,i] b[y,x]
    # New: a[i,j] b[y,x] + a[j,i] b[x,y]
    # Change: a[i,j] (b[y,x] - b[x,y]) + a[j,i] (b[x,y] - b[y,x])
    
    # These don't match. The formula above is incorrect because it doesn't account for the fact that
    # when we swap, the "other" index also changes its location in the context of the "self" term?
    # No, the self term a[i,i] b[p[i], p[i]] changes from b[x,x] to b[y,y].
    # My formula:
    # Delta_row_i includes v=i: a[i,i] (b[y,y] - b[x,x])
    # Delta_col_i includes u=i: a[i,i] (b[y,y] - b[x,x])
    # Total self change: 2 * a[i,i] (b[y,y] - b[x,x])
    # Actual self change: a[i,i] (b[y,y] - b[x,x])
    # So we are double counting the self terms.
    
    # Correction:
    # The sum over v in R_i includes v=i. The sum over u in C_i includes u=i.
    # So the self term is counted twice in the delta calculation.
    # Similarly, the cross terms are counted in both row and col deltas?
    # Let's check cross term (i,j) in Delta_row_i (v=j) and Delta_col_j (u=i).
    # They are distinct terms in the sums.
    # But the cost function sums over ALL u,v.
    # Term (i,j) appears once in the total sum.
    # In my delta decomposition:
    # R_i covers (i, v) for all v.
    # C_i covers (u, i) for all u.
    # So (i,j) is in R_i. (j,i) is in C_i.
    # (j,i) is in R_j. (i,j) is in C_j.
    # Wait.
    # Cost = sum_u sum_v a[u,v] b[p[u], p[v]]
    # R_i = sum_v a[i,v] b[p[i], p[v]]. This is the row i of the cost matrix.
    # C_i = sum_u a[u,i] b[p[u], p[i]]. This is the column i of the cost matrix.
    # Sum(R_i) = Sum(C_i) = Total Cost.
    # So we don't add R and C. We just compute the change in one of them, or sum of changes?
    # If we compute Delta = (New R_i + New C_i + New R_j + New C_j) - (Old R_i + Old C_i + Old R_j + Old C_j),
    # we are effectively summing the changes in 4 vectors.
    # But Total Cost = Sum(R). So Delta = (New R_i - Old R_i) + (New R_j - Old R_j) + ...
    # Wait, if we just change p[i] and p[j], then R_k for k!=i,j also changes?
    # R_k = sum_v a[k,v] b[p[k], p[v]].
    # If v=i, p[i] changes. So R_k changes for all k.
    # So my local delta calculation is WRONG. It only accounts for changes in rows i and j.
    # But row k (k!=i,j) has term a[k,i] b[p[k], p[i]]. Since p[i] changes, this term changes.
    # So we MUST account for all rows.
    
    # Correct Delta for swap (i,j):
    # Delta = sum_k (New R_k - Old R_k)
    # New R_k - Old R_k = sum_v a[k,v] (b[p[k], p[v]]_new - b[p[k], p[v]]_old)
    # For k != i,j: p[k] is unchanged.
    # Only p[i] and p[j] change.
    # So for k != i,j:
    # Change = a[k,i] (b[p[k], y] - b[p[k], x]) + a[k,j] (b[p[k], x] - b[p[k], y])
    # For k = i:
    # Change = sum_v a[i,v] (b[y, p[v]]_new - b[x, p[v]]_old)
    # Note p[v] is old for v!=i,j. p[i]=y, p[j]=x.
    # So Change_i = sum_{v!=i,j} a[i,v] (b[y, p[v]] - b[x, p[v]]) + a[i,i] (b[y,y] - b[x,x]) + a[i,j] (b[y,x] - b[x,y])
    # For k = j:
    # Change_j = sum_{v!=i,j} a[j,v] (b[x, p[v]] - b[y, p[v]]) + a[j,j] (b[x,x] - b[y,y]) + a[j,i] (b[x,y] - b[y,x])
    
    # Let's vectorize this.
    # Let diff_col_k = b[p[k], y] - b[p[k], x] for k != i,j.
    # Sum_{k!=i,j} a[k,i] diff_col_k + a[k,j] (-diff_col_k)
    # = sum_{k!=i,j} a[k,i] (b[p[k], y] - b[p[k], x]) - sum_{k!=i,j} a[k,j] (b[p[k], y] - b[p[k], x])
    # = (sum_{k!=i,j} a[k,i] b[p[k], y] - sum_{k!=i,j} a[k,i] b[p[k], x]) - (sum_{k!=i,j} a[k,j] b[p[k], y] - sum_{k!=i,j} a[k,j] b[p[k], x])
    
    # This is getting complicated to vectorize efficiently without precomputed sums.
    # Given n=100, full recalculation of cost is O(n^2) = 10k ops.
    # In a Python loop, 10k ops is ~1ms? No, numpy vectorized is fast.
    # But we are doing this in a loop.
    # If we can do 1000 swaps/sec, that's 1000 * 10k = 1e7 ops/sec.
    # 120s -> 1.2e9 ops.
    # So full recalculation is feasible if we use numpy vectorization for the cost calculation.
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    while time.time() - start_time < timeout:
        # First Improvement Tabu Search
        improved = False
        # Randomize order of pairs to avoid determinism?
        # Or just iterate all pairs.
        # n=100, 4950 pairs.
        # If we do full recalculation for each, 4950 * 10k = 50e6 ops per pass.
        # That's too slow for a single pass if we want many passes.
        # But we can break early on first improvement.
        
        # Let's try a few random swaps first, then full scan if no improvement?
        # Or just full scan with early exit.
        
        # To speed up, we can precompute some things?
        # For now, let's just do random swaps in SA style, but with Tabu.
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Check Tabu
        is_tabu = False
        # Remove expired tabu entries
        # Simple list check
        for k, (ti, tj, exp) in enumerate(tabu_list):
            if current_iter > exp:
                tabu_list.pop(k)
                break
        # Check if (i,j) or (j,i) is tabu
        for ti, tj, exp in tabu_list:
            if (ti == i and tj == j) or (ti == j and tj == i):
                is_tabu = True
                break
        
        x = p[i]
        y = p[j]
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        new_cost = calc_cost(p)
        delta = new_cost - current_cost
        
        # Aspiration rule: accept if new_cost < best_cost
        if delta < 0 or (not is_tabu and delta < 0) or (is_tabu and new_cost < best_cost):
            # Actually, standard Tabu: accept if improving OR (not tabu and not improving but allowed by SA?)
            # Let's stick to: Accept if improving. If not improving, accept with prob exp(-delta/T) if not tabu.
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_p = p.copy()
                # Add to tabu
                tabu_list.append((i, j, current_iter + tabu_len))
                improved = True
            else:
                # Reject
                p[i], p[j] = p[j], p[i]
        else:
            # Reject
            p[i], p[j] = p[j], p[i]
            
        T *= alpha
        if T < T_min:
            T = T_min
            # Perturbation: make a few random swaps
            for _ in range(5):
                pi = np.random.randint(n)
                pj = np.random.randint(n)
                if pi != pj:
                    p[pi], p[pj] = p[pj], p[pi]
            current_cost = calc_cost(p)
            T = 500.0 # Reset temp
            
        current_iter += 1
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
