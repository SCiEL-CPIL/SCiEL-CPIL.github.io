"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Initial Heuristic: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Precompute for fast delta evaluation
    # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
    # We maintain row_sums[i] = sum_j a[i,j] b[p[i],p[j]] and col_sums[j] = sum_i a[i,j] b[p[i],p[j]]
    # Note: cost = sum_i row_sums[i] = sum_j col_sums[j]
    
    def build_sums(p):
        b_perm = b[np.ix_(p, p)]
        row_sums = a.dot(b_perm)  # shape (n,)
        col_sums = b_perm.T.dot(a).T  # actually col_sums[j] = sum_i a[i,j] * b_perm[i,j]
        # Wait, let's be precise.
        # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
        # Let M = b[p[:,None], p[None,:]]
        # row_sums[i] = sum_j a[i,j] M[i,j]
        # col_sums[j] = sum_i a[i,j] M[i,j]
        # Note that M is symmetric only if b is symmetric. QAP instances are usually symmetric.
        # Let's assume symmetry for efficiency or handle general case.
        # For general case:
        # row_sums[i] = sum_j a[i,j] b[p[i], p[j]]
        # col_sums[j] = sum_i a[i,j] b[p[i], p[j]]
        
        # Efficient computation:
        # row_sums = a.dot(b[np.ix_(p, p)]) is correct for row_sums[i].
        # col_sums[j] = sum_i a[i,j] b[p[i], p[j]].
        # This is (a.T.dot(b_perm)) ? No.
        # Let's just compute both directly if n is small enough, or use vectorization.
        # n=100, so O(n^2) is 10k ops. Very fast.
        
        M = b[np.ix_(p, p)]
        row_sums = np.einsum('ij,ij->i', a, M)
        col_sums = np.einsum('ij,ij->j', a, M)
        return row_sums, col_sums, int(np.sum(row_sums))

    row_sums, col_sums, current_cost = build_sums(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Simulated Annealing parameters
    T = 500.0
    T_min = 10.0
    alpha = 0.9995
    
    # Local search parameters
    ls_interval = 5000  # Do local search every 5000 iterations
    ls_counter = 0
    
    def local_search(p, row_sums, col_sums, current_cost, max_iters=50):
        """First-improvement or best-improvement local search on swaps."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try all pairs? O(n^2) pairs, each O(n) delta. Total O(n^3) per LS step.
            # n=100 -> 1e6 ops. 50 iters -> 50e6 ops. Might be slow in Python loop.
            # Let's do a few random swaps in LS instead, or a smarter sweep.
            
            # Better: Randomized local search. Pick random pairs and accept if improvement.
            for _ in range(100):  # 100 random swap attempts per LS phase
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                # Delta for swapping p[i] and p[j]
                # Let x = p[i], y = p[j]
                # Old terms involving i: sum_k a[i,k] b[x, p[k]]
                # Old terms involving j: sum_k a[j,k] b[y, p[k]]
                # Terms involving k=i: sum_k a[k,i] b[p[k], x]
                # Terms involving k=j: sum_k a[k,j] b[p[k], y]
                
                # This is getting complicated to derive on the fly without precomputed structures.
                # Let's use the full recalculation for LS if it's infrequent, or a simpler delta.
                
                # Actually, let's stick to the SA loop but with a smarter move generation
                # or just optimize the SA delta calculation.
                
                # For now, let's just accept this function as a placeholder and rely on SA.
                # Or implement a simple 2-opt style check.
                
                # Let's compute delta for swap (i, j) where locations are x=p[i], y=p[j]
                # Cost change = - (terms with i or j using old locs) + (terms with i or j using new locs)
                
                # Terms to subtract:
                # 1. sum_{k != i, j} a[i,k] b[x, p[k]] + a[j,k] b[y, p[k]]
                # 2. sum_{k != i, j} a[k,i] b[p[k], x] + a[k,j] b[p[k], y]
                # 3. a[i,j] b[x,y] + a[j,i] b[y,x]
                
                # Terms to add:
                # 1. sum_{k != i, j} a[i,k] b[y, p[k]] + a[j,k] b[x, p[k]]
                # 2. sum_{k != i, j} a[k,i] b[p[k], y] + a[k,j] b[p[k], x]
                # 3. a[i,j] b[y,x] + a[j,i] b[x,y]
                
                # This is O(n) per swap.
                x = p[i]
                y = p[j]
                
                # Precompute vectors for row i and j of a, and row/col of b
                # a_i = a[i,:], a_j = a[j,:]
                # a_col_i = a[:,i], a_col_j = a[:,j]
                
                # Subtract part:
                # row_i_part = dot(a[i,:], b[x,:]) excluding j? No, b[x,p[k]]
                # We have b_perm = b[np.ix_(p,p)]
                # b[x, p[k]] is b_perm[x, k]
                # So sum_k a[i,k] b_perm[x,k] is dot(a[i,:], b_perm[x,:])
                # But we must exclude k=i and k=j in some cases?
                # Actually, the term a[i,i] b[x,x] is part of row i.
                # When we swap, p[i] becomes y.
                
                # Let's just use the full cost recalculation for LS if it's not too slow,
                # or better, optimize the SA delta.
                
                pass # Skip complex LS for now, focus on faster SA

        return p, row_sums, col_sums, current_cost

    # Optimize SA delta calculation
    # Delta for swapping p[i] and p[j] (locations x and y)
    # Let's precompute b_perm = b[np.ix_(p,p)]
    b_perm = b[np.ix_(p, p)]
    
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Perform swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Calculate delta
        # Old cost contribution of (i,j) pair and their interactions with others
        # This is complex. Let's use a simplified delta:
        # Delta = sum_{k} (a[i,k] (b[y,p[k]] - b[x,p[k]]) + a[k,i] (b[p[k],y] - b[p[k],x])) 
        #       + sum_{k} (a[j,k] (b[x,p[k]] - b[y,p[k]]) + a[k,j] (b[p[k],x] - b[p[k],y]))
        #       + (a[i,j] (b[y,x] - b[x,y]) + a[j,i] (b[x,y] - b[y,x]))
        
        # Note: b[y,p[k]] is b_perm[y,k]
        # b[x,p[k]] is b_perm[x,k]
        
        # Let's compute the change in row sums and col sums.
        # Change in row i: sum_k a[i,k] (b_perm[y,k] - b_perm[x,k])
        # Change in row j: sum_k a[j,k] (b_perm[x,k] - b_perm[y,k])
        # Change in col i: sum_k a[k,i] (b_perm[k,y] - b_perm[k,x])
        # Change in col j: sum_k a[k,j] (b_perm[k,x] - b_perm[k,y])
        
        # Note that b_perm[k,y] is b[p[k],y] which is b_perm[k,y] in the permuted matrix?
        # No, b_perm is fixed based on current p.
        # b_perm[r,c] = b[p[r], p[c]]
        # We want b[p[k], y] = b[p[k], p[j]] = b_perm[k, j]
        # We want b[p[k], x] = b[p[k], p[i]] = b_perm[k, i]
        
        # So:
        # Delta_row_i = dot(a[i,:], b_perm[y,:] - b_perm[x,:])
        # Delta_row_j = dot(a[j,:], b_perm[x,:] - b_perm[y,:])
        # Delta_col_i = dot(a[:,i], b_perm[:,y] - b_perm[:,x])  # Wait, b_perm[:,y] is column y of b_perm
        #             = dot(a[:,i], b_perm[:,j] - b_perm[:,i])
        # Delta_col_j = dot(a[:,j], b_perm[:,i] - b_perm[:,j])
        
        # Total Delta = Delta_row_i + Delta_row_j + Delta_col_i + Delta_col_j
        # But this double counts the (i,j) and (j,i) terms?
        # Let's verify.
        # The term a[i,j] b[x,y] is in row i (k=j) and col j (k=i).
        # In Delta_row_i, k=j term is a[i,j] (b_perm[y,j] - b_perm[x,j]) = a[i,j] (b[y,y] - b[x,y])
        # In Delta_col_j, k=i term is a[i,j] (b_perm[i,x] - b_perm[i,y]) = a[i,j] (b[x,x] - b[x,y])
        # This seems wrong. The swap changes p[i] to y and p[j] to x.
        # The term a[i,j] b[p[i],p[j]] becomes a[i,j] b[y,x].
        # Old term was a[i,j] b[x,y].
        # Change is a[i,j] (b[y,x] - b[x,y]).
        
        # My formula:
        # Delta_row_i includes a[i,j] (b[y,y] - b[x,y])? No, b_perm[y,j] is b[p[y], p[j]]? No, indices are locations.
        # b_perm is indexed by facility indices? No, b_perm = b[np.ix_(p,p)].
        # b_perm[u,v] = b[p[u], p[v]].
        # So b_perm[y,j] is b[p[y], p[j]]. But y is a location. p is facility->location.
        # This is confusing. Let's stick to facility indices in b_perm.
        # b_perm[i,j] = b[p[i], p[j]].
        # So b_perm[x,:] is not a row of b_perm because x is a location, not a facility index.
        
        # Correct approach:
        # We need vectors of length n (facilities).
        # For row i change: sum_k a[i,k] (b[p[i_new], p[k]] - b[p[i_old], p[k]])
        # p[i_new] = y, p[i_old] = x.
        # So we need vector v1 where v1[k] = b[y, p[k]] - b[x, p[k]].
        # b[y, p[k]] is b[y, p[k]]. This is not directly in b_perm unless we have a matrix B_loc where B_loc[u,v] = b[u,v].
        # b_perm[k,j] = b[p[k], p[j]] = b[p[k], x] if j=i (old).
        
        # Let's construct the vectors explicitly.
        # v_row_i[k] = b[y, p[k]] - b[x, p[k]]
        # This is b[np.ix_([y,x], p)] -> shape (2,n).
        # row_y = b[y, p]
        # row_x = b[x, p]
        # diff_row = row_y - row_x
        # Delta_row_i = dot(a[i,:], diff_row)
        
        # v_col_i[k] = b[p[k], y] - b[p[k], x]
        # col_y = b[p, y]
        # col_x = b[p, x]
        # diff_col = col_y - col_x
        # Delta_col_i = dot(a[:,i], diff_col)
        
        row_y = b[y, p]
        row_x = b[x, p]
        diff_row = row_y - row_x
        
        col_y = b[p, y]
        col_x = b[p, x]
        diff_col = col_y - col_x
        
        delta_row_i = np.dot(a[i, :], diff_row)
        delta_row_j = np.dot(a[j, :], -diff_row)
        
        delta_col_i = np.dot(a[:, i], diff_col)
        delta_col_j = np.dot(a[:, j], -diff_col)
        
        # The cross terms (i,j) and (j,i) are included in these sums?
        # For delta_row_i, k=j term is a[i,j] (b[y, p[j]] - b[x, p[j]]) = a[i,j] (b[y, x] - b[x, x])? No, p[j] is x (old).
        # So b[y, x] - b[x, x].
        # For delta_col_j, k=i term is a[i,j] (b[p[i], y] - b[p[i], x]) = a[i,j] (b[x, y] - b[x, x]).
        # Sum: a[i,j] (b[y,x] + b[x,y] - 2b[x,x]).
        # But the actual change for pair (i,j) is a[i,j](b[y,x]-b[x,y]) + a[j,i](b[x,y]-b[y,x]).
        # This does not match. The double counting is problematic.
        
        # Standard delta formula for QAP swap (i,j):
        # Let x=p[i], y=p[j].
        # Delta = -2 * sum_{k!=i,j} (a[i,k] b[x,p[k]] + a[k,i] b[p[k],x] + a[j,k] b[y,p[k]] + a[k,j] b[p[k],y])
        #        + 2 * sum_{k!=i,j} (a[i,k] b[y,p[k]] + a[k,i] b[p[k],y] + a[j,k] b[x,p[k]] + a[k,j] b[p[k],x])
        #        + a[i,j] (b[y,x] - b[x,y]) + a[j,i] (b[x,y] - b[y,x])
        #        - 2 * (a[i,j] b[x,y] + a[j,i] b[y,x])  # Remove old cross
        #        + 2 * (a[i,j] b[y,x] + a[j,i] b[x,y])  # Add new cross
        
        # This is getting messy. Given n=100, full recalculation O(n^2) is 10k ops.
        # 10k * 10000 iters = 1e8 ops. In Python, this is ~10s.
        # So full recalculation is feasible!
        
        # Let's just do full recalculation for simplicity and correctness.
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        
        # Recalculate cost
        b_perm_new = b[np.ix_(p, p)]
        new_cost = int(np.sum(a * b_perm_new))
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
            # Update b_perm for next iteration? No, we recalculate anyway.
        else:
            p[i], p[j] = p[j], p[i]
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
