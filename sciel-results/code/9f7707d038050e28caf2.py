"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # Precompute for fast cost evaluation
    # cost(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can write cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b)[:, p[i]] ... but let's think in terms of incremental updates.
    
    # For a 2-opt swap of p[i] and p[j], the change in cost can be computed efficiently.
    # Let's precompute: for each facility i, and each location l, the "row contribution"
    # Actually, let's define:
    #   C[i, l] = sum_j a[i, j] * b[l, p[j]]  ... but this depends on p.
    # Better: use the standard QAP 2-opt delta formula.
    
    # cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let's define for current permutation p:
    #   For swapping p[i] <-> p[j]:
    #   Only terms involving i or j change.
    #   Old contribution of i,j pair and their interactions:
    #   Let x = p[i], y = p[j]
    #   Terms that change:
    #     a[i,j] b[x,y] + a[j,i] b[y,x]  (the i-j and j-i terms)
    #     sum_{k != i,j} a[i,k] b[x, p[k]] + a[k,i] b[p[k], x]
    #     sum_{k != i,j} a[j,k] b[y, p[k]] + a[k,j] b[p[k], y]
    #   New:
    #     a[i,j] b[y,x] + a[j,i] b[x,y]
    #     sum_{k != i,j} a[i,k] b[y, p[k]] + a[k,i] b[p[k], y]
    #     sum_{k != i,j} a[j,k] b[x, p[k]] + a[k,j] b[p[k], x]
    #   
    # This is O(n) per swap evaluation. With n=150, that's 150 ops per swap.
    # Number of swaps: n*(n-1)/2 = ~11175. Total per iteration: ~1.7M ops.
    # In 120s, we can do maybe 1000-5000 iterations depending on overhead.
    
    # Let's use a smarter incremental approach.
    # Precompute:
    #   row_sum_a[i] = sum_j a[i,j]  (not directly useful)
    #   Instead, maintain:
    #   For each location l, and each facility i:
    #     S[i, l] = sum_{k: p[k] != l} a[i,k] * b[l, p[k]] + sum_{k: p[k] != l} a[k,i] * b[p[k], l]
    #   This is the "external" contribution to facility i if it were at location l, 
    #   excluding the interaction with the facility currently at l.
    
    # Actually, let's just do the O(n) delta computation per candidate swap.
    # For n=150, this should be fast enough in NumPy.
    
    # Initialize with rank-matching (same as seed, but we'll improve via tabu search)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu search
    tabu_tenure = 20
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    iteration = 0
    start_time = time.time()
    time_limit = 115  # leave some margin
    
    # Precompute a and b as float for speed
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_swap = None
        best_tabu_until = -1
        
        # Evaluate all 2-opt swaps
        for i in range(n):
            for j in range(i + 1, n):
                x = p[i]
                y = p[j]
                if x == y:
                    continue
                
                # Compute delta for swapping p[i] and p[j]
                # Old terms involving i or j:
                # a[i,j]*b[x,y] + a[j,i]*b[y,x]
                # + sum_{k!=i,j} a[i,k]*b[x,p[k]] + a[k,i]*b[p[k],x]
                # + sum_{k!=i,j} a[j,k]*b[y,p[k]] + a[k,j]*b[p[k],y]
                # New terms:
                # a[i,j]*b[y,x] + a[j,i]*b[x,y]
                # + sum_{k!=i,j} a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y]
                # + sum_{k!=i,j} a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
                
                # Vectorized:
                # Let's compute the change efficiently.
                # delta = (a[i,j]*b[y,x] + a[j,i]*b[x,y] - a[i,j]*b[x,y] - a[j,i]*b[y,x])
                #       + sum_{k!=i,j} [a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
                #                         + a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y])]
                
                # Let's compute the sum over k != i, j:
                # For facility i moving from x to y:
                #   sum_k a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
                #   = sum_k a[i,k]*b[y,p[k]] - sum_k a[i,k]*b[x,p[k]]
                #     + sum_k a[k,i]*b[p[k],y] - sum_k a[k,i]*b[p[k],x]
                #   = (a[i,:] * bf[y, p]) - (a[i,:] * bf[x, p])
                #     + (a[:,i] * bf[p, y]) - (a[:,i] * bf[p, x])
                # But this includes k=i and k=j. We need to subtract those.
                
                # This is getting complex. Let's just do the full O(n) loop in NumPy.
                # For k in range(n):
                #   if k == i or k == j: continue
                #   delta += a[i,k]*(bf[y,p[k]] - bf[x,p[k]]) + a[k,i]*(bf[p[k],y] - bf[p[k],x])
                #            + a[j,k]*(bf[x,p[k]] - bf[y,p[k]]) + a[k,j]*(bf[p[k],x] - bf[p[k],y])
                
                # Vectorized over k:
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                ks = np.where(mask)[0]
                pk = p[ks]
                
                # a[i, ks] * (bf[y, pk] - bf[x, pk])
                term1 = af[i, ks] * (bf[y, pk] - bf[x, pk])
                # a[ks, i] * (bf[pk, y] - bf[pk, x])
                term2 = af[ks, i] * (bf[pk, y] - bf[pk, x])
                # a[j, ks] * (bf[x, pk] - bf[y, pk])
                term3 = af[j, ks] * (bf[x, pk] - bf[y, pk])
                # a[ks, j] * (bf[pk, x] - bf[pk, y])
                term4 = af[ks, j] * (bf[pk, x] - bf[pk, y])
                
                sum_k = np.sum(term1) + np.sum(term2) + np.sum(term3) + np.sum(term4)
                
                # Diagonal terms:
                diag_old = af[i, j] * bf[x, y] + af[j, i] * bf[y, x]
                diag_new = af[i, j] * bf[y, x] + af[j, i] * bf[x, y]
                
                delta = diag_new - diag_old + sum_k
                
                # Check tabu
                is_tabu = False
                if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                    is_tabu = True
                if (j, i) in tabu_list and tabu_list[(j, i)] > iteration:
                    is_tabu = True
                
                # Aspiration: allow tabu move if it improves best
                if is_tabu and delta >= 0:
                    continue
                if is_tabu and (current_cost + delta) >= best_cost:
                    continue
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i, j)
        
        if best_swap is None:
            # No improving move found; try any non-tabu move or break
            # Try to find any move
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                        continue
                    if (j, i) in tabu_list and tabu_list[(j, i)] > iteration:
                        continue
                    best_swap = (i, j)
                    break
                if best_swap:
                    break
        
        if best_swap is None:
            break
        
        i, j = best_swap
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update tabu
        tabu_list[(i, j)] = iteration + tabu_tenure
        tabu_list[(j, i)] = iteration + tabu_tenure
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        iteration += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
