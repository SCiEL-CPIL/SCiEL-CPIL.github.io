# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    """
    Compute C[i] = sum_j a[i,j] * b[p[i], p[j]] for all i.
    The total cost is sum(C) / 2 if a is symmetric, but we use sum(C) as a proxy
    for incremental updates. Actually, the cost is sum_{i,j} a[i,j]*b[p[i],p[j]].
    If we define C[i] = sum_j a[i,j]*b[p[i],p[j]], then sum(C) = total cost.
    """
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _refine(a, b, p, C, deadline, best_cost, best_p):
    """
    Best-improvement 2-opt (swap) local search until deadline.
    p is a numpy int64 array (a copy); C is updated in place.
    Returns (best_cost, best_p) found so far.
    """
    n = len(p)
    cur = int(C.sum())
    
    # If current is better, update
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    while time.time() < deadline:
        improved = False
        best_delta = 0
        bi = -1
        bj = -1
        
        # Find best swap
        for i in range(n):
            ai = a[i]
            pi = p[i]
            # C[i] = sum_j ai[j] * b[pi, p[j]]
            # C[j] = sum_k aj[k] * b[p[j], p[k]]
            # After swap i<->j:
            # New C[i] = sum_k ai[k] * b[p[j], p[k]]  (with p[j] at pos i)
            #           = sum_{k!=i} ai[k]*b[p[j],p[k]] + ai[i]*b[p[j],p[j]]
            # New C[j] = sum_k aj[k] * b[pi, p[k]]
            #           = sum_{k!=j} aj[k]*b[pi,p[k]] + aj[j]*b[pi,pi]
            # 
            # Delta = (New C[i] + New C[j]) - (Old C[i] + Old C[j])
            # Old C[i] = sum_{k!=i} ai[k]*b[pi,p[k]] + ai[i]*b[pi,pi]
            # Old C[j] = sum_{k!=j} aj[k]*b[p[j],p[k]] + aj[j]*b[p[j],p[j]]
            #
            # Let's use the standard QAP delta formula:
            # delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
            #       + a[i,j]*(b[p[j],p[j]] - b[pi,pi]) ... wait, this is getting complicated.
            #
            # Simpler: just compute the change in C[i] and C[j].
            # C[i] depends on row i of a and row pi of b (indexed by p).
            # Let's precompute b[p,:] which is the B matrix.
            
            # Actually, the most efficient way for best-improvement is to compute delta for all j
            # But O(n^2) per pass is 10000 operations, which is fine for n=100.
            
            for j in range(i + 1, n):
                # delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
                #       + (a[i,j] + a[j,i]) * (b[p[i],p[j]] - b[p[i],p[i]]) ... no.
                #
                # Let's derive carefully.
                # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
                # Swap i and j.
                # Terms involving i or j change.
                # 1. Terms a[i,i] b[p[i],p[i]] -> a[i,i] b[p[j],p[j]]
                # 2. Terms a[j,j] b[p[j],p[j]] -> a[j,j] b[p[i],p[i]]
                # 3. Terms a[i,j] b[p[i],p[j]] -> a[i,j] b[p[j],p[i]]
                # 4. Terms a[j,i] b[p[j],p[i]] -> a[j,i] b[p[i],p[j]]
                # 5. Terms a[i,k] b[p[i],p[k]] for k!=i,j -> a[i,k] b[p[j],p[k]]
                # 6. Terms a[k,i] b[p[k],p[i]] for k!=i,j -> a[k,i] b[p[k],p[j]]
                # 7. Terms a[j,k] b[p[j],p[k]] for k!=i,j -> a[j,k] b[p[i],p[k]]
                # 8. Terms a[k,j] b[p[k],p[j]] for k!=i,j -> a[k,j] b[p[k],p[i]]
                #
                # Delta = [a[i,i](b[p[j],p[j]] - b[p[i],p[i]]) + 
                #          a[j,j](b[p[i],p[i]] - b[p[j],p[j]]) +
                #          a[i,j](b[p[j],p[i]] - b[p[i],p[j]]) +
                #          a[j,i](b[p[i],p[j]] - b[p[j],p[i]])]
                #        + sum_{k!=i,j} [ a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) +
                #                         a[k,i](b[p[k],p[j]] - b[p[k],p[i]]) +
                #                         a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) +
                #                         a[k,j](b[p[k],p[i]] - b[p[k],p[j]]) ]
                #
                # Assuming a is symmetric (QAP instances usually are), a[i,j]=a[j,i], a[i,k]=a[k,i], etc.
                # Delta = (a[i,i] - a[j,j])(b[p[j],p[j]] - b[p[i],p[i]]) +
                #         2 a[i,j] (b[p[j],p[i]] - b[p[i],p[j]])  # if symmetric, b is symmetric too?
                #         + sum_{k!=i,j} 2 (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                #
                # Wait, b is also symmetric in standard QAP.
                # If b is symmetric, b[p[j],p[i]] = b[p[i],p[j]], so that term is 0.
                # And b[p[j],p[j]] vs b[p[i],p[i]] remains.
                #
                # Let's check if wil100 has symmetric matrices. QAPLIB instances usually have symmetric A and B.
                # If so, delta = (a[i,i] - a[j,j])(b[p[j],p[j]] - b[p[i],p[i]]) + 
                #               2 sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                #
                # Let's implement the general case to be safe, or assume symmetry for speed.
                # Given the score, let's assume symmetry but keep the general formula if possible.
                # Actually, the vectorized delta calculation in the parent program was:
                # dA = a[i, :] - a[j, :]
                # dB = b[p[j], p] - b[p[i], p]
                # dB[i] = 0; dB[j] = 0
                # delta = np.dot(dA, dB)
                # This corresponds to sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                # This misses the diagonal and i-j terms.
                #
                # Let's use the C array approach which is more robust.
                # C[i] = sum_j a[i,j] b[p[i], p[j]]
                # If we swap i and j:
                # New C[i] = sum_j a[i,j] b[p[j], p[j]] ... no, p[i] becomes p[j].
                # New C[i] = sum_j a[i,j] b[p[j], p[j]] is wrong. p is the permutation.
                # New C[i] = sum_j a[i,j] b[new_p[i], new_p[j]] where new_p[i]=old_p[j], new_p[j]=old_p[i].
                # New C[i] = a[i,i] b[p[j],p[j]] + a[i,j] b[p[j],p[i]] + sum_{k!=i,j} a[i,k] b[p[j],p[k]]
                # New C[j] = a[j,j] b[p[i],p[i]] + a[j,i] b[p[i],p[j]] + sum_{k!=i,j} a[j,k] b[p[i],p[k]]
                #
                # Old C[i] = a[i,i] b[p[i],p[i]] + a[i,j] b[p[i],p[j]] + sum_{k!=i,j} a[i,k] b[p[i],p[k]]
                # Old C[j] = a[j,j] b[p[j],p[j]] + a[j,i] b[p[j],p[i]] + sum_{k!=i,j} a[j,k] b[p[j],p[k]]
                #
                # Delta C[i] + Delta C[j] = 
                #   a[i,i](b[p[j],p[j]] - b[p[i],p[i]]) + a[j,j](b[p[i],p[i]] - b[p[j],p[j]]) +
                #   a[i,j](b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i](b[p[i],p[j]] - b[p[j],p[i]]) +
                #   sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]]) +
                #   sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[k],p[j]] - b[p[k],p[i]])  # Wait, C[j] uses a[j,k] b[p[i],p[k]]
                #
                # The last two sums are:
                # sum_{k!=i,j} a[i,k] b[p[j],p[k]] - a[i,k] b[p[i],p[k]] - a[j,k] b[p[i],p[k]] + a[j,k] b[p[j],p[k]]
                # = sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                #
                # So total Delta = (a[i,i] - a[j,j])(b[p[j],p[j]] - b[p[i],p[i]]) +
                #                  (a[i,j] - a[j,i])(b[p[j],p[i]] - b[p[i],p[j]]) +
                #                  sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                #
                # If a is symmetric, a[i,j]=a[j,i], so the second term vanishes.
                # Then Delta = (a[i,i] - a[j,j])(b[p[j],p[j]] - b[p[i],p[i]]) +
                #              sum_{k!=i,j} (a[i,k] - a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
                #
                # Note that the sum includes k where b[p[j],p[k]] - b[p[i],p[k]] is 0 if k is such that... no.
                #
                # Let's implement this.
                
                pi = p[i]
                pj = p[j]
                
                # Diagonal terms
                delta = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
                
                # Off-diagonal i-j terms
                delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
                
                # Rest
                # sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
                # We can compute this as dot product, zeroing out i and j.
                dA = a[i, :] - a[j, :]
                dB = b[pj, :] - b[pi, :]
                # dB is a vector of length n, but we need b[pj, p[k]] - b[pi, p[k]]
                # So dB_k = b[pj, p[k]] - b[pi, p[k]]
                # This is (b[pj, :] - b[pi, :])[p]
                dB_vec = (b[pj, :] - b[pi, :])[p]
                dB_vec[i] = 0
                dB_vec[j] = 0
                delta += np.dot(dA, dB_vec)
                
                if delta < best_delta:
                    best_delta = delta
                    bi = i
                    bj = j
        
        if bi != -1:
            # Perform swap
            p[bi], p[bj] = p[bj], p[bi]
            cur += best_delta
            # Update C
            # C[bi] and C[bj] need to be updated.
            # C[i] = sum_j a[i,j] b[p[i], p[j]]
            # We can recompute C[bi] and C[bj] or update them incrementally.
            # C[bi] = sum_j a[bi,j] b[p[bi], p[j]]
            # C[bj] = sum_j a[bj,j] b[p[bj], p[j]]
            # Since only p[bi] and p[bj] changed, we can update C for all i?
            # No, C[i] depends on p[i] and all p[j].
            # If we swap p[i] and p[j], C[k] changes for all k.
            # C[k] = sum_l a[k,l] b[p[k], p[l]]
            # The term b[p[k], p[l]] changes only if p[k] or p[l] is pi or pj.
            # This is complex. It's easier to recompute C from scratch or use the delta to update cur.
            # We already have cur. We can recompute C if needed for next iteration.
            # But recomputing C is O(n^2). With n=100, it's 10000 ops.
            # Best-improvement pass is O(n^2) swaps, each O(n) delta. Total O(n^3) = 1M ops per pass.
            # Recomputing C is 10k ops. So recompute C every pass?
            # Or just keep cur and recompute C only when we need to?
            # Actually, we don't need C for the delta calculation if we use the formula above.
            # We only need C to initialize cur.
            # So we can skip updating C entirely and just track cur.
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
            improved = True
        
        if not improved:
            break
            
    return best_cost, best_p


def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost and C
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # 2. Local Search (2-opt) with best improvement
    # We will run multiple passes of best-improvement 2-opt.
    # To escape local minima, we can use Iterated Local Search (ILS):
    # - Perturb the solution (random swaps)
    # - Run local search
    # - Accept if better or with some probability
    
    # Let's do ILS.
    T = 5000.0
    T_end = 1.0
    alpha = 0.999
    
    # First, refine the initial solution
    C = _compute_C(a, b, p)
    best_cost, best_p = _refine(a, b, p, C, deadline, best_cost, best_p)
    
    # Now ILS
    while time.time() < deadline:
        # Perturb: random number of swaps
        n_swaps = np.random.randint(2, 10)
        p_perturbed = best_p.copy()
        for _ in range(n_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
        
        # Refine perturbed solution
        C_pert = _compute_C(a, b, p_perturbed)
        cost_pert, p_pert = _refine(a, b, p_perturbed, C_pert, deadline, best_cost, best_p)
        
        # Update best
        if cost_pert < best_cost:
            best_cost = cost_pert
            best_p = p_pert.copy()
            T = max(T * alpha, T_end)  # Reset temperature if improved
        else:
            # Accept with probability exp(-delta/T)
            delta = cost_pert - best_cost
            if np.random.rand() < np.exp(-delta / T):
                best_p = p_pert.copy()
                best_cost = cost_pert
            T *= alpha
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
