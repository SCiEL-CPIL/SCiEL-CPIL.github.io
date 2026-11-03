"""
Vectorized Batch Simulated Annealing with Spectral Initialization for QAPLIB wil100.
Strategy:
1. Spectral Initialization: Project facilities and locations onto the top eigenvectors
   of the symmetrized flow and distance matrices. Match ranks of projections to create
   a geometrically informed initial permutation.
2. Vectorized Batch Simulated Annealing:
   - Generate a batch of K random swap pairs (i, j) per step.
   - Compute the cost delta for all K swaps in parallel using NumPy vectorization.
   - Accept swaps based on the current temperature T.
   - Cool T over time.
3. This approach amortizes the overhead of Python loops by performing many evaluations
   in a single vectorized operation, allowing for a much higher number of effective
   moves within the time limit compared to scalar 2-opt/ILS.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral decomposition.
    Aligns eigenvectors of the flow matrix with eigenvectors of the distance matrix.
    """
    n = len(a)
    
    # Symmetrize for eigenvalue decomposition
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Compute eigenvectors
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(a_sym)
        eigenvalues_b, eigenvectors_b = np.linalg.eigh(b_sym)
        
        # Sort by eigenvalue (largest magnitude first for alignment)
        idx_a = np.argsort(np.abs(eigenvalues_a))[::-1]
        idx_b = np.argsort(np.abs(eigenvalues_b))[::-1]
        
        # Take the top eigenvector
        v_a = eigenvectors_a[:, idx_a[0]]
        v_b = eigenvectors_b[:, idx_b[0]]
        
        # Rank matching: 
        # Sort facilities by their projection on v_a
        # Sort locations by their projection on v_b
        # Assign facility with rank r to location with rank r
        
        # To handle the "high flow to high distance" intuition correctly:
        # In QAP, we want to minimize sum a_ij * b_pi_pj.
        # If a and b have similar spectral structure, aligning the principal components
        # (both positive to positive, negative to negative) tends to be good.
        
        facility_scores = v_a
        location_scores = v_b
        
        facility_order = np.argsort(facility_scores, kind="stable")
        location_order = np.argsort(location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        # facility_order[k] is the facility index with k-th smallest score
        # location_order[k] is the location index with k-th smallest score
        # We assign facility facility_order[k] to location location_order[k]
        p[facility_order] = location_order
        
        # Refine with second eigenvector? 
        # For simplicity and robustness, stick to top eigenvector for init.
        # The SA will do the heavy lifting.
        
    except np.linalg.LinAlgError:
        # Fallback to rank matching on row sums
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
    return p

def _vectorized_sa(a, b, p, time_limit, start_time, batch_size=2000):
    """
    Vectorized Simulated Annealing.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    T_start = 1000.0
    T_end = 1.0
    
    # Precompute row/col sums? No, we need full delta.
    # Delta for swapping i and j:
    # delta = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj]) 
    #         + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #         + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
    #         + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])
    
    # Vectorized approach for a batch of swaps (I, J):
    # pi = p[I], pj = p[J]
    
    # Term 1: (a[I,J] - a[J,I]) * (b[pj,pi] - b[pi,pj])
    # Term 2: a[I,I]*(b[pj,pj]-b[pi,pi]) + a[J,J]*(b[pi,pi]-b[pj,pj])
    # Term 3: sum_k (a[I,k]-a[J,k]) * (b[pj,p[k]] - b[pi,p[k]])
    # Term 4: sum_k (a[k,I]-a[k,J]) * (b[p[k],pj] - b[p[k],pi])
    
    # We compute over all k and subtract k=I and k=J contributions.
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / max(time_limit, 1.0))
        T = T_start * (T_end / T_start) ** frac
        
        # Generate batch of random swaps
        I = np.random.randint(0, n, batch_size)
        J = np.random.randint(0, n, batch_size)
        
        # Exclude i==j
        mask = I != J
        if not np.any(mask):
            continue
        I = I[mask]
        J = J[mask]
        
        pi = p[I]
        pj = p[J]
        
        # Term 1
        t1 = (a[np.ix_(I, J)] - a[np.ix_(J, I)]) * (b[np.ix_(pj, pi)] - b[np.ix_(pi, pj)])
        
        # Term 2
        t2 = a[np.ix_(I, I)] * (b[np.ix_(pj, pj)] - b[np.ix_(pi, pi)]) + \
             a[np.ix_(J, J)] * (b[np.ix_(pi, pi)] - b[np.ix_(pj, pj)])
        
        # Term 3: sum over k of (a[I,k] - a[J,k]) * (b[pj, p[k]] - b[pi, p[k]])
        # a_diff_row = a[I,:] - a[J,:]  (shape: batch x n)
        # b_diff_col = b[pj, p] - b[pi, p] (shape: batch x n)
        a_diff_row = a[I, :] - a[J, :]
        b_diff_col = b[np.ix_(pj, p)] - b[np.ix_(pi, p)]
        sum_t3 = np.sum(a_diff_row * b_diff_col, axis=1)
        
        # Subtract k=I and k=J
        # k=I: (a[I,I] - a[J,I]) * (b[pj, pi] - b[pi, pi])
        sub_t3_1 = (a[np.ix_(I, I)] - a[np.ix_(J, I)]) * (b[np.ix_(pj, pi)] - b[np.ix_(pi, pi)])
        # k=J: (a[I,J] - a[J,J]) * (b[pj, pj] - b[pi, pj])
        sub_t3_2 = (a[np.ix_(I, J)] - a[np.ix_(J, J)]) * (b[np.ix_(pj, pj)] - b[np.ix_(pi, pj)])
        t3 = sum_t3 - sub_t3_1 - sub_t3_2
        
        # Term 4: sum over k of (a[k,I] - a[k,J]) * (b[p[k], pj] - b[p[k], pi])
        a_diff_col = a[:, I] - a[:, J]
        b_diff_row = b[np.ix_(p, pj)] - b[np.ix_(p, pi)]
        sum_t4 = np.sum(a_diff_col * b_diff_row, axis=0)
        
        # Subtract k=I and k=J
        # k=I: (a[I,I] - a[I,J]) * (b[pi, pj] - b[pi, pi])
        sub_t4_1 = (a[np.ix_(I, I)] - a[np.ix_(I, J)]) * (b[np.ix_(pi, pj)] - b[np.ix_(pi, pi)])
        # k=J: (a[J,I] - a[J,J]) * (b[pj, pj] - b[pj, pi])
        sub_t4_2 = (a[np.ix_(J, I)] - a[np.ix_(J, J)]) * (b[np.ix_(pj, pj)] - b[np.ix_(pj, pi)])
        t4 = sum_t4 - sub_t4_1 - sub_t4_2
        
        delta = t1 + t2 + t3 + t4
        
        # Acceptance
        # delta < 0 is always accepted
        # delta > 0 accepted with prob exp(-delta/T)
        
        accept_mask = np.zeros(len(delta), dtype=bool)
        neg_mask = delta < 0
        accept_mask[neg_mask] = True
        
        pos_mask = ~neg_mask
        if np.any(pos_mask):
            r = np.random.rand(np.sum(pos_mask))
            prob = np.exp(-delta[pos_mask] / max(T, 1e-10))
            pos_accept = r < prob
            # Map back to full mask
            # This is tricky to vectorize cleanly without indices, so we do it simply
            pos_indices = np.where(pos_mask)[0]
            accept_mask[pos_indices[pos_accept]] = True
        
        # Apply accepted swaps
        # Note: If we accept multiple swaps that share indices, the cost calculation for subsequent swaps in the batch
        # is invalid. However, for SA, this is a common approximation if the batch is small relative to n, 
        # or we can just apply them sequentially in a loop over the accepted subset.
        # Given n=100 and batch=2000, collisions are very likely.
        # To be safe and correct, we should apply swaps one by one, but that defeats the purpose.
        # Alternative: Accept only swaps that don't conflict? Or just take the first accepted swap?
        # Let's take a subset of non-conflicting swaps.
        
        # Identify unique indices involved in accepted swaps
        # This is complex. A simpler, robust approach for vectorized SA in QAP:
        # Use a smaller batch size or just pick ONE swap from the batch that is accepted.
        # But that's slow.
        
        # Better: Apply swaps sequentially in a Python loop over the accepted ones.
        # Since we computed all deltas, we just iterate.
        # If 10% of 2000 are accepted, that's 200 swaps. 200 Python iterations is fine.
        
        accepted_indices = np.where(accept_mask)[0]
        
        # To avoid conflicts, we can just process them. If a swap conflicts with a previous one,
        # the delta is stale. But for a "batch" heuristic, we can just apply the first N non-conflicting.
        
        used = set()
        applied_count = 0
        for idx in accepted_indices:
            i = I[idx]
            j = J[idx]
            if i in used or j in used:
                continue
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta[idx])
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            used.add(i)
            used.add(j)
            applied_count += 1
            if applied_count > 50: # Limit to prevent too many updates per step
                break
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Spectral Initialization
    p_spec = _spectral_initial(a, b)
    # Quick local search or just SA?
    # Let's do a short SA from spectral start
    p_sa, cost_sa = _vectorized_sa(a, b, p_spec, 10.0, start_time, batch_size=1000)
    best_p = p_sa.copy()
    best_cost = cost_sa
    
    # Phase 2: Random Restarts with Vectorized SA
    # Try a few random starts
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 30:
        for r in range(5):
            if time.time() - start_time > total_deadline - 20:
                break
            p_rand = np.random.permutation(n).astype(np.int64)
            sa_time = min(15.0, total_deadline - (time.time() - start_time) - 10)
            if sa_time < 2.0:
                break
            p_sa, cost_sa = _vectorized_sa(a, b, p_rand, sa_time, start_time, batch_size=1000)
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
                
    # Phase 3: Final Polish with longer SA from best
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 5:
        sa_time = total_deadline - elapsed - 3
        p_final, cost_final = _vectorized_sa(a, b, best_p.copy(), sa_time, start_time, batch_size=2000)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
