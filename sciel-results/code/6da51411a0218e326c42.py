"""QAP solver for wil100: spectral init, vectorized delta-matrix local search, ILS with block moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _qap_cost(a, b, p):
    """Compute exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] = cost change if we swap positions i and j in permutation p.
    O(n^3) via matrix multiplications, fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] * b[p[k], p[j]]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] * b[p[k], p[j]]
    M2 = a @ Bp
    
    # Swap delta formula:
    # delta(i,j) = sum_k [(a[i,k]-a[j,k])*(b[p[j],p[k]] - b[p[i],p[k]]) + (a[k,i]-a[k,j])*(b[p[k],p[j]] - b[p[k],p[i]])]
    #           - (a[i,j]-a[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])  [correction for k=i,j terms]
    
    # Let's use the standard formula:
    # delta(i,j) = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    #             + (a[i,j] - a[j,i]) * (b[p[i], p[j]] - b[p[j], p[i]])
    #             + (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    # Wait, let me re-derive more carefully.
    
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # After swapping p[i] and p[j]:
    # delta = sum_k a[i,k]*(b[p[j],p[k]]-b[p[i],p[k]]) + sum_k a[k,i]*(b[p[k],p[j]]-b[p[k],p[i]])
    #       + sum_k a[j,k]*(b[p[i],p[k]]-b[p[j],p[k]]) + sum_k a[k,j]*(b[p[k],p[i]]-b[p[k],p[j]])
    #       - a[i,j]*(b[p[i],p[j]]-b[p[j],p[i]]) - a[j,i]*(b[p[j],p[i]]-b[p[i],p[j]])
    #       (the last terms correct for double-counting when k=i or k=j)
    
    # Let's define:
    # S1[i,j] = sum_k a[i,k] * b[p[j],p[k]]
    # S2[i,j] = sum_k a[i,k] * b[p[i],p[k]]
    # S3[i,j] = sum_k a[k,i] * b[p[k],p[j]]
    # S4[i,j] = sum_k a[k,i] * b[p[k],p[i]]
    
    # delta(i,j) = S1[i,j] - S2[i,j] + S3[i,j] - S4[i,j] + S1[j,i] - S2[j,i] + S3[j,i] - S4[j,i]
    #             - a[i,j]*(b[p[i],p[j]]-b[p[j],p[i]]) - a[j,i]*(b[p[j],p[i]]-b[p[i],p[j]])
    
    # But S1[i,j] = sum_k a[i,k] * b[p[j],p[k]] = (a @ Bp)[i, j] where Bp[k,j] = b[p[k],p[j]]
    # Wait, Bp is defined as b[np.ix_(p,p)], so Bp[k,j] = b[p[k], p[j]].
    # So (a @ Bp)[i,j] = sum_k a[i,k] * b[p[k], p[j]] = S2[i,j] (with j as the second index in b)
    
    # Let me redefine more carefully.
    # Let Bp[k, m] = b[p[k], p[m]]
    # Then (a @ Bp)[i, m] = sum_k a[i,k] * b[p[k], p[m]]
    # And (a.T @ Bp)[i, m] = sum_k a[k,i] * b[p[k], p[m]]
    
    # For swap of positions i and j:
    # delta(i,j) = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #            + sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #            + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #            + sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #            - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])
    #            - a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    
    # Note: b[p[j],p[k]] = Bp[j,k], b[p[i],p[k]] = Bp[i,k]
    # b[p[k],p[j]] = Bp[k,j], b[p[k],p[i]] = Bp[k,i]
    
    # Term 1: sum_k a[i,k]*Bp[j,k] - sum_k a[i,k]*Bp[i,k] = (a @ Bp)[i,j] - (a @ Bp)[i,i]
    # Term 2: sum_k a[k,i]*Bp[k,j] - sum_k a[k,i]*Bp[k,i] = (a.T @ Bp)[i,j] - (a.T @ Bp)[i,i]
    # Term 3: sum_k a[j,k]*Bp[i,k] - sum_k a[j,k]*Bp[j,k] = (a @ Bp)[j,i] - (a @ Bp)[j,j]
    # Term 4: sum_k a[k,j]*Bp[k,i] - sum_k a[k,j]*Bp[k,j] = (a.T @ Bp)[j,i] - (a.T @ Bp)[j,j]
    
    # Correction terms:
    # - a[i,j]*(Bp[i,j] - Bp[j,i])
    # - a[j,i]*(Bp[j,i] - Bp[i,j])
    
    M_a_Bp = a @ Bp
    M_at_Bp = a.T @ Bp
    
    D = np.zeros((n, n), dtype=np.int64)
    
    # Vectorized computation
    # D[i,j] = (M_a_Bp[i,j] - M_a_Bp[i,i]) + (M_at_Bp[i,j] - M_at_Bp[i,i])
    #         + (M_a_Bp[j,i] - M_a_Bp[j,j]) + (M_at_Bp[j,i] - M_at_Bp[j,j])
    #         - a[i,j]*(Bp[i,j] - Bp[j,i]) - a[j,i]*(Bp[j,i] - Bp[i,j])
    
    diag_a_Bp = np.diag(M_a_Bp)
    diag_at_Bp = np.diag(M_at_Bp)
    
    # First two terms
    T1 = M_a_Bp - diag_a_Bp[np.newaxis, :]
    T2 = M_at_Bp - diag_at_Bp[np.newaxis, :]
    
    # Third and fourth terms (transposed)
    T3 = M_a_Bp.T - diag_a_Bp[:, np.newaxis]
    T4 = M_at_Bp.T - diag_at_Bp[:, np.newaxis]
    
    D = T1 + T2 + T3 + T4
    
    # Correction terms
    diff_Bp = Bp - Bp.T
    D -= a * diff_Bp
    D -= a.T * (-diff_Bp)  # a[j,i]*(Bp[j,i]-Bp[i,j]) = -a[j,i]*(Bp[i,j]-Bp[j,i]) = -a.T * diff_Bp... 
    # Wait: a[j,i]*(Bp[j,i] - Bp[i,j]) = a.T[i,j] * (Bp[j,i] - Bp[i,j]) = a.T[i,j] * (-diff_Bp[i,j])
    # So the full correction is: -a[i,j]*diff_Bp[i,j] - a.T[i,j]*(-diff_Bp[i,j]) = -a[i,j]*diff_Bp[i,j] + a.T[i,j]*diff_Bp[i,j]
    # Let me redo:
    # correction = -a[i,j]*(Bp[i,j]-Bp[j,i]) - a[j,i]*(Bp[j,i]-Bp[i,j])
    #            = -a[i,j]*diff_Bp[i,j] - a[j,i]*(-diff_Bp[i,j])
    #            = -a[i,j]*diff_Bp[i,j] + a[j,i]*diff_Bp[i,j]
    #            = (a[j,i] - a[i,j]) * diff_Bp[i,j]
    #            = (a.T - a)[i,j] * diff_Bp[i,j]
    
    D -= (a * diff_Bp)
    D += (a.T * diff_Bp)
    
    # Set diagonal to 0 (swapping with self is no-op)
    np.fill_diagonal(D, 0)
    
    return D

def _local_search(a, b, p, max_iters=50):
    """Steepest-descent local search using vectorized delta matrix."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    for it in range(max_iters):
        D = _compute_delta_matrix(a, b, p)
        
        # Find the most negative delta
        min_val = np.min(D)
        if min_val >= 0:
            break
        
        # Find the pair (i, j) with the minimum delta
        idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = idx
        
        # Apply the swap
        p[i], p[j] = p[j], p[i]
        current_cost += int(min_val)
    
    return p, current_cost

def _block_move_delta(a, b, p, block_start, block_len, new_pos):
    """
    Compute the cost delta for moving a contiguous block of facilities from
    positions [block_start, block_start+block_len) to position new_pos.
    This is an Or-opt style move.
    """
    n = len(p)
    if block_len <= 0:
        return 0
    if new_pos == block_start:
        return 0
    
    # Extract the block
    block = p[block_start:block_start+block_len]
    
    # Simulate the move
    p_new = p.copy()
    if new_pos > block_start:
        # Remove block, then insert at new_pos (adjusted)
        before = p_new[:block_start]
        after_block = p_new[block_start+block_len:]
        # new_pos is in the original indexing, so after removal, it shifts
        insert_pos = new_pos - block_len if new_pos >= block_start + block_len else new_pos
        p_new = np.concatenate([before, after_block[:insert_pos], block, after_block[insert_pos:]])
    else:
        before = p_new[:new_pos]
        between = p_new[new_pos:block_start]
        after_block = p_new[block_start+block_len:]
        p_new = np.concatenate([before, block, between, after_block])
    
    # Compute cost difference
    old_cost = _qap_cost(a, b, p)
    new_cost = _qap_cost(a, b, p_new)
    return new_cost - old_cost

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    # Spectral initialization
    try:
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        facility_scores = v_a
        location_scores = v_b
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
    except Exception:
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # ILS loop
    while time.time() - start_time < timeout:
        # Local search
        p, current_cost = _local_search(a, b, p, max_iters=30)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() - start_time > timeout - 5:
            break
        
        # Perturbation: random swaps + occasional block moves
        num_swaps = max(2, n // 15)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            p[i], p[j] = p[j], p[i]
        
        # Occasional Or-opt block move
        if np.random.random() < 0.3:
            block_len = np.random.choice([2, 3, 4])
            block_start = np.random.randint(0, n - block_len + 1)
            new_pos = np.random.randint(0, n - block_len + 1)
            if new_pos != block_start and abs(new_pos - block_start) > block_len:
                # Apply the block move
                block = p[block_start:block_start+block_len].copy()
                if new_pos > block_start:
                    before = p[:block_start]
                    after_block = p[block_start+block_len:]
                    insert_pos = new_pos - block_len
                    p = np.concatenate([before, after_block[:insert_pos], block, after_block[insert_pos:]])
                else:
                    before = p[:new_pos]
                    between = p[new_pos:block_start]
                    after_block = p[block_start+block_len:]
                    p = np.concatenate([before, block, between, after_block])
        
        # Update cost
        current_cost = _qap_cost(a, b, p)
        
        # If perturbation made things worse, revert to best
        if current_cost > best_cost * 1.5:
            p = best_p.copy()
            current_cost = best_cost
    
    # Final polish
    if time.time() - start_time < timeout - 2:
        p, current_cost = _local_search(a, b, p, max_iters=50)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
