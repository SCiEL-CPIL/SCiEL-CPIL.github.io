"""Search-based solver for QAPLIB wil100: Quantum-inspired Annealing with Unitary Representations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Initialize permutation using spectral embedding.
    Aligns eigenvectors of A (flow) and B (distance) to minimize cost.
    """
    n = a.shape[0]
    
    # Use symmetric parts to ensure real eigenvectors and stable ordering
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b_sym)
        
        # Sort by eigenvalue magnitude (descending) to align principal structures
        idx_a = np.argsort(np.abs(w_a))[::-1]
        idx_b = np.argsort(np.abs(w_b))[::-1]
        
        vecs_a = v_a[:, idx_a]
        vecs_b = v_b[:, idx_b]
        
        # Flip signs to ensure alignment (dot product positive)
        for k in range(n):
            if np.dot(vecs_a[:, k], vecs_b[:, k]) < 0:
                vecs_b[:, k] = -vecs_b[:, k]
                
        similarity = vecs_a @ vecs_b.T
        
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-similarity)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
        
    except Exception:
        return np.random.permutation(n)

def _quantum_swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    This is the 'gradient' signal used by the quantum-inspired annealer.
    """
    pi = p[i]
    pj = p[j]
    
    # Delta = sum_{k} [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])]
    #        + [a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]
    #        + diagonal and cross terms
    
    # Vectorized computation for a single pair (i, j)
    # Note: b is the distance matrix, p is the permutation.
    # b[pj, p[k]] is b[p[j], p[k]]
    
    # Let's compute the delta efficiently.
    # Term 1: Interaction of i with all k
    # Term 2: Interaction of j with all k
    # Term 3: Interaction between i and j
    
    # Using NumPy for speed
    # b_pi = b[p[i], p]
    # b_pj = b[p[j], p]
    
    # But we need to be careful with indices.
    
    # Delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    #       + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    #       + (a[i,i] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[i]])
    #       + (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    
    # Let's just compute the cost difference directly for clarity and correctness, 
    # though it's O(n). For n=100, this is fast enough if we don't do it for ALL pairs.
    
    # However, for the 'quantum' step, we might want to evaluate a batch of moves.
    # For now, let's stick to a single swap evaluation which is O(n).
    
    # Actually, let's use the precomputed row sums idea from inspiration 3 but adapted.
    # Or just compute the delta directly.
    
    # Direct computation:
    # old_cost = sum a[u,v] b[p[u], p[v]]
    # new_cost = ...
    
    # It is safer to compute the delta incrementally.
    
    # Let's use the formula:
    # Delta = sum_{k} a[i,k] (b[pj, p[k]] - b[pi, p[k]]) + sum_{k} a[k,i] (b[p[k], pj] - b[p[k], pi])
    #       + sum_{k} a[j,k] (b[pi, p[k]] - b[pj, p[k]]) + sum_{k} a[k,j] (b[p[k], pi] - b[p[k], pj])
    #       - 2 * (terms involving both i and j in the sums above to avoid double counting? No, the sums are over all k)
    
    # Wait, the sums above include k=i and k=j.
    # If k=i:
    #   a[i,i] (b[pj, pi] - b[pi, pi]) + a[i,i] (b[pi, pj] - b[pi, pi]) ... this is getting messy.
    
    # Let's just recompute the cost for the affected rows/cols.
    # The cost contribution of facility i is sum_j a[i,j] b[pi, p[j]]
    # The cost contribution of facility j is sum_i a[j,i] b[pj, p[i]]
    
    # Actually, the simplest robust way for n=100 is:
    # 1. Swap in a copy of p? No, that's O(n^2) to recompute cost.
    # 2. Use the O(n) delta formula.
    
    # Let's derive the O(n) delta formula carefully.
    # C(p) = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    # Swap p[i] and p[j]. Let pi = p[i], pj = p[j].
    # New permutation p' is identical to p except p'[i] = pj, p'[j] = pi.
    
    # Change in cost:
    # 1. Terms where u=i or v=i or u=j or v=j.
    
    # Let's group by pairs (u,v):
    # - (i, i): a[i,i] (b[pj,pj] - b[pi,pi])
    # - (j, j): a[j,j] (b[pi,pi] - b[pj,pj])
    # - (i, j): a[i,j] (b[pj,pi] - b[pi,pj])
    # - (j, i): a[j,i] (b[pi,pj] - b[pj,pi])
    # - (i, k) for k != i,j: a[i,k] (b[pj, p[k]] - b[pi, p[k]])
    # - (k, i) for k != i,j: a[k,i] (b[p[k], pj] - b[p[k], pi])
    # - (j, k) for k != i,j: a[j,k] (b[pi, p[k]] - b[pj, p[k]])
    # - (k, j) for k != i,j: a[k,j] (b[p[k], pi] - b[p[k], pj])
    
    # We can vectorize this.
    
    b_pi = b[pi, :]
    b_pj = b[pj, :]
    b_col_pi = b[:, pi]
    b_col_pj = b[:, pj]
    
    # Terms 1-4
    t1 = a[i,i] * (b[pj,pj] - b[pi,pi])
    t2 = a[j,j] * (b[pi,pi] - b[pj,pj])
    t3 = a[i,j] * (b[pj,pi] - b[pi,pj])
    t4 = a[j,i] * (b[pi,pj] - b[pj,pi])
    
    # Terms 5-8
    # Sum over k != i, j
    # a[i,k] (b[pj, p[k]] - b[pi, p[k]]) = a[i,k] (b_pj[p[k]] - b_pi[p[k]])
    # a[k,i] (b[p[k], pj] - b[p[k], pi]) = a[k,i] (b_col_pj[p[k]] - b_col_pi[p[k]])
    # a[j,k] (b[pi, p[k]] - b[pj, p[k]]) = a[j,k] (b_pi[p[k]] - b_pj[p[k]])
    # a[k,j] (b[p[k], pi] - b[p[k], pj]) = a[k,j] (b_col_pi[p[k]] - b_col_pj[p[k]])
    
    # Let diff_row = b_pj - b_pi  (vector of length n, indexed by location)
    # Let diff_col = b_col_pj - b_col_pi
    
    # We need to sum over k.
    # p[k] is the location.
    
    # S1 = sum_{k != i,j} a[i,k] * diff_row[p[k]]
    # S2 = sum_{k != i,j} a[k,i] * diff_col[p[k]]
    # S3 = sum_{k != i,j} a[j,k] * (-diff_row[p[k]])
    # S4 = sum_{k != i,j} a[k,j] * (-diff_col[p[k]])
    
    # Note: diff_row[p[k]] is the value at index p[k] in the vector diff_row.
    
    # Let's create arrays indexed by facility k.
    # loc_diff_row[k] = diff_row[p[k]]
    # loc_diff_col[k] = diff_col[p[k]]
    
    loc_diff_row = diff_row[p]
    loc_diff_col = diff_col[p]
    
    # Now compute sums excluding i and j
    # S1 = sum_{k != i,j} a[i,k] * loc_diff_row[k]
    # S2 = sum_{k != i,j} a[k,i] * loc_diff_col[k]
    # S3 = - sum_{k != i,j} a[j,k] * loc_diff_row[k]
    # S4 = - sum_{k != i,j} a[k,j] * loc_diff_col[k]
    
    # We can compute the full sum and subtract the i and j terms.
    
    s1_full = np.dot(a[i, :], loc_diff_row)
    s2_full = np.dot(a[:, i], loc_diff_col)
    s3_full = np.dot(a[j, :], loc_diff_row)
    s4_full = np.dot(a[:, j], loc_diff_col)
    
    # Subtract k=i and k=j terms
    # For S1:
    s1_i = a[i,i] * loc_diff_row[i]
    s1_j = a[i,j] * loc_diff_row[j]
    
    # For S2:
    s2_i = a[i,i] * loc_diff_col[i]
    s2_j = a[j,i] * loc_diff_col[j]
    
    # For S3:
    s3_i = a[j,i] * loc_diff_row[i]
    s3_j = a[j,j] * loc_diff_row[j]
    
    # For S4:
    s4_i = a[i,j] * loc_diff_col[i]
    s4_j = a[j,j] * loc_diff_col[j]
    
    s1 = s1_full - s1_i - s1_j
    s2 = s2_full - s2_i - s2_j
    s3 = s3_full - s3_i - s3_j
    s4 = s4_full - s4_i - s4_j
    
    delta = t1 + t2 + t3 + t4 + s1 + s2 - s3 - s4
    
    return delta

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start_time = time.time()
    
    np.random.seed(42)
    
    # 1. Spectral Initialization
    p = _spectral_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Quantum-inspired Annealing
    # We use a 'quantum' temperature that allows tunneling (accepting bad moves with higher probability)
    # and a 'rotation' step size that adapts.
    
    t0 = 1000.0
    t_min = 1.0
    
    # Precompute nothing specific for the quantum step, but we can cache some b rows?
    # For n=100, b[pi, :] is cheap.
    
    # To speed up, we can try to vectorize the selection of moves.
    # However, the delta calculation is O(n) and depends on the specific pair (i, j).
    # Let's try a 'batch' approach: evaluate a random set of K swaps, pick the best one (or accept probabilistically).
    
    K = 20  # Number of candidate swaps to evaluate per step
    
    while time.time() - start_time < deadline:
        elapsed = time.time() - start_time
        frac = elapsed / deadline
        temp = t0 * (t_min / t0) ** frac
        
        # Generate K random swap candidates
        # Ensure i != j
        i_candidates = np.random.randint(n, size=K)
        j_candidates = np.random.randint(n, size=K)
        
        # Filter out i == j
        valid = i_candidates != j_candidates
        if not np.any(valid):
            continue
            
        i_candidates = i_candidates[valid]
        j_candidates = j_candidates[valid]
        
        # Evaluate deltas for these candidates
        # This is a loop in Python, but K is small (20).
        # 20 * O(n) = 20 * 100 = 2000 ops. Very fast.
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        for k in range(len(i_candidates)):
            i = i_candidates[k]
            j = j_candidates[k]
            d = _quantum_swap_delta(a, b, p, i, j)
            if d < best_delta:
                best_delta = d
                best_i = i
                best_j = j
        
        if best_i == -1:
            continue
            
        # Acceptance criterion
        # Standard SA: accept if d < 0 or random < exp(-d/T)
        # Quantum-inspired: we can increase the probability of accepting bad moves slightly,
        # or just use standard SA which already allows it.
        # The 'quantum' aspect here is the name and the potential for future enhancement (e.g., using complex amplitudes).
        # For now, standard SA acceptance with the fast delta.
        
        if best_delta < 0 or np.random.random() < np.exp(-best_delta / max(temp, 1e-10)):
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Check time
        if time.time() - start_time > deadline - 1:
            break

    return best_p.tolist()

# EVOLVE-BLOCK-END
