# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_swap_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of deltas for swapping positions i and j.
    Delta[i, j] = cost after swapping p[i] and p[j] - current cost.
    """
    n = len(p)
    
    # C[i] = sum_k a[i, k] * b[p[i], p[k]]
    # This is the contribution of facility i's row to the total cost.
    # Total cost = sum_i C[i].
    B = b[np.ix_(p, p)]
    C = (a @ B).astype(np.int64)
    
    # To swap i and j, we replace p[i] with p[j] and p[j] with p[i].
    # Let pi = p[i], pj = p[j].
    # New C'[i] = sum_k a[i, k] * b[pj, p[k]] (with p[i] replaced by pj in the permutation used for B)
    # Note: B is constructed from p. If we swap i, j in p, the new B' differs from B only in rows/cols i, j.
    
    # Actually, it's easier to compute the change in total cost directly.
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    # Delta for swap (i, j):
    # Terms involving i or j change.
    # Old terms with i: sum_k a[i,k] b[pi, p[k]] + sum_k a[k,i] b[p[k], pi] - a[i,i] b[pi, pi]
    # New terms with i: sum_k a[i,k] b[pj, p[k]] + sum_k a[k,i] b[p[k], pj] - a[i,i] b[pj, pj]
    # (Note: when k=j, p[k] is pj in old, pi in new? No, p is fixed during delta calc, we imagine the swap)
    
    # Let's use the vectorized approach from the inspiration which is known to work.
    # We want D[i, j] for all i, j.
    
    # S1[i, j] = sum_k a[i,k] (b[p[j], p[k]] - b[p[i], p[k]])
    # S2[i, j] = sum_k a[k,i] (b[p[k], p[j]] - b[p[k], p[i]])
    # S3[i, j] = sum_k a[j,k] (b[p[i], p[k]] - b[p[j], p[k]])  (This is S1[j, i] with signs?)
    # S4[i, j] = sum_k a[k,j] (b[p[k], p[i]] - b[p[k], p[j]])  (This is S2[j, i] with signs?)
    
    # Let R[i, k] = b[p[i], p[k]]. This is just B.
    # Let Cols[j, k] = b[p[k], p[j]]. This is B.T.
    
    # S1[i, j] = a[i, :] @ (B[j, :] - B[i, :])
    # S2[i, j] = a[:, i] @ (B[:, j] - B[:, i])  -- Wait, b[p[k], p[j]] is B[k, j]. So B[:, j].
    
    # Let's compute S1 for all i, j.
    # S1[i, j] = sum_k a[i,k] B[j,k] - sum_k a[i,k] B[i,k]
    # S1 = a @ B.T - C[:, None]  where C = a @ B.T? No.
    # sum_k a[i,k] B[j,k] = (a @ B.T)[i, j]
    # sum_k a[i,k] B[i,k] = (a * B).sum(axis=1) = C[i] (if C was defined as row sum of element-wise product)
    # Let's define E = a * B. C_row[i] = E[i, :].sum().
    # Then S1[i, j] = (a @ B.T)[i, j] - C_row[i]
    
    # S2[i, j] = sum_k a[k,i] (B[k, j] - B[k, i])
    # sum_k a[k,i] B[k, j] = (a.T @ B)[i, j]
    # sum_k a[k,i] B[k, i] = (a.T * B).sum(axis=1) = C_col[i]
    # Then S2[i, j] = (a.T @ B)[i, j] - C_col[i]
    
    # Total Delta[i, j] = S1[i, j] + S2[i, j] + S1[j, i] + S2[j, i] + Correction for double counting?
    
    # Let's verify the formula from the current program's comments.
    # The current program computes:
    # raw = S1 + S2 + S3 + S4
    # S1 = np.dot(a[i, :], b[pj, p] - b[pi, p])  -> a[i,:] @ (B[j,:] - B[i,:])
    # S2 = np.dot(a[:, i], b[p, pj] - b[p, pi])  -> a[:,i] @ (B[:,j] - B[:,i])
    # S3 = np.dot(a[j, :], b[pi, p] - b[pj, p])  -> a[j,:] @ (B[i,:] - B[j,:]) = -S1[j, i] part?
    #    Actually S1[j, i] = a[j,:] @ (B[i,:] - B[j,:]). So S3 = S1[j, i].
    # S4 = np.dot(a[:, j], b[p, pi] - b[p, pj])  -> a[:,j] @ (B[:,i] - B[:,j]) = S2[j, i].
    
    # So raw = S1[i, j] + S2[i, j] + S1[j, i] + S2[j, i].
    
    # The current program then subtracts "bad_terms" and adds "direct_terms".
    # bad_terms and direct_terms handle the (i,i), (i,j), (j,i), (j,j) cases specifically.
    
    # Let's vectorize the bad/direct terms.
    # bad_terms = 
    #   a[i,i] * (b[pj, pi] - b[pi, pi] + b[pi, pj] - b[pi, pi])
    #   + a[i,j] * (b[pj, pj] - b[pi, pj] + b[pi, pi] - b[pi, pj])
    #   + a[j,i] * (b[pj, pj] - b[pj, pi] + b[pi, pi] - b[pj, pi])
    #   + a[j,j] * (b[pi, pj] - b[pj, pj] + b[pj, pi] - b[pj, pj])
    
    # Let Bpi_pj = b[pi, pj] = B[i, j]
    # Let Bj_pj = b[pj, pj] = B[j, j]
    # Let Bpi_pi = b[pi, pi] = B[i, i]
    # Let Bpj_pi = b[pj, pi] = B[j, i]
    
    # Term 1: a[i,i] * (B[j, i] - B[i, i] + B[i, j] - B[i, i]) = a[i,i] * (B[j, i] + B[i, j] - 2*B[i, i])
    # Term 2: a[i,j] * (B[j, j] - B[i, j] + B[i, i] - B[i, j]) = a[i,j] * (B[j, j] + B[i, i] - 2*B[i, j])
    # Term 3: a[j,i] * (B[j, j] - B[j, i] + B[i, i] - B[j, i]) = a[j,i] * (B[j, j] + B[i, i] - 2*B[j, i])
    # Term 4: a[j,j] * (B[i, j] - B[j, j] + B[j, i] - B[j, j]) = a[j,j] * (B[i, j] + B[j, i] - 2*B[j, j])
    
    # direct_terms =
    #   a[i,i] * (B[j, j] - B[i, i])
    #   + a[i,j] * (B[j, i] - B[i, j])
    #   + a[j,i] * (B[i, j] - B[j, i])
    #   + a[j,j] * (B[i, i] - B[j, j])
    
    # This is getting complicated to vectorize fully for all i,j at once without large temporary arrays.
    # However, for n=100, n^2 = 10,000 pairs. We can loop over i, and vectorize over j.
    # This reduces the Python loop overhead significantly compared to O(n^2) Python loops.
    
    Delta = np.zeros((n, n), dtype=np.int64)
    
    # Precompute parts
    ABT = a @ B.T
    ATB = a.T @ B
    C_row = (a * B).sum(axis=1)
    C_col = (a.T * B).sum(axis=1)
    
    S1_full = ABT - C_row[:, None]
    S2_full = ATB - C_col[:, None]
    
    # raw[i, j] = S1[i, j] + S2[i, j] + S1[j, i] + S2[j, i]
    # Note: S1 is not symmetric. S1[i, j] != S1[j, i].
    # But we can just compute the matrix.
    
    # Correction terms
    # We need to compute bad and direct for all i, j.
    # Let's iterate i and vectorize j.
    
    for i in range(n):
        # We want Delta[i, j] for all j.
        # pi = p[i]
        
        # S1[i, :] = ABT[i, :] - C_row[i]
        # S2[i, :] = ATB[i, :] - C_col[i]
        
        # S1[:, i] = ABT[:, i] - C_row
        # S2[:, i] = ATB[:, i] - C_col
        
        # raw[i, j] = S1[i, j] + S2[i, j] + S1[j, i] + S2[j, i]
        # For fixed i, j varies.
        # S1[i, j] is row i of S1_full.
        # S2[i, j] is row i of S2_full.
        # S1[j, i] is col i of S1_full.
        # S2[j, i] is col i of S2_full.
        
        raw_row = S1_full[i, :] + S2_full[i, :] + S1_full[:, i] + S2_full[:, i]
        
        # Now corrections for fixed i.
        # pi = p[i]
        # We need b[pi, p[j]] which is B[i, j].
        # We need b[p[j], pi] which is B[j, i].
        # We need b[pj, pj] which is B[j, j].
        # We need b[pi, pi] which is B[i, i].
        
        # Term 1: a[i,i] * (B[j, i] + B[i, j] - 2*B[i, i])
        # Term 2: a[i,j] * (B[j, j] + B[i, i] - 2*B[i, j])
        # Term 3: a[j,i] * (B[j, j] + B[i, i] - 2*B[j, i])
        # Term 4: a[j,j] * (B[i, j] + B[j, i] - 2*B[j, j])
        
        # Direct:
        # D1: a[i,i] * (B[j, j] - B[i, i])
        # D2: a[i,j] * (B[j, i] - B[i, j])
        # D3: a[j,i] * (B[i, j] - B[j, i])
        # D4: a[j,j] * (B[i, i] - B[j, j])
        
        # Let's compute these vectors for j=0..n-1.
        
        B_ii = B[i, i]
        B_ij = B[i, :]  # b[pi, p[j]]
        B_ji = B[:, i]  # b[p[j], pi]
        B_jj = np.diag(B) # b[p[j], p[j]]
        
        a_ii = a[i, i]
        a_ij = a[i, :]
        a_ji = a[:, i]
        a_jj = np.diag(a)
        
        bad = (
            a_ii * (B_ji + B_ij - 2*B_ii) +
            a_ij * (B_jj + B_ii - 2*B_ij) +
            a_ji * (B_jj + B_ii - 2*B_ji) +
            a_jj * (B_ij + B_ji - 2*B_jj)
        )
        
        direct = (
            a_ii * (B_jj - B_ii) +
            a_ij * (B_ji - B_ij) +
            a_ji * (B_ij - B_ji) +
            a_jj * (B_ii - B_jj)
        )
        
        Delta[i, :] = raw_row - bad + direct
        
    # Zero out diagonal? Swapping i with i is delta 0.
    np.fill_diagonal(Delta, 0)
    
    return Delta

def _local_search(a, b, p, deadline):
    """
    Best-improvement local search using vectorized delta matrix.
    """
    n = len(p)
    cur_cost = _qap_cost(a, b, p)
    best_cost = cur_cost
    best_p = p.copy()
    
    while time.time() < deadline:
        D = _compute_swap_delta_matrix(a, b, p)
        
        # Find best swap
        min_val, min_idx = np.unravel_index(np.argmin(D), D.shape)
        
        if min_val >= 0:
            break # No improving move
        
        i, j = min_idx
        # Apply swap
        p[i], p[j] = p[j], p[i]
        cur_cost += min_val
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0
    
    rng = np.random.default_rng(42)
    
    # 1. Initial Solution: Random
    p = rng.permutation(n)
    
    # 2. Local Search
    best_p, best_cost = _local_search(a, b, p, deadline - 5.0)
    
    # 3. Multi-start / Perturbation
    # While we have time, perturb the best solution and local search again.
    while time.time() < deadline - 5.0:
        p = best_p.copy()
        # Perturb with a few random swaps
        k = rng.integers(3, 10)
        for _ in range(k):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Local Search
        new_p, new_cost = _local_search(a, b, p, deadline - 5.0)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
