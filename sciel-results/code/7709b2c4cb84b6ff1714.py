"""
Tabu Search QAP solver for wil100 using vectorized neighborhood evaluation.
Strategy:
1. Spectral initialization (eigenvector alignment)
2. Best-improvement tabu search with vectorized swap delta computation
3. Intensification (local search) on best solution at end
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _spectral_init(a, b):
    """
    Initialize via spectral method: align principal eigenvectors of A and B.
    Then refine with Hungarian on a surrogate cost matrix.
    """
    n = a.shape[0]
    
    # Symmetrize for eigen decomposition
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get principal eigenvectors
    eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
    eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
    
    # Use the largest eigenvector
    # For minimization, we want to align with negative eigenvalues or use the
    # structure of the problem. Let's use the eigenvector corresponding to
    # the smallest eigenvalue (most negative) for both, which tends to put
    # high-flow facilities far apart in high-distance locations.
    
    # Actually for QAP minimization, we want high flow -> low distance.
    # The spectral approach: assign facility i to location j to minimize
    # trace(A * P^T * B * P) = sum a[i,j] b[p[i],p[j]]
    # This is related to maximizing alignment with certain eigenvectors.
    
    # Let's try: sort by projection onto principal eigenvectors
    # High flow centrality should map to low distance centrality
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Build surrogate cost matrix:
    # We want to assign facilities with high total flow to locations with low total distance
    # cost[i,j] = a_row_sums[i] * b_row_sums[j]  (minimize this)
    C = np.outer(a_row_sums, b_row_sums)
    
    # Also add spectral component
    # Project onto first eigenvector
    v_a = eigvecs_a[:, -1]  # largest eigenvalue eigenvector
    v_b = eigvecs_b[:, -1]
    
    # Normalize
    v_a = v_a / (np.linalg.norm(v_a) + 1e-10)
    v_b = v_b / (np.linalg.norm(v_b) + 1e-10)
    
    # Add spectral term: we want to match similar "positions" in the embedding
    spectral_term = -np.outer(v_a, v_b)  # negative to encourage alignment
    
    # Combine
    # Normalize both terms
    C_max = np.max(np.abs(C))
    S_max = np.max(np.abs(spectral_term))
    if C_max > 0:
        C = C / C_max
    if S_max > 0:
        spectral_term = spectral_term / S_max
    
    C_total = C + 0.5 * spectral_term
    
    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(C_total)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _vectorized_all_swaps(a, b, p):
    """
    Compute delta for all (i,j) swaps where i < j.
    Returns array of shape (n, n) with delta[i,j] for swapping positions i and j.
    Uses the formula:
    delta(i,j) = (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
               + a[i,i]*(b[p[j],p[j]]-b[p[i],p[i]]) + a[j,j]*(b[p[i],p[i]]-b[p[j],p[j]])
               + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
               + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    
    For vectorization, we compute:
    T1[i,j] = sum_k (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    T2[i,j] = sum_k (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    
    Then subtract the k=i and k=j terms.
    
    This is O(n^3) but fully vectorized with matrix operations.
    For n=100, this is 10^6 * some constant, which is fast in NumPy.
    """
    n = len(p)
    
    # Precompute
    diff_a = a - a.T  # diff_a[i,j] = a[i,j] - a[j,i]
    
    # b_p = b indexed by p: b_p[i,j] = b[p[i],p[j]]
    b_p = b[np.ix_(p, p)]
    
    # T1[i,j] = sum_k (a[i,k]-a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    # Let's define:
    # row_diff[i,k] = a[i,k] - a[j,k]  -- but this depends on j, so we need outer structure
    # 
    # Alternative approach:
    # T1[i,j] = sum_k a[i,k]*b_p[j,k] - sum_k a[i,k]*b_p[i,k] 
    #          - sum_k a[j,k]*b_p[j,k] + sum_k a[j,k]*b_p[i,k]
    # = (a @ b_p.T)[i,j] - (a * b_p).sum(axis=1)[i] ... no this doesn't work directly
    #
    # Let me think differently:
    # T1[i,j] = sum_k (a[i,k] - a[j,k]) * (b_p[j,k] - b_p[i,k])
    # = sum_k a[i,k]*b_p[j,k] - sum_k a[i,k]*b_p[i,k]
    #   - sum_k a[j,k]*b_p[j,k] + sum_k a[j,k]*b_p[i,k]
    
    # Compute matrix products
    # M1 = a @ b_p.T  => M1[i,j] = sum_k a[i,k]*b_p.T[k,j] = sum_k a[i,k]*b_p[j,k]
    M1 = a @ b_p.T
    # M2[i,j] = sum_k a[i,k]*b_p[i,k]  -- this is (a * b_p) summed over k for fixed i, but depends on j via b_p[i,k]
    # Actually M2 doesn't depend on j! It's just (a * b_p).sum(axis=1)
    M2 = (a * b_p).sum(axis=1)  # M2[i] = sum_k a[i,k]*b_p[i,k]
    
    # Similarly:
    # M3[i,j] = sum_k a[j,k]*b_p[j,k]  -- this is M2[j], doesn't depend on i
    # M4[i,j] = sum_k a[j,k]*b_p[i,k] = M1.T[i,j] = (a @ b_p.T)[j,i]
    
    T1 = M1 - M2[:, None] - M2[None, :] + M1.T
    
    # T2[i,j] = sum_k (a[k,i]-a[k,j]) * (b_p[k,j] - b_p[k,i])
    # = sum_k a[k,i]*b_p[k,j] - sum_k a[k,i]*b_p[k,i]
    #   - sum_k a[k,j]*b_p[k,j] + sum_k a[k,j]*b_p[k,i]
    # 
    # N1[i,j] = sum_k a[k,i]*b_p[k,j] = (a.T @ b_p)[i,j]
    N1 = a.T @ b_p
    # N2[i] = sum_k a[k,i]*b_p[k,i] = (a.T * b_p).sum(axis=0) ... 
    # Actually (a.T)[k,i] * b_p[k,i], sum over k
    N2 = (a.T * b_p).sum(axis=0)  # N2[i] = sum_k a[k,i]*b_p[k,i]
    
    T2 = N1 - N2[:, None] - N2[None, :] + N1.T
    
    # Now subtract k=i and k=j contributions
    # For T1, the k=i term: (a[i,i]-a[j,i])*(b_p[j,i]-b_p[i,i])
    # The k=j term: (a[i,j]-a[j,j])*(b_p[j,j]-b_p[i,j])
    
    # b_p[i,j] = b[p[i],p[j]]
    # k=i term for T1[i,j]: (a[i,i]-a[j,i]) * (b_p[j,i] - b_p[i,i])
    # k=j term for T1[i,j]: (a[i,j]-a[j,j]) * (b_p[j,j] - b_p[i,j])
    
    k_i_term_T1 = (a - a.T) * (b_p.T - b_p[np.ix_(np.arange(n), np.arange(n))][:, None])
    # Wait, let me be more careful.
    # k_i_term_T1[i,j] = (a[i,i]-a[j,i]) * (b_p[j,i] - b_p[i,i])
    # b_p[i,i] = b[p[i],p[i]] -- this is a vector
    diag_b_p = b_p[np.arange(n), np.arange(n)]  # b_p[i,i] for all i
    
    k_i_T1 = (a - a.T) * (b_p.T - diag_b_p[None, :])
    # k_i_T1[i,j] = (a[i,j]-a[j,i]) * (b_p[j,i] - diag_b_p[j]) ... 
    # Hmm, b_p.T[i,j] = b_p[j,i] = b[p[j],p[i]]
    # diag_b_p[j] = b[p[j],p[j]]
    # So k_i_T1[i,j] = (a[i,j]-a[j,i])*(b[p[j],p[i]] - b[p[j],p[j]])
    # But we need (a[i,i]-a[j,i])*(b_p[j,i]-b_p[i,i])
    # These are different! Let me redo.
    
    # The full sum T1[i,j] includes k=i and k=j. We need to subtract them.
    # k=i contribution to T1[i,j]: (a[i,i]-a[j,i])*(b_p[j,i]-b_p[i,i])
    # k=j contribution to T1[i,j]: (a[i,j]-a[j,j])*(b_p[j,j]-b_p[i,j])
    
    # Let's compute these separately
    # k=i: for each pair (i,j), value = (a[i,i]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[i]])
    a_ii = a[np.arange(n), np.arange(n)]  # a[i,i]
    
    k_i_val = (a_ii[:, None] - a.T) * (b_p.T - diag_b_p[:, None])
    # k_i_val[i,j] = (a[i,i]-a[j,i])*(b_p[j,i]-diag_b_p[i])
    # b_p[j,i] = b[p[j],p[i]], diag_b_p[i] = b[p[i],p[i]]
    # So k_i_val[i,j] = (a[i,i]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[i]])  ✓
    
    # k=j: for each pair (i,j), value = (a[i,j]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[j]])
    k_j_val = (a - a_ii[None, :]) * (diag_b_p[:, None] - b_p)
    # k_j_val[i,j] = (a[i,j]-a[j,j])*(diag_b_p[j]-b_p[i,j])
    # = (a[i,j]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[j]])  ✓
    
    T1_corrected = T1 - k_i_val - k_j_val
    
    # Similarly for T2:
    # k=i contribution to T2[i,j]: (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])
    # k=j contribution to T2[i,j]: (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])
    
    k_i_val_T2 = (a_ii[:, None] - a) * (b_p - diag_b_p[:, None])
    # k_i_val_T2[i,j] = (a[i,i]-a[i,j])*(b_p[i,j]-diag_b_p[i])
    # = (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])  ✓
    
    k_j_val_T2 = (a.T - a_ii[None, :]) * (diag_b_p[:, None] - b_p.T)
    # k_j_val_T2[i,j] = (a[j,i]-a[j,j])*(diag_b_p[j]-b_p.T[i,j])
    # = (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])  ✓
    
    T2_corrected = T2 - k_i_val_T2 - k_j_val_T2
    
    # Direct pair term
    direct = (a - a.T) * (b_p.T - b_p)
    # direct[i,j] = (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])  ✓
    
    # Diagonal terms
    diag_term = a_ii[:, None] * (diag_b_p[None, :] - diag_b_p[:, None]) + \
                a_ii[None, :] * (diag_b_p[:, None] - diag_b_p[None, :])
    # diag_term[i,j] = a[i,i]*(b[p[j],p[j]]-b[p[i],p[i]]) + a[j,j]*(b[p[i],p[i]]-b[p[j],p[j]])
    # = (a[i,i]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[i]])  ✓
    
    delta_all = direct + diag_term + T1_corrected + T2_corrected
    
    return delta_all


def _tabu_search(a, b, p, time_limit):
    """
    Best-improvement tabu search with vectorized neighborhood.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Tabu list: store (i, j, time_added)
    tabu_len = 20
    tabu_list = []  # list of (i, j) that are tabu
    tabu_timestamps = {}  # (i,j) -> time when added
    
    # Diversification: track how many times each position has been in best
    diversity_counter = np.zeros(n, dtype=np.int32)
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        
        # Remove expired tabu entries
        now = elapsed
        expired = [(ij) for ij, t in tabu_timestamps.items() if now - t > tabu_len]
        for ij in expired:
            del tabu_timestamps[ij]
        
        # Get all swap deltas
        delta_all = _vectorized_all_swaps(a, b, p)
        
        # Set tabu moves to +infinity
        for ij in tabu_timestamps.keys():
            i, j = ij
            delta_all[i, j] = np.inf
            delta_all[j, i] = np.inf
        
        # Find best move (i, j) with i < j
        # Only consider upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        masked_delta = np.where(mask, delta_all, np.inf)
        
        flat_idx = np.argmin(masked_delta)
        i, j = np.unravel_index(flat_idx, (n, n))
        
        best_delta = delta_all[i, j]
        
        if np.isinf(best_delta):
            # All moves are tabu, pick the least bad non-tabu or break
            break
        
        # Aspiration criterion: accept tabu move if it improves best
        # (already handled by setting to inf, but we could add aspiration)
        
        # Apply move
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
        # Add to tabu list
        tabu_timestamps[(i, j)] = now
        tabu_timestamps[(j, i)] = now
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
            diversity_counter = np.zeros(n, dtype=np.int32)
            for idx, pos in enumerate(p):
                diversity_counter[pos] += 1
        else:
            no_improve_count += 1
        
        # If too many non-improving steps, intensify
        if no_improve_count > 50:
            no_improve_count = 0
            # Do a few local improvements
            for _ in range(5):
                if time.time() - start_time > time_limit - 2:
                    break
                delta_all2 = _vectorized_all_swaps(a, b, p)
                mask2 = np.triu(np.ones((n, n), dtype=bool), k=1)
                masked2 = np.where(mask2, delta_all2, np.inf)
                flat2 = np.argmin(masked2)
                i2, j2 = np.unravel_index(flat2, (n, n))
                if delta_all2[i2, j2] < 0:
                    p[i2], p[j2] = p[j2], p[i2]
                    current_cost += int(delta_all2[i2, j2])
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    break
    
    return best_p, best_cost


def _local_search(a, b, p, time_limit):
    """Best-improvement local search until no improving swap exists."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        delta_all = _vectorized_all_swaps(a, b, p)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        masked = np.where(mask, delta_all, np.inf)
        
        flat_idx = np.argmin(masked)
        i, j = np.unravel_index(flat_idx, (n, n))
        
        best_delta = delta_all[i, j]
        
        if best_delta >= 0 or np.isinf(best_delta):
            break
        
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = 145.0  # Safety margin before 150s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Step 1: Spectral initialization
    p_init = _spectral_init(a, b)
    cost_init = _qap_cost(a, b, p_init)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Step 2: Quick local search to polish initial
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        p_ls, cost_ls = _local_search(a, b, p_init.copy(), min(5.0, time_remaining * 0.1))
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Step 3: Tabu search (main optimization)
    time_remaining = deadline - (time.time() - start)
    tabu_time = max(0, time_remaining - 15)
    if tabu_time > 5:
        p_tabu, cost_tabu = _tabu_search(a, b, best_p.copy(), tabu_time)
        if cost_tabu < best_cost:
            best_cost = cost_tabu
            best_p = p_tabu.copy()
    
    # Step 4: Final local search polish
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        p_final, cost_final = _local_search(a, b, best_p.copy(), min(10.0, time_remaining - 3))
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    return best_p.tolist()
