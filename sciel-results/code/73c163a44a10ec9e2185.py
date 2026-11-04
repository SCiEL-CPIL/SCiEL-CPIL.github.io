"""
Simulated Annealing solver for QAPLIB wil100.
Uses Benders-inspired decomposition: a linearized surrogate guides initial
assignment, while exact quadratic cost feedback tightens the search.
Multi-start SA with adaptive cooling and final local polish.
Improved: vectorized local search, batched SA moves, better time allocation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T1 = np.dot(diff_a_col, diff_b_row)
    corr_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    corr_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_T1 - corr_i - corr_j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T2 = np.dot(diff_a_row, diff_b_col)
    corr2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    corr2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_T2 - corr2_i - corr2_j
    
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(T1 + T2 + T3)

def _vectorized_local_search(a, b, p, current_cost, max_iters=100, time_limit=10.0):
    """
    Fully vectorized steepest-descent local search.
    Computes all O(n^2) swap deltas simultaneously using matrix operations.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    # Precompute constant parts that don't change with p
    # For each pair (i,j), delta = T1 + T2 + T3
    # We'll compute all deltas in a batch using broadcasting
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all swap deltas vectorized
        # pi = p[i], pj = p[j]
        # T1 involves: (a[:,i] - a[:,j]) dot (b[p,pj] - b[p,pi])
        # T2 involves: (a[i,:] - a[j,:]) dot (b[pj,p] - b[pi,p])
        # T3 involves: (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
        
        # Build index arrays
        idx_i = np.arange(n)[:, None]  # (n, 1)
        idx_j = np.arange(n)[None, :]  # (1, n)
        
        # pi[i] = p[i], pj[j] = p[j]
        pi = p[idx_i]  # (n, 1)
        pj = p[idx_j]  # (1, n)
        
        # a[:, i] - a[:, j]: shape (n, n, n) -> too large
        # Instead, compute per-pair more efficiently
        
        # For T1: sum_k (a[k,i] - a[k,j]) * (b[p_k, pj] - b[p_k, pi])
        # = sum_k a[k,i]*b[p_k,pj] - sum_k a[k,i]*b[p_k,pi] - sum_k a[k,j]*b[p_k,pj] + sum_k a[k,j]*b[p_k,pi]
        
        # Let M1[i,j] = sum_k a[k,i] * b[p_k, p_j]
        # Let M2[i,j] = sum_k a[k,i] * b[p_k, p_i]
        # Let M3[i,j] = sum_k a[k,j] * b[p_k, p_j]
        # Let M4[i,j] = sum_k a[k,j] * b[p_k, p_i]
        
        # M1: a.T @ b[p, p_j] for each j -> a.T (n,n) @ b[p, p_j] (n,n) 
        # Actually: M1[i,j] = sum_k a[k,i] * b[p[k], p[j]]
        # = (a.T @ b[p, :])[i, p[j]]
        # So M1 = a.T @ b[p, :] then index with [idx_i, p[idx_j]]
        
        Ab = a.T @ b[p, :]  # (n, n): Ab[i, loc] = sum_k a[k,i]*b[loc]
        # M1[i,j] = Ab[i, p[j]]
        M1 = Ab[np.arange(n)[:, None], p]  # (n, n)
        
        # M2[i,j] = sum_k a[k,i] * b[p[k], p[i]]
        # = (a.T @ b[p, :])[i, p[i]] for all i (independent of j)
        M2_vec = Ab[np.arange(n), p]  # (n,)
        M2 = M2_vec[:, None] * np.ones((1, n))  # (n, n) broadcast
        
        # M3[i,j] = sum_k a[k,j] * b[p[k], p[j]]
        # = Ab[j, p[j]] for all j (independent of i)
        M3_vec = Ab[np.arange(n), p]  # (n,)
        M3 = np.ones((n, 1)) * M3_vec[None, :]  # (n, n) broadcast
        
        # M4[i,j] = sum_k a[k,j] * b[p[k], p[i]]
        # = Ab[j, p[i]]
        M4 = Ab[np.arange(n)[None, :], p[:, None]]  # (n, n): M4[i,j] = Ab[j, p[i]]
        
        T1 = M1 - M2 - M3 + M4
        
        # Now subtract corrections for i and j terms
        # corr_i = (a[i,i] - a[i,j]) * (b[pi,pj] - b[pi,pi])
        # corr_j = (a[j,i] - a[j,j]) * (b[pj,pj] - b[pj,pi])
        
        # a[i,i] - a[i,j]: A_diag[i] - A[i,j] where A_diag[i] = a[i,i]
        A_diag = np.diag(a)
        A_ij = a[np.arange(n)[:, None], np.arange(n)[None, :]]  # (n,n)
        A_ji = A_ij.T
        
        b_pi_pj = b[p[:, None], p[None, :]]  # (n,n): b[p[i], p[j]]
        b_pi_pi = b[p, p]  # (n,)
        b_pj_pj = b[p, p]  # (n,)
        b_pj_pi = b[p[None, :], p[:, None]]  # (n,n): b[p[j], p[i]]
        
        corr_i = (A_diag[:, None] - A_ij) * (b_pi_pj - b_pi_pi[:, None])
        corr_j = (A_ji - A_diag[None, :]) * (b_pj_pj[None, :] - b_pj_pi)
        
        T1_corrected = T1 - corr_i - corr_j
        
        # T2: sum_k (a[i,k] - a[j,k]) * (b[pj, p_k] - b[pi, p_k])
        # = sum_k a[i,k]*b[pj,p_k] - sum_k a[i,k]*b[pi,p_k] - sum_k a[j,k]*b[pj,p_k] + sum_k a[j,k]*b[pi,p_k]
        
        # Let N1[i,j] = sum_k a[i,k] * b[p_j, p_k]
        # Let N2[i,j] = sum_k a[i,k] * b[p_i, p_k]
        # Let N3[i,j] = sum_k a[j,k] * b[p_j, p_k]
        # Let N4[i,j] = sum_k a[j,k] * b[p_i, p_k]
        
        # N1[i,j] = sum_k a[i,k] * b[p_j, p_k]
        # = (a @ b[p_j, :].T)[i, p_j] ... hmm
        # Actually: sum_k a[i,k] * b[p_j, p_k] = a[i, :] dot b[p_j, p]
        # Let Bp = b[p, :]  # (n, n): Bp[loc, k] = b[loc, p_k]
        # Then sum_k a[i,k] * b[p_j, p_k] = a[i, :] dot Bp[p_j, :]
        
        Bp = b[p, :]  # (n, n)
        # N1[i,j] = a[i, :] dot Bp[p_j, :] = (a @ Bp.T)[i, p_j]
        aBpT = a @ Bp.T  # (n, n)
        N1 = aBpT[np.arange(n)[:, None], p]  # (n, n)
        
        # N2[i,j] = sum_k a[i,k] * b[p_i, p_k] = a[i, :] dot Bp[p_i, :] = aBpT[i, p_i]
        N2_vec = aBpT[np.arange(n), p]  # (n,)
        N2 = N2_vec[:, None] * np.ones((1, n))
        
        # N3[i,j] = sum_k a[j,k] * b[p_j, p_k] = aBpT[j, p_j]
        N3_vec = aBpT[np.arange(n), p]  # (n,)
        N3 = np.ones((n, 1)) * N3_vec[None, :]
        
        # N4[i,j] = sum_k a[j,k] * b[p_i, p_k] = aBpT[j, p_i]
        N4 = aBpT[np.arange(n)[None, :], p[:, None]]  # (n, n)
        
        T2_raw = N1 - N2 - N3 + N4
        
        # Corrections for T2:
        # corr2_i = (a[i,i] - a[j,i]) * (b[pj,pi] - b[pi,pi])
        # corr2_j = (a[i,j] - a[j,j]) * (b[pj,pj] - b[pi,pj])
        
        corr2_i = (A_diag[:, None] - A_ji) * (b_pj_pi - b_pi_pi[:, None])
        corr2_j = (A_ij - A_diag[None, :]) * (b_pj_pj[None, :] - b_pi_pj)
        
        T2_corrected = T2_raw - corr2_i - corr2_j
        
        # T3: (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
        T3 = (A_ij - A_ji) * (b_pj_pi - b_pi_pj)
        
        # Total delta
        deltas = T1_corrected + T2_corrected + T2_corrected + T3
        
        # Set diagonal to 0 (no self-swaps)
        np.fill_diagonal(deltas, 0)
        
        # Find the best improving swap
        best_delta = np.min(deltas)
        
        if best_delta >= 0:
            break  # Local minimum
        
        # Find position of best delta
        best_pos = np.unravel_index(np.argmin(deltas), deltas.shape)
        best_i, best_j = int(best_pos[0]), int(best_pos[1])
        
        if best_i == best_j:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)
    
    return p, cost

def _sa_run_vectorized(a, b, p_init, current_cost, duration, T_start, T_end, batch_size=10):
    """
    Simulated annealing with batched move evaluation for reduced Python overhead.
    Returns (best_p, best_cost).
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        # Process batch_size swaps per temperature step
        for _ in range(batch_size):
            if time.time() - start_time >= duration:
                break
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = cost + delta
            
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / T)
                accept = np.random.rand() < accept_prob
            
            if accept:
                p[i], p[j] = p[j], p[i]
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
    
    return best_p, best_cost

def _tabu_search_vectorized(a, b, p, current_cost, time_limit=5.0, tabu_tenure=15):
    """
    Tabu search with vectorized delta computation.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Tabu list: set of (min(i,j), max(i,j)) pairs
    tabu_set = set()
    iteration = 0
    
    while time.time() - start_time < time_limit:
        # Remove expired tabu entries (we track by iteration)
        # For simplicity, use a list with expiry
        if iteration > 0:
            # This is a simplified approach; in practice we'd use a proper data structure
            pass
        
        # Compute all deltas vectorized (same as in _vectorized_local_search)
        idx_i = np.arange(n)[:, None]
        idx_j = np.arange(n)[None, :]
        
        pi = p[idx_i]
        pj = p[idx_j]
        
        Ab = a.T @ b[p, :]
        M1 = Ab[np.arange(n)[:, None], p]
        M2_vec = Ab[np.arange(n), p]
        M2 = M2_vec[:, None] * np.ones((1, n))
        M3_vec = Ab[np.arange(n), p]
        M3 = np.ones((n, 1)) * M3_vec[None, :]
        M4 = Ab[np.arange(n)[None, :], p[:, None]]
        T1 = M1 - M2 - M3 + M4
        
        A_diag = np.diag(a)
        A_ij = a[np.arange(n)[:, None], np.arange(n)[None, :]]
        A_ji = A_ij.T
        b_pi_pj = b[p[:, None], p[None, :]]
        b_pi_pi = b[p, p]
        b_pj_pj = b[p, p]
        b_pj_pi = b[p[None, :], p[:, None]]
        
        corr_i = (A_diag[:, None] - A_ij) * (b_pi_pj - b_pi_pi[:, None])
        corr_j = (A_ji - A_diag[None, :]) * (b_pj_pj[None, :] - b_pj_pi)
        T1_corrected = T1 - corr_i - corr_j
        
        Bp = b[p, :]
        aBpT = a @ Bp.T
        N1 = aBpT[np.arange(n)[:, None], p]
        N2_vec = aBpT[np.arange(n), p]
        N2 = N2_vec[:, None] * np.ones((1, n))
        N3_vec = aBpT[np.arange(n), p]
        N3 = np.ones((n, 1)) * N3_vec[None, :]
        N4 = aBpT[np.arange(n)[None, :], p[:, None]]
        T2_raw = N1 - N2 - N3 + N4
        
        corr2_i = (A_diag[:, None] - A_ji) * (b_pj_pi - b_pi_pi[:, None])
        corr2_j = (A_ij - A_diag[None, :]) * (b_pj_pj[None, :] - b_pi_pj)
        T2_corrected = T2_raw - corr2_i - corr2_j
        
        T3 = (A_ij - A_ji) * (b_pj_pi - b_pi_pj)
        
        deltas = T1_corrected + T2_corrected + T2_corrected + T3
        np.fill_diagonal(deltas, 0)
        
        # Mask tabu moves
        for ti, tj in tabu_set:
            deltas[ti, tj] = 0
            deltas[tj, ti] = 0
        
        best_delta = np.min(deltas)
        
        if best_delta >= 0:
            break
        
        best_pos = np.unravel_index(np.argmin(deltas), deltas.shape)
        best_i, best_j = int(best_pos[0]), int(best_pos[1])
        
        if best_i == best_j:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)
        
        # Add to tabu
        key = (min(best_i, best_j), max(best_i, best_j))
        tabu_set.add(key)
        iteration += 1
        
        if len(tabu_set) > tabu_tenure * 2:
            # Remove oldest entries (simplified: just clear and rebuild)
            # For now, just keep the last tabu_tenure entries
            if len(tabu_set) > tabu_tenure:
                # Remove arbitrary entries to maintain size
                items = list(tabu_set)
                for item in items[:len(items) - tabu_tenure]:
                    tabu_set.discard(item)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_greedy, cost_greedy), (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: SA runs (vectorized with batching)
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0  # Reserve 20s for final polish
    
    # Run 1: From best initial
    seg1_time = remaining * 0.40
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run_vectorized(a, b, best_p, best_cost, seg1_time, T_start, T_end, batch_size=20)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Run 2: From second-best initial
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    seg2_time = remaining * 0.35
    T_start = 300.0
    T_end = 3.0
    p2, cost2 = _sa_run_vectorized(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end, batch_size=15)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()
    
    # Run 3: From third-best initial
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    seg3_time = remaining * 0.30
    T_start = 200.0
    T_end = 2.0
    p3, cost3 = _sa_run_vectorized(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end, batch_size=10)
    if cost3 < best_cost:
        best_cost = cost3
        best_p = p3.copy()
    
    # Phase 3: Final polish - vectorized local search then tabu search
    elapsed = time.time() - start_time
    polish_budget = max(0.0, total_deadline - elapsed - 3.0)
    
    if polish_budget > 1.0:
        # First: vectorized steepest-descent local search
        ls_time = polish_budget * 0.5
        best_p, best_cost = _vectorized_local_search(a, b, best_p, best_cost, max_iters=200, time_limit=ls_time)
        
        # Then: vectorized tabu search to escape any remaining local minimum
        elapsed = time.time() - start_time
        tabu_time = max(0.0, total_deadline - elapsed - 3.0)
        if tabu_time > 1.0:
            best_p, best_cost = _tabu_search_vectorized(a, b, best_p, best_cost, time_limit=tabu_time, tabu_tenure=15)
    
    # Final safety check
    if time.time() - start_time > total_deadline - 2.0:
        pass  # Just return what we have
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
