"""Vectorized Simulated Annealing for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a_sym, b, p):
    """
    Compute the cost change for all possible swaps (i, j) in a single vectorized operation.
    
    Parameters:
    a_sym: (n, n) symmetric flow matrix (a + a.T)
    b: (n, n) distance matrix
    p: (n,) current permutation
    
    Returns:
    delta: (n, n) matrix where delta[i, j] is the cost change of swapping positions i and j.
    """
    n = len(p)
    # D_a[i, j] = a_sym[i, :] - a_sym[j, :]
    # D_b[i, j] = b[p[i], :] - b[p[j], :]
    
    # Efficient construction using broadcasting
    # a_sym[i, :] is row i, a_sym[j, :] is row j
    # We want a matrix M where M[i, j] = a_sym[i, :] - a_sym[j, :]
    # This is (a_sym.T - a_sym) ? No.
    # a_sym[i, :] is shape (n,). 
    # Let A = a_sym.
    # We want D[i, j, k] = A[i, k] - A[j, k].
    # This can be viewed as outer difference.
    
    # D_b[i, j, k] = b[p[i], k] - b[p[j], k]
    # Let B_p = b[p, :]. Shape (n, n).
    # D_b[i, j, k] = B_p[i, k] - B_p[j, k]
    
    # Delta[i, j] = sum_k (A[i, k] - A[j, k]) * (B_p[i, k] - B_p[j, k])
    #             = sum_k A[i, k]B_p[i, k] - sum_k A[i, k]B_p[j, k] - sum_k A[j, k]B_p[i, k] + sum_k A[j, k]B_p[j, k]
    
    # Let S = A * B_p (element-wise product). Shape (n, n).
    # S[i, k] = A[i, k] * B_p[i, k]
    # Term 1: sum_k S[i, k] -> S.sum(axis=1) -> shape (n,)
    # Term 4: sum_k S[j, k] -> S.sum(axis=1) -> shape (n,)
    # Term 2: sum_k A[i, k] B_p[j, k] -> A @ B_p.T -> shape (n, n)
    # Term 3: sum_k A[j, k] B_p[i, k] -> (A @ B_p.T).T -> shape (n, n)
    
    S = a_sym * b[p, :]
    row_sums = S.sum(axis=1)
    cross = a_sym @ b[p, :].T
    
    # Delta[i, j] = row_sums[i] - cross[i, j] - cross[j, i] + row_sums[j]
    delta = row_sums[:, np.newaxis] - cross - cross.T + row_sums[np.newaxis, :]
    
    # Subtract diagonal correction terms for i==j or specific indices?
    # The formula derived:
    # delta(i, j) = (A_i - A_j) . (B_{p_i} - B_{p_j}) - (A_ii - A_jj)(B_{p_i p_i} - B_{p_j p_j}) - (A_ij - A_ji)(B_{p_j p_i} - B_{p_i p_j})
    # Wait, the standard swap delta formula is:
    # C = sum_{k,l} A[k,l] B[p[k], p[l]]
    # Swap i, j:
    # Change = (A[i,:] - A[j,:]) . (B[p[j],:] - B[p[i],:]) 
    #         - (A[i,i] - A[j,j]) * (B[p[j], p[j]] - B[p[i], p[i]])
    #         - (A[i,j] - A[j,i]) * (B[p[j], p[i]] - B[p[i], p[j]])
    
    # My vectorized formula:
    # sum_k (A[i,k] - A[j,k]) * (B[p[i],k] - B[p[j],k])
    # = sum_k A[i,k]B[p[i],k] - sum_k A[i,k]B[p[j],k] - sum_k A[j,k]B[p[i],k] + sum_k A[j,k]B[p[j],k]
    # This matches the first term.
    
    # The second and third terms in the standard formula account for the fact that B[p[i], p[i]] and B[p[j], p[j]] change, and B[p[i], p[j]] and B[p[j], p[i]] change.
    # In the first term expansion, we summed over all k.
    # When k=i: A[i,i]B[p[i],i] - A[i,i]B[p[j],i] - A[j,i]B[p[i],i] + A[j,i]B[p[j],i]
    # When k=j: A[i,j]B[p[i],j] - A[i,j]B[p[j],j] - A[j,j]B[p[i],j] + A[j,j]B[p[j],j]
    # The standard formula subtracts corrections because the "dot product" assumes B is fixed, but B[p[i], p[i]] becomes B[p[j], p[j]] etc.
    
    # Let's stick to the simpler, slightly slower but safer vectorized approach if the math is tricky, OR verify the math.
    # Actually, the formula:
    # delta = (A_i - A_j) @ (B_pi - B_pj) 
    # This is exactly what the matrix multiplication does.
    # But we must subtract the terms involving the diagonal and the off-diagonal swap of B.
    
    # Let's use the explicit correction terms in a vectorized way.
    # Term 2: (A[i,i] - A[j,j]) * (B[p[j], p[j]] - B[p[i], p[i]])
    # Let D_A = np.diag(a_sym)
    # Let D_B_p = np.diag(b[p, :]) # This is b[p[i], p[i]]
    # T2[i, j] = (D_A[i] - D_A[j]) * (D_B_p[j] - D_B_p[i])
    
    # Term 3: (A[i,j] - A[j,i]) * (B[p[j], p[i]] - B[p[i], p[j]])
    # Let Off_A = a_sym - np.diag(np.diag(a_sym)) # Actually a_sym is symmetric, so A[i,j]=A[j,i]. 
    # Wait, if A is symmetric, A[i,j] - A[j,i] = 0. 
    # The QAP flow matrix A is typically symmetric. 
    # If A is symmetric, Term 3 vanishes!
    # Let's check tai256c. QAPLIB instances usually have symmetric flow matrices.
    # If A is symmetric, the formula simplifies to:
    # delta = (A_i - A_j) . (B_pi - B_pj) - (A_ii - A_jj)(B_{p_j p_j} - B_{p_i p_i})
    
    # So:
    # delta = row_sums[:, np.newaxis] - cross - cross.T + row_sums[np.newaxis, :] 
    #         - (D_A[:, np.newaxis] - D_A[np.newaxis, :]) * (D_B_p[np.newaxis, :] - D_B_p[:, np.newaxis])
    
    # Note: cross.T is cross[j, i] which is sum_k A[j,k]B[p[i],k].
    # cross[i, j] is sum_k A[i,k]B[p[j],k].
    
    D_A = np.diag(a_sym)
    D_B_p = np.diag(b[p, :])
    
    t2 = (D_A[:, np.newaxis] - D_A[np.newaxis, :]) * (D_B_p[np.newaxis, :] - D_B_p[:, np.newaxis])
    
    delta = row_sums[:, np.newaxis] - cross - cross.T + row_sums[np.newaxis, :] - t2
    
    # Zero out diagonal (swap i, i is not a move)
    np.fill_diagonal(delta, 0)
    
    return delta

def _simulated_annealing(a, b, p_init, time_limit, start_time, seed=42):
    """
    Vectorized Simulated Annealing.
    """
    n = len(p)
    rng = random.Random(seed)
    a_sym = a + a.T
    
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Temperature schedule
    T0 = 1000.0
    T_end = 1.0
    # We don't know the number of steps in advance, so use time-based cooling or fixed steps with time check.
    # Let's aim for ~5000 steps.
    total_steps = 5000
    cooling_factor = (T_end / T0) ** (1.0 / total_steps)
    
    T = T0
    no_improve = 0
    
    for step in range(total_steps):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Compute all deltas
        delta_mat = _compute_delta_matrix(a_sym, b, p)
        
        # Find best move
        min_idx = np.unravel_index(np.argmin(delta_mat), delta_mat.shape)
        i, j = min_idx
        best_delta = delta_mat[i, j]
        
        # Acceptance probability
        if best_delta < 0:
            accept = True
        else:
            # Exp(-delta / T)
            # Clamp delta to avoid overflow/underflow issues if T is very small
            if T > 0:
                prob = np.exp(-best_delta / T)
            else:
                prob = 0
            accept = rng.random() < prob
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T *= cooling_factor
        
        # Diversification kick if stuck too long
        if no_improve > 500:
             # Randomly swap a few pairs
             for _ in range(5):
                 ri = rng.randint(0, n-1)
                 rj = rng.randint(0, n-1)
                 if ri != rj:
                     p[ri], p[rj] = p[rj], p[ri]
             current_cost = _qap_cost(a, b, p)
             no_improve = 0
             T = max(T, 50.0) # Reheat slightly
    
    return best_p, best_cost

def _greedy_init(a, b, seed=42):
    """Greedy rank-matching initialization."""
    n = len(a)
    rng = random.Random(seed)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.copy()

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    best_p = None
    best_cost = np.inf
    
    # Run multiple islands with different seeds
    # Vectorized SA is fast, so we can run more iterations or more islands.
    # Let's run 3 islands.
    K = 3
    time_per_island = time_limit / K
    
    for k in range(K):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 10:
            break
        
        # Different initializations
        if k == 0:
            p_init = _greedy_init(a, b, seed=42 + k)
        else:
            p_init = _greedy_init(a, b, seed=42 + k)
            rng = random.Random(999 + k)
            for _ in range(n // 4):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
        
        ts_time = min(remaining_time - 5, time_per_island)
        p_best, cost_best = _simulated_annealing(a, b, p_init.copy(), ts_time, start_time, seed=1000 + k)
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        best_p = _greedy_init(a, b, seed=0)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
