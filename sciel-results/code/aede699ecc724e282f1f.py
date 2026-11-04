"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _local_search_2opt(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T
    
    # Precompute diagonal of a
    a_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
                
            # Vectorized delta calculation
            # delta(i, j) = 2 * (a_sym[i] - a_sym[j]) @ (D[j] - D[i]) 
            #              - (a_sym[i, i] - a_sym[j, i]) * (D[j, i] - D[i, i])
            #              - (a_sym[i, j] - a_sym[j, j]) * (D[j, j] - D[i, j])
            #              + (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
            #              + (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            
            # Let's break it down:
            # term1 = 2 * (a_sym[i] - a_sym[js]) @ (D[js] - D[i])
            #        = 2 * (a_sym[i] @ D[js] - a_sym[i] @ D[i] - a_sym[js] @ D[js] + a_sym[js] @ D[i])
            
            # We need to compute these efficiently.
            # a_sym[i] @ D[js] -> This is a dot product of a vector and a matrix? No, D[js] is a matrix of shape (m, n).
            # Actually, the formula is a scalar for each j.
            # sum_k (a_sym[i,k] - a_sym[j,k]) * (D[j,k] - D[i,k])
            # = sum_k a_sym[i,k]D[j,k] - sum_k a_sym[i,k]D[i,k] - sum_k a_sym[j,k]D[j,k] + sum_k a_sym[j,k]D[i,k]
            
            # Let's compute:
            # A = a_sym @ D  (n x n)
            # A[i, j] = sum_k a_sym[i,k] D[j,k]
            # A[j, i] = sum_k a_sym[j,k] D[i,k]
            # A[j, j] = sum_k a_sym[j,k] D[j,k]
            # A[i, i] = sum_k a_sym[i,k] D[i,k]
            
            # So the first part is: 2 * (A[i, js] - A[i, i] - np.diag(A[js, js]) + A[js, i])
            # Wait, A[js, js] is not simply diag(A[js, :]).
            # A[i, j] is the (i,j) element of the matrix product a_sym @ D.
            
            A_mat = a_sym @ D
            
            # term1 for each j in js:
            # 2 * (A_mat[i, j] - A_mat[i, i] - A_mat[j, j] + A_mat[j, i])
            
            term1 = 2 * (A_mat[i, js] - A_mat[i, i] - A_mat[js, js] + A_mat[js, i])
            
            # term2: - (a_sym[i, i] - a_sym[j, i]) * (D[j, i] - D[i, i])
            # a_sym[i, i] is constant.
            # a_sym[js, i] is a vector.
            # D[js, i] is a vector.
            # D[i, i] is constant.
            term2 = - (a_sym[i, i] - a_sym[js, i]) * (D[js, i] - D[i, i])
            
            # term3: - (a_sym[i, j] - a_sym[j, j]) * (D[j, j] - D[i, j])
            # a_sym[i, js] is a vector.
            # a_sym[js, js] is a vector (diagonal elements of submatrix).
            # D[js, js] is a vector.
            # D[i, js] is a vector.
            term3 = - (a_sym[i, js] - a_sym[js, js]) * (D[js, js] - D[i, js])
            
            # term4: (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
            term4 = (a_diag[i] - a_diag[js]) * (D[js, js] - D[i, i])
            
            # term5: (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            # a[i, js] is a vector.
            # a[js, i] is a vector.
            # D[js, i] is a vector.
            # D[i, js] is a vector.
            term5 = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = term1 + term2 + term3 + term4 + term5
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost


def _eigen_init(a, b, n, rng):
    """
    Eigenvalue-based initialization.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get top eigenvectors
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Take top k eigenvectors
        k = min(8, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        # Create score vectors
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        # Sort facilities by score_a descending, locations by score_b ascending
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        # Create permutation
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        return p
    except:
        return rng.permutation(n)


def _hungarian_init(a, b):
    """
    Hungarian algorithm initialization using surrogate cost.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _hungarian_reopt(a, b, p):
    """
    Hungarian algorithm re-optimization.
    """
    n = len(p)
    B_p = b[p, :]
    M1 = a @ b[:, p].T
    M2 = a.T @ B_p
    C = M1 + M2
    
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def _perturb(p, k, rng):
    """Randomly swap k pairs."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def _block_reversal(p, rng):
    """
    Reverse a random sub-sequence of the permutation.
    This is a more disruptive move than simple swaps.
    """
    n = len(p)
    if n < 3:
        return p.copy()
    
    # Choose length of block to reverse
    # Smaller blocks are more frequent
    max_len = min(n, 20)
    length = rng.integers(2, max_len + 1)
    
    start = rng.integers(0, n - length + 1)
    end = start + length
    
    p = p.copy()
    p[start:end] = p[start:end][::-1]
    return p


def _mcmc_step(a, b, p, current_cost, temperature, rng):
    """
    Perform one MCMC step (block reversal).
    Returns (new_p, new_cost, accepted).
    """
    n = len(p)
    
    # Generate new permutation
    new_p = _block_reversal(p, rng)
    
    # Compute new cost
    # We could compute delta, but for simplicity and correctness with block reversal,
    # we compute full cost. With n=256, this is fast enough.
    new_cost = int(np.sum(a * b[np.ix_(new_p, new_p)]))
    
    # Metropolis-Hastings acceptance
    delta = new_cost - current_cost
    
    if delta < 0:
        return new_p, new_cost, True
    else:
        # Accept with probability exp(-delta / T)
        if temperature > 0:
            # Avoid overflow
            if delta / temperature > 500:
                return p, current_cost, False
            prob = np.exp(-delta / temperature)
            if rng.random() < prob:
                return new_p, new_cost, True
        return p, current_cost, False


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # Safety margin for 120s limit
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(123)
    
    # Phase 1: Initialization
    initial_solutions = []
    
    # 1. Hungarian (strong structural start)
    p_hung = _hungarian_init(a, b)
    initial_solutions.append(p_hung)
    
    # 2. Eigen-based
    p_eigen = _eigen_init(a, b, n, rng)
    initial_solutions.append(p_eigen)
    
    # 3. Random
    initial_solutions.append(rng.permutation(n))
    
    # Local search on each initial solution
    # Allocate time: 15s each for 3 inits = 45s
    time_per_init = 15
    
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        
        t_limit = min(time_per_init, remaining - 5)
        p_ls, cost_ls = _local_search_2opt(a, b, p_init, t_limit, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: MCMC / ILS Hybrid
    # Use MCMC to escape local optima, then local search to polish.
    # This combines the stochastic exploration of MCMC with the exploitation of LS.
    
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed - 10  # Leave 10s for final polish
    
    if remaining_time < 5:
        # Not enough time, skip to final
        pass
    else:
        # MCMC Parameters
        # Start with high temperature, cool down
        T_start = (best_cost - np.min(a) * np.min(b) * n) / 10.0 # Rough estimate
        if T_start <= 0:
            T_start = 1000.0
        T_end = 1.0
        
        # Number of MCMC steps to try before a local search
        # We want to do several cycles: MCMC -> LS
        # Let's say we have ~60s for this phase.
        # Each LS takes ~5-10s. So maybe 5-10 cycles.
        
        current_p = best_p.copy()
        current_cost = best_cost
        consecutive_no_improve = 0
        
        # Cooling schedule
        # We will use a fixed number of MCMC steps between LS calls
        mcmc_steps_per_ls = 50
        
        while time.time() - start_time < time_budget - 10:
            elapsed = time.time() - start_time
            remaining = time_budget - 10 - elapsed
            if remaining < 5:
                break
            
            # Determine temperature based on progress
            # Simple linear cooling
            # Or adaptive: if no improvement, keep T high
            
            # For now, let's just use a fixed moderate temperature for exploration
            # and rely on LS to exploit.
            # Actually, let's try a simple cooling schedule.
            # We don't know total steps, so let's just use a fixed T for now.
            # T = 100.0
            
            # Let's try to adapt T based on best_cost change
            # If best_cost hasn't changed, increase T slightly to explore more.
            # If it has changed, decrease T.
            
            # Simple heuristic:
            if consecutive_no_improve > 5:
                T = 500.0
            elif consecutive_no_improve > 2:
                T = 100.0
            else:
                T = 10.0
            
            # Run MCMC steps
            for _ in range(mcmc_steps_per_ls):
                if time.time() - start_time > time_budget - 10:
                    break
                current_p, current_cost, accepted = _mcmc_step(a, b, current_p, current_cost, T, rng)
            
            # Local Search on the MCMC result
            t_limit = min(10.0, remaining - 2)
            p_ls, cost_ls = _local_search_2opt(a, b, current_p, t_limit, start_time)
            
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
                consecutive_no_improve = 0
                # Update current state to the improved one
                current_p = p_ls.copy()
                current_cost = cost_ls
            else:
                # Even if not better than global best, keep the MCMC state for further exploration?
                # Or reset to best_p?
                # Standard ILS resets to best_p.
                # MCMC usually keeps the chain state.
                # Let's keep the chain state to maintain momentum, but if it's stuck, reset.
                if consecutive_no_improve > 10:
                    current_p = best_p.copy()
                    current_cost = best_cost
                consecutive_no_improve += 1
    
    # Phase 3: Final Polish
    # Use remaining time for a final local search on the best solution found.
    elapsed = time.time() - start_time
    remaining = time_budget - 3 - elapsed
    if remaining > 0:
        p_final, cost_final = _local_search_2opt(a, b, best_p, remaining, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
