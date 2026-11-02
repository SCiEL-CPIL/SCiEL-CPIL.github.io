import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _find_half_integer_solution(A, b, R, m):
    """
    Find a half-integer vector w of length R such that A @ w = b.
    A is (m x R), b is (m,)
    Returns w or None.
    """
    # First, check if the system is consistent
    # Use SVD to get rank, null space, and particular solution
    U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Determine numerical rank
    tol = 1e-10
    rank_A = np.sum(S > tol)
    
    # Particular solution (minimum norm)
    if rank_A == 0:
        # A is all zeros; only solvable if b is all zeros
        if np.max(np.abs(b)) < tol:
            return np.zeros(R)
        return None
    
    # Minimum norm least squares solution
    Vt_r = Vt[:rank_A, :]
    S_inv = np.zeros(rank_A)
    S_inv[:rank_A] = 1.0 / S[:rank_A]
    w_particular = Vt_r.T @ (S_inv * (U_svd[:, :rank_A].T @ b))
    
    # Check if particular solution satisfies A @ w = b
    residual = np.max(np.abs(A @ w_particular - b))
    if residual > 1e-8:
        # System may be inconsistent; try to find any solution
        # Use lstsq as fallback
        w_ls, residuals, rank_lstsq, sv = np.linalg.lstsq(A, b, rcond=None)
        residual = np.max(np.abs(A @ w_ls - b))
        if residual > 1e-8:
            return None
        w_particular = w_ls
        rank_A = rank_lstsq
    
    # Now w_particular is a particular solution. General solution: w = w_particular + N @ t
    # where N is the null space basis (R x (R - rank_A))
    
    if rank_A == R:
        # Unique solution; check if it's half-integer
        w_rounded = np.round(w_particular * 2) / 2
        res = np.max(np.abs(A @ w_rounded - b))
        if res < 1e-9:
            return w_rounded
        return None
    
    # Null space basis
    if rank_A < R:
        N = Vt[rank_A:, :].T  # (R x (R - rank_A))
    else:
        N = np.zeros((R, 0))
    
    # We need: A @ (w_particular + N @ t) = b
    # Since A @ w_particular = b and A @ N = 0, this is automatically satisfied.
    # So we just need w_particular + N @ t to be half-integer.
    
    # Try to find t such that w_particular + N @ t is half-integer
    # This is a system of R equations (one per component) in (R - rank_A) unknowns
    
    # Strategy: try random t values and check
    best_w = None
    best_res = np.inf
    
    # Try t = 0 first
    w_candidate = w_particular
    w_rounded = np.round(w_candidate * 2) / 2
    res = np.max(np.abs(w_rounded - w_candidate))
    if res < 1e-9:
        return w_rounded
    
    # Try small integer/half-integer t values
    null_dim = R - rank_A
    if null_dim == 0:
        return None
    
    # Try random t values
    for attempt in range(500):
        if null_dim <= 6:
            # Enumerate small values
            t = np.random.randint(-3, 4, size=(null_dim,)) * 0.5
        else:
            t = np.random.uniform(-2, 2, size=(null_dim,))
        
        w_candidate = w_particular + N @ t
        w_rounded = np.round(w_candidate * 2) / 2
        res = np.max(np.abs(w_rounded - w_candidate))
        
        if res < 1e-9:
            # Verify A @ w_rounded = b
            final_res = np.max(np.abs(A @ w_rounded - b))
            if final_res < 1e-9:
                return w_rounded
        
        if res < best_res:
            best_res = res
            best_w = w_rounded
    
    # If we found a close solution, verify it
    if best_w is not None:
        final_res = np.max(np.abs(A @ best_w - b))
        if final_res < 1e-9:
            return best_w
    
    return None

def _try_solve_for_W(U, V, n, m, p, R):
    """
    Given U (n*m x R) and V (m*p x R), try to find W (p*n x R) with half-integer entries
    such that for all i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    W_sol = np.zeros((p * n, R))
    
    for i in range(n):
        for k in range(p):
            # Build A matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve A @ w = ones(m)
            w_sol = _find_half_integer_solution(A, np.ones(m), R, m)
            
            if w_sol is None:
                return None
            
            W_sol[k * n + i, :] = w_sol
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :] * V[j * p + k, :], W_sol[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return W_sol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    np.random.seed(42)
    
    # Entry sets to try - include more diverse options
    entry_sets = [
        np.array([-1, -0.5, 0, 0.5, 1]),
        np.array([-2, -1, -0.5, 0, 0.5, 1, 2]),
        np.array([0, 1]),
        np.array([-1, 0, 1]),
        np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5]),
        np.array([-2, -1, 0, 1, 2]),
    ]
    
    # Try ranks from lower_bound up to best_rank - 1
    # Focus on ranks that are more likely to work
    ranks_to_try = list(range(lower_bound, best_rank))
    # Prioritize: try ranks near the middle and lower ranks first
    # Sort by distance from 50 (a heuristic middle value)
    ranks_to_try.sort(key=lambda R: (abs(R - 50), R))
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        if R >= best_rank:
            continue
        
        # Number of attempts based on rank
        if R < 45:
            max_attempts = 80
        elif R < 55:
            max_attempts = 50
        elif R < 70:
            max_attempts = 30
        else:
            max_attempts = 15
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Choose entry set
            es = entry_sets[attempt % len(entry_sets)]
            
            # Generate random U and V
            U_rand = np.random.choice(es, size=(n * m, R))
            V_rand = np.random.choice(es, size=(m * p, R))
            
            # Try to solve for W
            W_sol = _try_solve_for_W(U_rand, V_rand, n, m, p, R)
            
            if W_sol is not None:
                # Verify
                if _verify(U_rand.tolist(), V_rand.tolist(), W_sol.tolist(), n, m, p):
                    best_rank = R
                    best_solution = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_sol.tolist()
                    }
                    found = True
                    break  # Move to next rank
        
        # If we found a solution at this rank, continue to try lower ranks
        # If not, continue
    
    return best_solution
