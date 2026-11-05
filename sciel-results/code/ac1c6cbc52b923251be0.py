"""Combined program: block-partitioned Strassen + Winograd-style search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    def check_time():
        return time.time() - start_time > time_limit - 10
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    error += (val - 1.0) ** 2
        return error < 1e-9
    
    def validate_fast(U_arr, V_arr, W_arr):
        """Fast validation using numpy."""
        rank = U_arr.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    error += (val - 1.0) ** 2
        return error < 1e-9
    
    def schoolbook_block(i0, j0, k0, ns, ms, ps, U, V, W, col):
        """Schoolbook decomposition for a sub-block."""
        for i_s in range(ns):
            for j_s in range(ms):
                for k_s in range(ps):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    def strassen2_block(i0, j0, k0, U, V, W, col):
        """Strassen 2x2x2 block decomposition (rank 7)."""
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+0)][col] = -1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+0)*n+(i0+1)][col] = 1
        col += 1
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = -1; W[(k0+1)*n+(i0+0)][col] = 1
        col += 1
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+0)][col] = -1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[(i0+0)*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        col += 1
        return col
    
    # Strategy 1: Block partition with Strassen for 2x2x2 blocks
    def block_partition_strassen():
        """
        Partition n=[1,2], m=[2,1,2], p=[2,1,2] to maximize 2x2x2 Strassen blocks.
        """
        rank_estimate = 71  # From inspiration
        U = [[0] * rank_estimate for _ in range(n * m)]
        V = [[0] * rank_estimate for _ in range(m * p)]
        W = [[0] * rank_estimate for _ in range(p * n)]
        
        n_parts = [1, 2]
        m_parts = [2, 1, 2]
        p_parts = [2, 1, 2]
        n_off = [0, 1]
        m_off = [0, 2, 3]
        p_off = [0, 2, 3]
        
        col = 0
        for bi in range(2):
            for bj in range(3):
                for bk in range(3):
                    ns = n_parts[bi]
                    ms = m_parts[bj]
                    ps = p_parts[bk]
                    i0 = n_off[bi]
                    j0 = m_off[bj]
                    k0 = p_off[bk]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        col = strassen2_block(i0, j0, k0, U, V, W, col)
                    else:
                        col = schoolbook_block(i0, j0, k0, ns, ms, ps, U, V, W, col)
        
        # Trim if we used fewer columns
        if col < rank_estimate:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
        
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try block partition with Strassen
    if not check_time():
        block_result = block_partition_strassen()
        if block_result and validate(block_result["U"], block_result["V"], block_result["W"]):
            block_rank = len(block_result["U"][0]) if block_result["U"] else 0
            if block_rank < best_rank:
                best_result = block_result
                best_rank = block_rank
    
    # Strategy 2: Optimization to find lower rank
    def try_rank_optimized(rank, max_restarts=3):
        """Try to find a rank-`rank` decomposition using optimization with multiple restarts."""
        if check_time():
            return None
        
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if check_time():
                break
            
            # Better initialization: use schoolbook-like structure
            U_init = np.zeros((n * m, rank))
            V_init = np.zeros((m * p, rank))
            W_init = np.zeros((p * n, rank))
            
            # Initialize with a sparse pattern
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U_init[i * m + j, r] = 1.0
                V_init[j * p + k, r] = 1.0
                W_init[k * n + i, r] = 1.0
            
            # Add some random perturbation
            U_init += np.random.randn(n * m, rank) * 0.01
            V_init += np.random.randn(m * p, rank) * 0.01
            W_init += np.random.randn(p * n, rank) * 0.01
            
            def objective(params):
                if check_time():
                    return 1e10
                U = params[:n * m * rank].reshape(n * m, rank)
                V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                # Compute the error efficiently
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                            error += (val - 1.0) ** 2
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                                         'maxls': 50})
                
                if result.fun < best_error:
                    best_error = result.fun
                    U = result.x[:n * m * rank].reshape(n * m, rank)
                    V = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                    W = result.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if validate_fast(U_rounded, V_rounded, W_rounded):
                        best_solution = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        break
                    
                    # Also try integer rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if validate_fast(U_int, V_int, W_int):
                        best_solution = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        break
                        
            except Exception:
                pass
        
        return best_solution
    
    # Try a range of ranks below the current best
    lower_bound = max(n * m, m * p, p * n)  # 25
    
    ranks_to_try = list(range(lower_bound, best_rank + 1))
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if check_time():
            break
        
        result = try_rank_optimized(rank, max_restarts=2)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == lower_bound:
                break
    
    # Check time
    if check_time():
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
