"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, time_limit, rng):
    """Try to find a decomposition of given rank using random search."""
    start = time.time()
    
    # Random initialization
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    
    # We need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of n*m*p = 56 equations in rank*3 = 3*rank unknowns (but coupled)
    
    # Use iterative refinement: for each equation, adjust one variable
    # Actually, let's use a different approach: fix U and V, solve for W
    # W[k*n+i, c] appears in equations where (i,j,k) has specific i and k
    
    # For a given (i, k), the equations are over j:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1, for j = 0, ..., m-1
    
    # So for each (i, k) pair, we have m equations in rank unknowns (W[k*n+i, c] for c=0..rank-1)
    # If rank >= m, we can potentially solve these
    
    # Let's try: fix U and V randomly, then for each (i,k), solve the linear system for W row
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        
        # For each (i, k), build the system:
        # For j in 0..m-1: sum_c U[i*m+j, c] * V[j*p+k, c] * W_row[c] = 1
        # This is: (U_row_i * V_row_k)^T * W_row = 1
        # Where U_row_i is the submatrix of U for fixed i (m rows), 
        # V_row_k is the submatrix of V for fixed k (m rows)
        
        W = np.zeros((p*n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build the m x rank matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                # Solve A * W_row = ones(m)
                try:
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if solution is exact
                    if np.allclose(A @ W_row, np.ones(m), atol=1e-10):
                        W[k*n+i, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-9):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = (rank_school, U_school, V_school, W_school)
    
    # Try to find better decompositions
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But we need to check if lower ranks are possible
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 28 upward
    for rank in range(28, rank_school):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, deadline - time.time() - 5, rng)
        if result is not None:
            U, V, W = result
            best = (rank, U, V, W)
            break  # Found a valid one, stop
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
