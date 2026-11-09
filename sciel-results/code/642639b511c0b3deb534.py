"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need U (n*m x r), V (m*p x r), W (p*n x r) with:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, m*min(n,p)) = max(15, 12) = 15
    lower_bound = max(n * p, m * min(n, p))
    
    best_result = None
    
    for r in range(lower_bound, n * m * p + 1):
        if time.time() > deadline:
            break
        if best_result is not None:
            break  # We found a valid one, stop
            
        result = _try_rank(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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


def _try_rank(n, m, p, r, deadline):
    """Try to find an exact half-integer decomposition of rank r."""
    
    # Initialize with random values, then optimize
    np.random.seed(42)
    
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    # Alternating least squares: fix two, solve for the third
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # We can reformulate: for fixed U and V, W[k*n+i,c] appears in constraints
    # where (i,j,k) with that specific (k,i) pair.
    
    # Let's use a simpler approach: iterative projection
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Update U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is: for fixed (i,j), U[i*m+j,:] needs to satisfy:
        #   U[i*m+j,c] * sum_k (V[j*p+k,c] * W[k*n+i,c]) ... no, that's not right.
        
        # Actually, the constraint is: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed V and W, and fixed (i,j): 
        #   sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] = sum_k 1 = p
        # But we need each individual k to satisfy the equation.
        
        # Let me define: for fixed (i,j) and fixed c, let A_c = V[j*p+k,c] * W[k*n+i,c] for each k
        # Then we need: sum_c U[i*m+j,c] * A_c(k) = 1 for each k
        # This is a linear system in U[i*m+j,:]: 
        #   [A_1(1) A_2(1) ... A_r(1)]
        #   [A_1(2) A_2(2) ... A_r(2)]
        #   ...
        #   [A_1(p) A_2(p) ... A_r(p)]
        #   * U[i*m+j, :]^T = [1, 1, ..., 1]^T
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Build the system
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve A @ U[i*m+j, :]^T = ones(p)
                # Use least squares
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence and round to half-integers
        if iteration % 50 == 0:
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify exactness
            if _verify(n, m, p, U_round, V_round, W_round):
                return _to_lists(n, m, p, U_round, V_round, W_round)
    
    return None


def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _to_lists(n, m, p, U, V, W):
    """Convert numpy arrays to lists of lists with exact half-integer values."""
    r = U.shape[1]
    U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
