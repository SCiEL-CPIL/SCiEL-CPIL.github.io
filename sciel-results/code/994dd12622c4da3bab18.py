import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition for M_{n,m,p}.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    Condition: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] == delta(j, j') for all i, k, j, j'
    """
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _als_solve(rank, n, m, p, max_iters=500, restarts=100, timeout=100):
    """
    Uses Alternating Least Squares to find a rank 'rank' decomposition.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start_time = time.time()
    
    # Pre-compute indices for efficiency?
    # U rows: i*m + j
    # V rows: j*p + k
    # W rows: k*n + i
    
    for restart in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A and b
                    A = np.zeros((p * m, rank))
                    b = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for jp in range(m):
                            # A[idx, c] = W[k*n+i, c] * V[jp*p+k, c]
                            A[idx] = W[k * n + i] * V[jp * p + k]
                            b[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    # Solve A x = b
                    x, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n * m * m, rank))
                    b = np.zeros(n * m * m)
                    idx = 0
                    for i in range(n):
                        for j_prime in range(m):
                            for jp in range(m):
                                # A[idx, c] = W[k*n+i, c] * U[i*m+j_prime, c]
                                A[idx] = W[k * n + i] * U[i * m + j_prime]
                                b[idx] = 1.0 if j_prime == jp else 0.0
                                idx += 1
                    x, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m * m, rank))
                    b = np.zeros(m * m)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            # A[idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                            A[idx] = U[i * m + j] * V[jp * p + k]
                            b[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    x, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x
            
            # Check error
            # We can check a subset or full error
            # Full error check is cheap enough for small dimensions
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < 1e-12:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify strictly
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 110  # Leave 10s margin
    
    # Try ranks from 19 up to 23
    # If we find a valid one, return it
    # Start with lower ranks as they are better
    for r in range(19, 24):
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
        result = _als_solve(r, n, m, p, max_iters=200, restarts=20, timeout=deadline - time.time() - 5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
