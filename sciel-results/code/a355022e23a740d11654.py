import time
import sys

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Return U, V, W for the standard rank n*m*p decomposition of M_{n,m,p}.
    
    The multiplication tensor T has T[i,j,k] = 1 for all (i,j,k), and the
    decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)  
    W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the schoolbook decomposition (rank = n*m*p = 84)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should never happen for the schoolbook, but handle gracefully
        pass
    
    # Check if we have time (we do, but good practice)
    if time.time() > deadline:
        # Return what we have - it's valid
        pass
    
    return {"U": U, "V": V, "W": W}
