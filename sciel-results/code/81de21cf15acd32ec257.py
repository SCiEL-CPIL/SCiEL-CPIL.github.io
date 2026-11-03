"""Matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build target tensor T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def check(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)
    
    # Try structured rank-84 decomposition
    # Strategy: use 4x4x4 decomposition (rank 23) extended to 6 columns
    # For 4x4x4, known rank is 23. For 4x4x6, we can potentially do better than 96.
    
    # Alternative: use a "split" approach
    # Split p=6 into 3+3, each 4x4x3 has schoolbook rank 48, total 96. No help.
    
    # Try: use the fact that for fixed i, the row i of A times B gives row i of C
    # This is a linear map from B (4x6) to C_row (1x6) for each i
    
    # Let me try a direct construction for rank 84
    # Save 12 terms by sharing some factors
    
    # Approach: for each (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
    # The tensor condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta
    
    # Let me try to find a valid rank-84 by solving a system
    # This is complex, so let me try a heuristic: start with 96 and try to zero out 12 terms
    
    best_rank = None
    best_result = None
    
    for r in range(96, 83, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find rank-r by starting with schoolbook and setting some to zero
        # then adjusting
        U = np.zeros((n*m, r), dtype=np.float64)
        V = np.zeros((m*p, r), dtype=np.float64)
        W = np.zeros((p*n, r), dtype=np.float64)
        
        # Initialize with first r schoolbook terms
        for idx in range(r):
            i = idx // (m*p)
            rem = idx % (m*p)
            j = rem // p
            k = rem % p
            if i < n and j < m and k < p:
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
        
        # Coordinate descent to fix the decomposition
        err = np.sum((T - np.einsum('ar,br,cr->abc', U, V, W))**2)
        
        for it in range(100):
            if time.time() > deadline - 8:
                break
            improved = False
            for c in range(r):
                for _ in range(2):
                    Uo, Vo, Wo = U[:,c].copy(), V[:,c].copy(), W[:,c].copy()
                    U[:,c] += np.random.randn(n*m)*0.003
                    V[:,c] += np.random.randn(m*p)*0.003
                    W[:,c] += np.random.randn(p*n)*0.003
                    new_err = np.sum((T - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        U[:,c], V[:,c], W[:,c] = Uo, Vo, Wo
            if not improved or err < 1e-10:
                break
        
        if err < 1e-10:
            best_rank = r
            best_result = (U, V, W)
            break
    
    if best_result:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
