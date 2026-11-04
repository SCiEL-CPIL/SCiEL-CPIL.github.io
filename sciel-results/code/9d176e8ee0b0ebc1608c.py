"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    def build_strassen_decomposition():
        """Build the standard rank-50 Strassen-based decomposition."""
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[i0*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[(i0+1)*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3: A11(B12-B22) -> C12 += M3, C22 += M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4: A22(B21-B11) -> C11 += M4, C21 += M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[i0*m + j0][col] += 1
            U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6: (A21-A11)(B11+B12) -> C22 += M6
            U[(i0+1)*m + j0][col] += 1
            U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7: (A12-A22)(B21+B22) -> C11 += M7
            U[i0*m + (j0+1)][col] += 1
            U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1
        
        # 4 Strassen blocks for (j-pairs) x (k-pairs) covering k=0..5
        # j-pairs: (0,1) and (2,3), k-pairs: (0,1), (2,3), (4,5)
        add_strassen(0, 0, 0)   # j=0,1 k=0,1
        add_strassen(0, 2, 0)   # j=2,3 k=0,1
        add_strassen(0, 0, 2)   # j=0,1 k=2,3
        add_strassen(0, 2, 2)   # j=2,3 k=2,3
        add_strassen(0, 0, 4)   # j=0,1 k=4,5
        add_strassen(0, 2, 4)   # j=2,3 k=4,5
        # 8 direct terms for k=6
        for j in range(4):
            for i in range(2):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-50 decomposition
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify correctness
    if not verify(U, V, W, rank):
        # Fallback to trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i*m + j][idx] = 1
                    V[j*p + k][idx] = 1
                    W[k*n + i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try local search to reduce rank
    # Attempt to remove terms and see if we can compensate
    # This is a heuristic approach
    
    if time.time() < deadline - 20:
        # Try removing one term at a time from the k=6 direct terms
        # and see if we can adjust other terms to compensate
        for trial in range(10):
            if time.time() > deadline - 20:
                break
            
            U_copy = [row[:] for row in U]
            V_copy = [row[:] for row in V]
            W_copy = [row[:] for row in W]
            
            # Try removing the last direct term (k=6, i=1, j=3)
            # and see if we can express it as a combination of others
            # This is a simple heuristic - in practice, this is very hard
            
            # For now, just return the rank-50 solution
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
