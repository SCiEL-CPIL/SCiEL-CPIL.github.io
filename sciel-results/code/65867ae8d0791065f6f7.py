"""Tensor decomposition for 2x2 by 2x16 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix A by a 2x16 matrix B to produce a 2x16 matrix C.
    
    The tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    where n=2, m=2, p=16.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks of columns.
    Divide the 16 columns into 8 blocks of 2 columns each.
    For each block, use rank-7 Strassen decomposition.
    Total rank = 8 * 7 = 56.
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def schoolbook():
        """Standard schoolbook decomposition with rank = n*m*p = 64."""
        n, m, p = DIMENSIONS
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_block_decomp():
        """
        Use Strassen's algorithm on 8 blocks of 2 columns each.
        Each block uses rank 7, total rank = 56.
        """
        n, m, p = DIMENSIONS  # n=2, m=2, p=16
        num_blocks = p // 2  # 8 blocks of 2 columns each
        rank = num_blocks * 7  # 56
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(num_blocks):
            k0 = block * 2
            k1 = k0 + 1
            base = block * 7
            
            # M1 = (A00+A11)(B0k0+B1k1)
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            V[0 * 16 + k0][base + 0] = 1
            V[1 * 16 + k1][base + 0] = 1
            W[k0 * 2 + 0][base + 0] = 1
            W[k1 * 2 + 0][base + 0] = 1
            W[k0 * 2 + 1][base + 0] = 1
            W[k1 * 2 + 1][base + 0] = 1
            
            # M2 = (A10+A11)B0k0
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            V[0 * 16 + k0][base + 1] = 1
            W[k0 * 2 + 1][base + 1] = 1
            
            # M3 = A00(B1k0-B1k1)
            U[0][base + 2] = 1
            V[1 * 16 + k0][base + 2] = 1
            V[1 * 16 + k1][base + 2] = -1
            W[k1 * 2 + 0][base + 2] = 1
            
            # M4 = A11(B0k1-B0k0)
            U[3][base + 3] = 1
            V[0 * 16 + k1][base + 3] = 1
            V[0 * 16 + k0][base + 3] = -1
            W[k0 * 2 + 0][base + 3] = 1
            W[k0 * 2 + 1][base + 3] = 1
            
            # M5 = (A00+A01)B1k1
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            V[1 * 16 + k1][base + 4] = 1
            W[k0 * 2 + 0][base + 4] = -1
            W[k1 * 2 + 0][base + 4] = 1
            
            # M6 = (A10-A00)(B0k0+B0k1)
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            V[0 * 16 + k0][base + 5] = 1
            V[0 * 16 + k1][base + 5] = 1
            W[k1 * 2 + 1][base + 5] = 1
            
            # M7 = (A01-A11)(B1k0+B1k1)
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            V[1 * 16 + k0][base + 6] = 1
            V[1 * 16 + k1][base + 6] = 1
            W[k0 * 2 + 0][base + 6] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Vectorized verification: for each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    try:
        decomp = strassen_block_decomp()
        if verify(decomp):
            return decomp
        else:
            # Debug: print which (i,j,k) failed
            U = np.array(decomp["U"], dtype=float)
            V = np.array(decomp["V"], dtype=float)
            W = np.array(decomp["W"], dtype=float)
            n, m, p = DIMENSIONS
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            pass  # silently fall through
            return schoolbook()
    except Exception:
        pass
    
    return schoolbook()
# EVOLVE-BLOCK-END
