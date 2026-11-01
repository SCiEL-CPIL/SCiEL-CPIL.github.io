"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Partition into 8 independent 2x2 * 2x2 block multiplications,
    each decomposed via Strassen (rank 7). Total rank = 8 * 7 = 56.
    
    Block structure:
      A (2x4): split columns into 2 blocks of 2x2 -> A[u] for u in {0,1}
      B (4x8): split into 2x4 blocks of 2x2 -> B[u,v] for u in {0,1}, v in {0,1,2,3}
      C (2x8): split columns into 4 blocks of 2x2 -> C[v] for v in {0,1,2,3}
      
      C[v] = A[0] * B[0,v] + A[1] * B[1,v]
      
    Each 2x2 * 2x2 product uses Strassen's 7 multiplications.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U: (n*m) x rank = 8 x 56
    # V: (m*p) x rank = 32 x 56  
    # W: (p*n) x rank = 16 x 56
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Index conventions:
    # A[i,j] -> U index: i*m + j = i*4 + j
    # B[j,k] -> V index: j*p + k = j*8 + k
    # C[i,k] -> W index: k*n + i = k*2 + i
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices (2x2 block at column offset 2u)
            a11 = 0 * 4 + 2 * u      # A[0, 2u]
            a12 = 0 * 4 + 2 * u + 1  # A[0, 2u+1]
            a21 = 1 * 4 + 2 * u      # A[1, 2u]
            a22 = 1 * 4 + 2 * u + 1  # A[1, 2u+1]
            
            # B block indices (2x2 block at row offset 2u, col offset 2v)
            b11 = 2 * u * 8 + 2 * v      # B[2u, 2v]
            b12 = 2 * u * 8 + 2 * v + 1  # B[2u, 2v+1]
            b21 = (2 * u + 1) * 8 + 2 * v      # B[2u+1, 2v]
            b22 = (2 * u + 1) * 8 + 2 * v + 1  # B[2u+1, 2v+1]
            
            # C block indices (2x2 block at col offset 2v)
            # C[0,2v] -> W: (2v)*2 + 0 = 4v
            # C[0,2v+1] -> W: (2v+1)*2 + 0 = 4v+2
            # C[1,2v] -> W: (2v)*2 + 1 = 4v+1
            # C[1,2v+1] -> W: (2v+1)*2 + 1 = 4v+3
            c11 = 4 * v      # C[0, 2v]
            c12 = 4 * v + 2  # C[0, 2v+1]
            c21 = 4 * v + 1  # C[1, 2v]
            c22 = 4 * v + 3  # C[1, 2v+1]
            
            # Strassen 7 products for X*Y where X=A[u], Y=B[u,v]
            # X11=a11, X12=a12, X21=a21, X22=a22
            # Y11=b11, Y12=b12, Y21=b21, Y22=b22
            
            # M1 = (X11+X22)*(Y11+Y22)
            # Contributes: C11 += 1, C22 += 1
            U[a11][base_col] += 1
            U[a22][base_col] += 1
            V[b11][base_col] += 1
            V[b22][base_col] += 1
            W[c11][base_col] += 1
            W[c22][base_col] += 1
            
            # M2 = (X21+X22)*Y11
            # Contributes: C21 += 1, C22 -= 1
            U[a21][base_col + 1] += 1
            U[a22][base_col + 1] += 1
            V[b11][base_col + 1] += 1
            W[c21][base_col + 1] += 1
            W[c22][base_col + 1] -= 1
            
            # M3 = X11*(Y12-Y22)
            # Contributes: C12 += 1, C22 += 1
            U[a11][base_col + 2] += 1
            V[b12][base_col + 2] += 1
            V[b22][base_col + 2] -= 1
            W[c12][base_col + 2] += 1
            W[c22][base_col + 2] += 1
            
            # M4 = X22*(Y21-Y11)
            # Contributes: C11 += 1, C21 += 1
            U[a22][base_col + 3] += 1
            V[b21][base_col + 3] += 1
            V[b11][base_col + 3] -= 1
            W[c11][base_col + 3] += 1
            W[c21][base_col + 3] += 1
            
            # M5 = (X11+X12)*Y22
            # Contributes: C11 -= 1, C12 += 1
            U[a11][base_col + 4] += 1
            U[a12][base_col + 4] += 1
            V[b22][base_col + 4] += 1
            W[c11][base_col + 4] -= 1
            W[c12][base_col + 4] += 1
            
            # M6 = (X21-X11)*(Y11+Y12)
            # Contributes: C22 += 1
            U[a21][base_col + 5] += 1
            U[a11][base_col + 5] -= 1
            V[b11][base_col + 5] += 1
            V[b12][base_col + 5] += 1
            W[c22][base_col + 5] += 1
            
            # M7 = (X12-X22)*(Y21+Y22)
            # Contributes: C11 += 1
            U[a12][base_col + 6] += 1
            U[a22][base_col + 6] -= 1
            V[b21][base_col + 6] += 1
            V[b22][base_col + 6] += 1
            W[c11][base_col + 6] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
