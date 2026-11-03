"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # A is n×m = 2×4, B is m×p = 4×8, C is n×p = 2×8
    # Flattened: A[i*m+k], B[k*p+j], C[i*p+j]
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)  
    # W shape: (p*n, rank) = (16, rank)
    # W index: k*n+i where k is column of C, i is row of C
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    t = 0
    for a in range(2):  # col group of A / row group of B (0 or 1)
        for b in range(4):  # col group of B (0..3)
            # A block indices (flattened 2x4, row-major)
            u00 = 0 * m + 2 * a + 0      # row 0, col 2a
            u01 = 0 * m + 2 * a + 1      # row 0, col 2a+1
            u10 = 1 * m + 2 * a + 0      # row 1, col 2a
            u11 = 1 * m + 2 * a + 1      # row 1, col 2a+1
            
            # B block indices (flattened 4x8, row-major)
            v00 = (2 * a) * p + 2 * b + 0
            v01 = (2 * a) * p + 2 * b + 1
            v10 = (2 * a + 1) * p + 2 * b + 0
            v11 = (2 * a + 1) * p + 2 * b + 1
            
            # C block indices (trace convention: W[k*n+i] where k=col, i=row)
            w00 = (2 * b) * n + 0        # col 2b, row 0
            w01 = (2 * b + 1) * n + 0    # col 2b+1, row 0
            w10 = (2 * b) * n + 1        # col 2b, row 1
            w11 = (2 * b + 1) * n + 1    # col 2b+1, row 1
            
            # Strassen:
            # m1 = (a00+a11)(b00+b11) -> c00,c01,c10,c11
            # m2 = (a10+a11)b00       -> c10,c11
            # m3 = a00(b01-b11)       -> c01,c11
            # m4 = a11(b10-b00)       -> c00,c10
            # m5 = (a00+a01)b11       -> c00,c01
            # m6 = (a10-a00)(b00+b01) -> c11
            # m7 = (a01-a11)(b10+b11) -> c00,c01
            
            # s=0: m1
            U[u00, t] += 1
            U[u11, t] += 1
            V[v00, t] += 1
            V[v11, t] += 1
            W[w00, t] += 1
            W[w01, t] += 1
            W[w10, t] += 1
            W[w11, t] += 1
            t += 1
            
            # s=1: m2
            U[u10, t] += 1
            U[u11, t] += 1
            V[v00, t] += 1
            W[w10, t] += 1
            W[w11, t] += 1
            t += 1
            
            # s=2: m3
            U[u00, t] += 1
            V[v01, t] += 1
            V[v11, t] += -1
            W[w01, t] += 1
            W[w11, t] += 1
            t += 1
            
            # s=3: m4
            U[u11, t] += 1
            V[v10, t] += 1
            V[v00, t] += -1
            W[w00, t] += 1
            W[w10, t] += 1
            t += 1
            
            # s=4: m5
            U[u00, t] += 1
            U[u01, t] += 1
            V[v11, t] += 1
            W[w00, t] += 1
            W[w01, t] += 1
            t += 1
            
            # s=5: m6
            U[u10, t] += 1
            U[u00, t] += -1
            V[v00, t] += 1
            V[v01, t] += 1
            W[w11, t] += 1
            t += 1
            
            # s=6: m7
            U[u01, t] += 1
            U[u11, t] += -1
            V[v10, t] += 1
            V[v11, t] += 1
            W[w00, t] += 1
            W[w01, t] += 1
            t += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
