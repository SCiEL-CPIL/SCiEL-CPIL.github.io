"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Convention: C[i,k] = sum_c f_c(A) * g_c(B) * W[k*n+i, c]
where f_c(A) = sum_{i,j} U[i*m+j, c] * A[i,j]
      g_c(B) = sum_{j,k} V[j*p+k, c] * B[j,k]

Schoolbook: one term per (i,j,k) triple, rank = n*m*p = 56.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify: C[i,k] = sum_c [sum_{i',j'} U[i'*m+j',c]*A[i',j']] * [sum_{j',k'} V[j'*p+k',c]*B[j',k']] * W[k*n+i,c]
    # For schoolbook term (i0,j0,k0): f_c(A)=A[i0,j0], g_c(B)=B[j0,k0], W[k*n+i,c]=1 iff i=i0,k=k0
    # So C[i,k] = sum_{j0} A[i,j0]*B[j0,k] = (A@B)[i,k] ✓
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
