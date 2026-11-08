"""Matrix multiplication 2x4 x 4x4 tensor decomposition with rank 28 via block Strassen."""
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    col = 0
    
    for q in range(2):
        for r in range(2):
            j0, j1 = 2 * q, 2 * q + 1
            k0, k1 = 2 * r, 2 * r + 1
            
            terms = [
                # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                {'U': {0*m + j0: 1, 1*m + j1: 1},
                 'V': {j0*p + k0: 1, j1*p + k1: 1},
                 'W': {k0*n + 0: 1, k1*n + 1: 1}},
                # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                {'U': {1*m + j0: 1, 1*m + j1: 1},
                 'V': {j0*p + k0: 1},
                 'W': {k0*n + 1: 1, k1*n + 1: -1}},
                # M3 = A[0,j0] * (B[j1,k0]-B[j0,k0])
                {'U': {0*m + j0: 1},
                 'V': {j1*p + k0: 1, j0*p + k0: -1},
                 'W': {k1*n + 0: 1, k1*n + 1: 1}},
                # M4 = A[1,j1] * (B[j0,k1]-B[j1,k1])
                {'U': {1*m + j1: 1},
                 'V': {j0*p + k1: 1, j1*p + k1: -1},
                 'W': {k0*n + 0: 1, k0*n + 1: 1}},
                # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                {'U': {0*m + j0: 1, 0*m + j1: 1},
                 'V': {j1*p + k1: 1},
                 'W': {k0*n + 0: -1, k1*n + 0: 1}},
                # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                {'U': {1*m + j0: 1, 0*m + j0: -1},
                 'V': {j0*p + k0: 1, j0*p + k1: 1},
                 'W': {k1*n + 1: 1}},
                # M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                {'U': {0*m + j1: 1, 1*m + j1: -1},
                 'V': {j1*p + k0: 1, j1*p + k1: 1},
                 'W': {k0*n + 0: 1}},
            ]
            
            for term in terms:
                for idx, coeff in term['U'].items():
                    U[idx][col] += coeff
                for idx, coeff in term['V'].items():
                    V[idx][col] += coeff
                for idx, coeff in term['W'].items():
                    W[idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
