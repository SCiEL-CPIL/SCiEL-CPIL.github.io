"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def build_base():
        R = 54
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0

        def put(u_pairs, v_pairs, w_pairs):
            nonlocal c
            for idx, val in u_pairs:
                U[idx, c] += val
            for idx, val in v_pairs:
                V[idx, c] += val
            for idx, val in w_pairs:
                W[idx, c] += val
            c += 1

        def strassen(ja, jb, ka, kb):
            ua0 = 0 * m + ja
            ub0 = 1 * m + ja
            ua1 = 0 * m + jb
            ub1 = 1 * m + jb
            
            # Correct indices for U (rows i, cols j)
            # U[i*m + j]
            # A is 2x5. i in 0,1. j in 0..4.
            # Strassen block uses A[i, ja], A[i, jb].
            
            # Standard Strassen for 2x2x2:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # U indices:
            # A00 -> U[0*m + ja]
            # A11 -> U[1*m + jb]
            # A10 -> U[1*m + ja]
            # A01 -> U[0*m + jb]
            
            # V indices:
            # B00 -> V[ja*p + ka]
            # B11 -> V[jb*p + kb]
            # B01 -> V[ja*p + kb]
            # B10 -> V[jb*p + ka]
            
            # W indices:
            # C00 -> W[ka*n + 0]
            # C11 -> W[kb*n + 1]
            # C10 -> W[kb*n + 0]
            # C01 -> W[ka*n + 1]
            
            # M1
            put([(0*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1), (kb*n+1, 1)])
            # M2
            put([(1*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1)], [(ka*n+1, 1), (kb*n+1, -1)])
            # M3
            put([(0*m+ja, 1)], [(ja*p+kb, 1), (jb*p+kb, -1)], [(kb*n+0, 1), (kb*n+1, 1)])
            # M4
            put([(1*m+jb, 1)], [(jb*p+ka, 1), (ja*p+ka, -1)], [(ka*n+0, 1), (ka*n+1, 1)])
            # M5
            put([(0*m+ja, 1), (0*m+jb, 1)], [(jb*p+kb, 1)], [(ka*n+0, -1), (kb*n+0, 1)])
            # M6
            put([(1*m+ja, 1), (0*m+ja, -1)], [(ja*p+ka, 1), (ja*p+kb, 1)], [(kb*n+1, 1)])
            # M7
            put([(0*m+jb, 1), (1*m+jb, -1)], [(jb*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1)])

        for (ja, jb) in [(0, 1), (2, 3)]:
            for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
                strassen(ja, jb, ka, kb)

        for k in range(p):
            for i in range(n):
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

        return U, V, W, c

    U, V, W, R = build_base()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
