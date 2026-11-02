"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- Schoolbook rank: 2*2*14 = 56.
- Block-wise Strassen: split B (2x14) into 7 blocks of 2x2,
  multiply each with A (2x2) using Strassen (rank 7 per block).
  Total rank = 7 * 7 = 49.

Strassen for 2x2 * 2x2:
  Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[i,j],[k,l]]
  
  p1 = (a+d)(e+h)
  p2 = (c+d)e
  p3 = a(f-h)
  p4 = d(g-e)
  p5 = (a+b)h
  p6 = (c-a)(e+f)
  p7 = (b-d)(g+h)
  
  i = p1 + p4 - p5 + p7
  j = p3 + p5
  k = p2 + p4
  l = p1 - p2 + p3 + p6

Tensor convention:
  U: (n*m, rank) = (4, 49), indexed by i*m+j
  V: (m*p, rank) = (28, 49), indexed by j*p+k
  W: (p*n, rank) = (28, 49), indexed by k*n+i
  
  Requirement: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14

    num_blocks = p // 2  # 7
    spb = 7  # Strassen products per block
    rank = num_blocks * spb  # 49

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for t in range(num_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1

        for s in range(spb):
            col = t * spb + s

            if s == 0:
                # p1 = (A00+A11)(B0k0+B1k1)
                U[0, col] = 1.0
                U[3, col] = 1.0
                V[0 * p + k0, col] = 1.0
                V[1 * p + k1, col] = 1.0
                W[k0 * n + 0, col] = 1.0
                W[k1 * n + 1, col] = 1.0
            elif s == 1:
                # p2 = (A10+A11) * B0k0
                U[2, col] = 1.0
                U[3, col] = 1.0
                V[0 * p + k0, col] = 1.0
                W[k0 * n + 1, col] = 1.0
                W[k1 * n + 1, col] = -1.0
            elif s == 2:
                # p3 = A00 * (B0k1-B1k1)
                U[0, col] = 1.0
                V[0 * p + k1, col] = 1.0
                V[1 * p + k1, col] = -1.0
                W[k1 * n + 0, col] = 1.0
                W[k1 * n + 1, col] = 1.0
            elif s == 3:
                # p4 = A11 * (B1k0-B0k0)
                U[3, col] = 1.0
                V[1 * p + k0, col] = 1.0
                V[0 * p + k0, col] = -1.0
                W[k0 * n + 0, col] = 1.0
                W[k0 * n + 1, col] = 1.0
            elif s == 4:
                # p5 = (A00+A01) * B1k1
                U[0, col] = 1.0
                U[1, col] = 1.0
                V[1 * p + k1, col] = 1.0
                W[k0 * n + 0, col] = -1.0
                W[k1 * n + 0, col] = 1.0
            elif s == 5:
                # p6 = (A10-A00) * (B0k0+B0k1)
                U[0, col] = -1.0
                U[2, col] = 1.0
                V[0 * p + k0, col] = 1.0
                V[0 * p + k1, col] = 1.0
                W[k1 * n + 1, col] = 1.0
            elif s == 6:
                # p7 = (A01-A11) * (B1k0+B1k1)
                U[1, col] = 1.0
                U[3, col] = -1.0
                V[1 * p + k0, col] = 1.0
                V[1 * p + k1, col] = 1.0
                W[k0 * n + 0, col] = 1.0

    # Full verification: check all (i,j,k) triples
    if time.time() < deadline:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err

        if max_err > 1e-9:
            # Fallback to schoolbook rank-56 decomposition
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=float)
            V_fb = np.zeros((m * p, rank_fb), dtype=float)
            W_fb = np.zeros((p * n, rank_fb), dtype=float)
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_fb[ii * m + jj, col] = 1.0
                        V_fb[jj * p + kk, col] = 1.0
                        W_fb[kk * n + ii, col] = 1.0
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
