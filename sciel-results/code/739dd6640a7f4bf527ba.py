"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- Schoolbook rank is 2*2*14 = 56.
- Split B (2x14) into 7 blocks of 2x2.
- Each 2x2 * 2x2 block uses Strassen's rank-7 algorithm.
- Total rank = 7 * 7 = 49.

Strassen for 2x2 * 2x2 (A=[[a,b],[c,d]], B=[[e,f],[g,h]]):
  P1 = (a+d)(e+h)
  P2 = (c+d)e
  P3 = a(f-h)
  P4 = d(g-e)
  P5 = (a+b)h
  P6 = (c-a)(e+f)
  P7 = (b-d)(g+h)
  C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6

Tensor convention:
  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14

    # Strassen-based rank-49 decomposition
    num_blocks = p // 2  # 7
    strassen_per_block = 7
    rank = num_blocks * strassen_per_block  # 49

    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    # Pre-define the 7 Strassen terms for a single 2x2 block.
    # Each term: (U_coeffs, V_local_coeffs, W_local_coeffs)
    # U row indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W local indices: 0=C00, 1=C10, 2=C01, 3=C11

    strassen_terms = [
        # P1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A10+A11)B00 -> C10+=1, C11+=-1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # P3 = A00(B01-B11) -> C01+=1, C11+=1
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # P4 = A11(B10-B00) -> C00+=1, C10+=1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # P5 = (A00+A01)B11 -> C00+=-1, C01+=1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # P6 = (A10-A00)(B00+B01) -> C11+=1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7 = (A01-A11)(B10+B11) -> C00+=1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        k0 = 2 * t      # first column of block
        k1 = 2 * t + 1  # second column of block

        for s in range(strassen_per_block):
            col = t * strassen_per_block + s
            u_coeffs, v_local, w_local = strassen_terms[s]

            # Fill U (global indices: i*m+j)
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] = coeff

            # Fill V: local B index -> global V index
            # Local 0=(0,k0), 1=(0,k1), 2=(1,k0), 3=(1,k1)
            for v_loc, coeff in v_local.items():
                j = v_loc // 2
                k_col = k0 + (v_loc % 2)
                v_idx = j * p + k_col
                V[v_idx][col] = coeff

            # Fill W: local C index -> global W index (k*n+i)
            # Local 0=C00=(i=0,k=k0), 1=C10=(i=1,k=k0),
            #        2=C01=(i=0,k=k1), 3=C11=(i=1,k=k1)
            for w_loc, coeff in w_local.items():
                i = w_loc % 2       # 0 for C0x, 1 for C1x
                k_col = k0 + (w_loc // 2)  # k0 for x0, k1 for x1
                w_idx = k_col * n + i
                W[w_idx][col] = coeff

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback to schoolbook rank-56
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
