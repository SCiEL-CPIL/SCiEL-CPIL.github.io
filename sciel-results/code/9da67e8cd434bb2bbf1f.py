import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Exact rank-88 tensor decomposition for 3x4x8 matrix multiplication.
    The multiplication tensor T[i,j,k] is decomposed as sum of rank-1 terms.
    Part 1 (32 terms): schoolbook for i=0
    Part 2 (56 terms): Strassen 2x2x2 blocks for i=1,2
    """
    t0 = time.time()
    hard_deadline = t0 + 140.0

    n, m, p = 3, 4, 8
    R = 88
    u_dim = n * m  # 12
    v_dim = m * p  # 32
    w_dim = p * n  # 24

    U = np.zeros((u_dim, R), dtype=int)
    V = np.zeros((v_dim, R), dtype=int)
    W = np.zeros((w_dim, R), dtype=int)

    # ---- Part 1: i=0 schoolbook, 4*8=32 terms ----
    # For each j in 0..3, k in 0..7:
    #   U[0*m+j, t] = 1
    #   V[j*p+k, t] = 1
    #   W[k*n+0, t] = 1
    base = 0
    for j in range(m):
        for k in range(p):
            idx = base + j * p + k
            U[j, idx] = 1
            V[j * p + k, idx] = 1
            W[k * n, idx] = 1
    base = 32

    # ---- Part 2: i=1,2 using Strassen 2x2x2, 4 blocks * 7 = 28 terms each, total 56 ----
    # Strassen 2x2x2 decomposition (7 scalar multiplications)
    # For 2x2 matrices: C = A*B
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6

    # Each 2x2x2 block operates on 2x2 submatrices.
    # For our 2x4x8 problem (i=1,2), we have:
    #   A is 2x4 (rows i=1,2; cols j=0..3)
    #   B is 4x8 (rows j=0..3; cols k=0..7)
    #   C is 2x8 (rows i=1,2; cols k=0..7)
    #
    # We partition j into 2 blocks of 2, and k into 4 blocks of 2.
    # For each (j_block, k_block) pair, we apply Strassen to the 2x2x2 sub-problem.

    # Strassen factor matrices for 2x2x2 (7 terms):
    # U factors (A-side): 7x4 where 4 = 2*2 (i,j indices within the 2x2 block)
    # V factors (B-side): 7x4 where 4 = 2*2 (j,k indices within the 2x2 block)
    # W factors (C-side): 7x4 where 4 = 2*2 (i,k indices within the 2x2 block)

    # Index convention: for a 2x2 matrix with rows (r0,r1) and cols (c0,c1),
    # the flat index is r*2+c.
    # A indices: (i,j) where i in {0,1} (local), j in {0,1} (local)
    # B indices: (j,k) where j in {0,1} (local), k in {0,1} (local)
    # C indices: (i,k) where i in {0,1} (local), k in {0,1} (local)

    # Strassen M1 = (A11+A22)(B11+B22)
    #   A11 = A[0,0], A22 = A[1,1] -> U1 = [1,0, 0,1] in (i,j) flat order
    #   B11 = B[0,0], B22 = B[1,1] -> V1 = [1,0, 0,1] in (j,k) flat order
    #   C11 += 1, C12 += 0, C21 += 0, C22 -= 1 -> W1 = [1,0, 0,-1]
    # Strassen M2 = (A21+A22)B11
    #   A21 = A[1,0], A22 = A[1,1] -> U2 = [0,0, 1,1]
    #   B11 = B[0,0] -> V2 = [1,0, 0,0]
    #   C11 += 0, C12 += 0, C21 += 1, C22 -= 1 -> W2 = [0,0, 1,-1]
    # Strassen M3 = A11(B21-B22)
    #   A11 = A[0,0] -> U3 = [1,0, 0,0]
    #   B21 = B[1,0], B22 = B[1,1] -> V3 = [0,0, 1,-1]
    #   C11 += 0, C12 += 1, C21 += 0, C22 += 1 -> W3 = [0,1, 0,1]
    # Strassen M4 = A22(B12-B11)
    #   A22 = A[1,1] -> U4 = [0,0, 0,1]
    #   B12 = B[0,1], B11 = B[0,0] -> V4 = [0,1, -1,0]
    #   C11 += 1, C12 += 0, C21 += 1, C22 += 0 -> W4 = [1,0, 1,0]
    # Strassen M5 = (A11+A12)B22
    #   A11 = A[0,0], A12 = A[0,1] -> U5 = [1,1, 0,0]
    #   B22 = B[1,1] -> V5 = [0,0, 0,1]
    #   C11 -= 1, C12 += 1, C21 += 0, C22 += 0 -> W5 = [-1,1, 0,0]
    # Strassen M6 = (A21-A11)(B11+B12)
    #   A21 = A[1,0], A11 = A[0,0] -> U6 = [-1,0, 1,0]
    #   B11 = B[0,0], B12 = B[0,1] -> V6 = [1,1, 0,0]
    #   C11 += 0, C12 += 0, C21 += 0, C22 += 1 -> W6 = [0,0, 0,1]
    # Strassen M7 = (A12-A22)(B21+B22)
    #   A12 = A[0,1], A22 = A[1,1] -> U7 = [0,1, 0,-1]
    #   B21 = B[1,0], B22 = B[1,1] -> V7 = [0,0, 1,1]
    #   C11 += 1, C12 += 0, C21 += 0, C22 += 0 -> W7 = [1,0, 0,0]

    S_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ], dtype=int)

    S_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=int)

    S_W = np.array([
        [1, 0, 0, -1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=int)

    for j_blk in range(m // 2):
        for k_blk in range(p // 2):
            for t in range(7):
                if time.time() > hard_deadline:
                    break
                col = base + (j_blk * (p // 2) + k_blk) * 7 + t

                # U: local (li, lj) -> global (gi=1+li, gj=j_blk*2+lj)
                for li in range(2):
                    for lj in range(2):
                        val = S_U[t, li * 2 + lj]
                        if val != 0:
                            gi = 1 + li
                            gj = j_blk * 2 + lj
                            U[gi * m + gj, col] += val

                # V: local (lj, lk) -> global (gj=j_blk*2+lj, gk=k_blk*2+lk)
                for lj in range(2):
                    for lk in range(2):
                        val = S_V[t, lj * 2 + lk]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            gk = k_blk * 2 + lk
                            V[gj * p + gk, col] += val

                # W: local (li, lk) -> global (gi=1+li, gk=k_blk*2+lk)
                for li in range(2):
                    for lk in range(2):
                        val = S_W[t, li * 2 + lk]
                        if val != 0:
                            gi = 1 + li
                            gk = k_blk * 2 + lk
                            W[gk * n + gi, col] += val

    # Safety: if we ran out of time, return what we have
    # (but for rank 88 this should complete well within time)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
