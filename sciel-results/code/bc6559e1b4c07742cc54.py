import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a, col] = 1
                    V[a * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-40 decomposition
    # The tensor T has 48 nonzero entries.
    # If we can find 40 rank-1 tensors whose sum is T, we are done.
    # This is a non-linear optimization problem.
    # Given the time constraints and the "keep design" instruction, we will attempt a quick local search
    # but likely fallback to the known rank-44 solution.

    # Rank 44 Construction
    # 1. Split A (2x3) into A1 (2x2) and A2 (2x1)
    # 2. Split B (3x8) into B1 (2x8) and B2 (1x8)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 by 2x8. Split B1 into four 2x2 blocks B10, B11, B12, B13.
    # A1*B1j uses Strassen 2x2x2 (rank 7). 4 blocks * 7 = 28.
    # A2*B2: 2x1 by 1x8. Rank 16.
    # Total Rank 44.

    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen factors for 2x2x2
    # C = AB
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # U (A factors), V (B factors), W (C factors)
    # Indices: A: i*2+a, B: a*2+k, C: k*2+i

    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8)
    # A1 corresponds to rows 0,1 and cols 0,1 of A. Index i*m + a for i in 0,1 and a in 0,1.
    # B1 corresponds to rows 0,1 of B and cols 0..7 of B. Index a*p + k for a in 0,1 and k in 0..7.
    # C part corresponds to rows 0..7 and cols 0,1 of C. Index k*n + i for k in 0..7 and i in 0,1.

    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li*2 + la]
                    if val != 0:
                        U[li * m + la, col] = val
            # V
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la*2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk), col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk*2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li, col] = val
            col += 1

    if (time.time() - t0) > deadline:
        return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 corresponds to rows 0,1 and col 2 of A. Index i*m + 2.
    # B2 corresponds to row 2 of B and cols 0..7. Index 2*p + k.
    # C part corresponds to rows 0..7 and cols 0,1 of C. Index k*n + i.
    for i in range(2):
        for k in range(8):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    if (time.time() - t0) > deadline:
        return _schoolbook()

    # Validation
    # T[i, a, k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i, a, k.
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Efficient validation using einsum or loops
    # T[i, a, k] = sum_c U[i, a, c] V[a, k, c] W[k, i, c]
    # Reshape to 3D
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    
    # Compute tensor T
    # T[i, a, k] = sum_c U3[i, a, c] * V3[a, k, c] * W3[k, i, c]
    
    # We can just loop over the 48 elements since rank is small (44)
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0.0
                u_row = U_np[i * m + a]
                v_row = V_np[a * p + k]
                w_row = W_np[k * n + i]
                # Dot product of element-wise product
                acc = np.sum(u_row * v_row * w_row)
                if abs(acc - 1.0) > 1e-9:
                    return _schoolbook()

    return {"U": U_np, "V": V_np, "W": W_np}

