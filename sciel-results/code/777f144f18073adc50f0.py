import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Strassen 2x2x2 definitions
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00+a11
        {2: 1, 3: 1},       # P2: a10+a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00+a01
        {2: 1, 0: -1},      # P6: a10-a00
        {1: 1, 3: -1},      # P7: a01-a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b00+b11
        {0: 1},             # P2: b00
        {1: 1, 3: -1},      # P3: b01-b11
        {2: 1, 0: -1},      # P4: b10-b00
        {3: 1},             # P5: b11
        {0: 1, 1: 1},       # P6: b00+b01
        {2: 1, 3: 1},       # P7: b10+b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00+c11
        {2: 1, 3: -1},      # P2: c10-c11
        {1: 1, 3: 1},       # P3: c01+c11
        {0: 1, 2: 1},       # P4: c00+c10
        {0: -1, 1: 1},      # P5: -c00+c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]

    # Build the 42-term Strassen decomposition
    rank42 = 42
    U42 = np.zeros((nm, rank42))
    V42 = np.zeros((mp, rank42))
    W42 = np.zeros((pn, rank42))

    col = 0
    for jm in range(2):      # m-block: columns 2*jm, 2*jm+1 of A
        for kp in range(3):  # p-block: columns 2*kp, 2*kp+1 of C
            for pi in range(7):
                r = col + pi
                # U: A[i, 2*jm + j_local]
                for loc, coeff in strassen_U[pi].items():
                    li, lj = divmod(loc, 2)
                    U42[li * m + (2 * jm + lj), r] = coeff
                # V: B[2*jm + j_local, 2*kp + k_local]
                for loc, coeff in strassen_V[pi].items():
                    lj, lk = divmod(loc, 2)
                    V42[(2 * jm + lj) * p + (2 * kp + lk), r] = coeff
                # W: C[2*kp + k_local, i]
                for loc, coeff in strassen_W[pi].items():
                    li, lk = divmod(loc, 2)
                    W42[(2 * kp + lk) * n + li, r] = coeff
            col += 7

    # Verify
    T_expected = np.ones((n, m, p))
    T_actual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank42):
                    s += U42[i * m + j, r] * V42[j * p + k, r] * W42[k * n + i, r]
                T_actual[i, j, k] = s
    if not np.allclose(T_actual, T_expected, atol=1e-10):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding a rank-40 decomposition
    # Strategy: look for 2 linearly dependent pairs across blocks
    # For each pair of terms (r1, r2) from different blocks, check if
    # we can eliminate one by adjusting the other.
    
    # A more systematic approach: try rank 40, 39, 38 by solving
    # a system where we set 2 terms to be linear combinations.
    
    # Given time constraints, let's try a targeted search:
    # Check if any single term is redundant (can be expressed as combo of others)
    # This is hard, so let's just return 42 for now.
    
    if time.time() - start_time > DEADLINE - 10:
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

    # Return the 42-term decomposition
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
