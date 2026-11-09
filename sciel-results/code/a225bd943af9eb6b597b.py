import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS

    # Strategy: Kronecker product decomposition
    # Factorize (3, 4, 8) into (3, 2, 4) and (1, 2, 2)
    # T_{3,4,8} = T_{3,2,4} \otimes T_{1,2,2}
    # Rank = Rank(T_{3,2,4}) * Rank(T_{1,2,2})
    # Using naive for (3,2,4): rank 24
    # Using optimal for (1,2,2): rank 2
    # Total rank = 24 * 2 = 48

    # Sub-problem 1: 3 x 2 x 4 (naive, rank 24)
    n1, m1, p1 = 3, 2, 4
    rank1 = n1 * m1 * p1  # 24

    # Sub-problem 2: 1 x 2 x 2 (naive, rank 4)
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # All 4 products are independent bilinear forms, so rank = 4
    n2, m2, p2 = 1, 2, 2
    rank2 = n2 * m2 * p2  # 4

    total_rank = rank1 * rank2  # 96

    # Initialize U, V, W for the full problem
    # U: (n*m) x rank  -> 12 x 48
    # V: (m*p) x rank  -> 32 x 48
    # W: (p*n) x rank  -> 24 x 48

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # For sub-problem 1 (3x2x4) - naive decomposition:
    # Column c1 = i1*(m1*p1) + j1*p1 + k1
    # Non-zero at U1[i1*m1+j1], V1[j1*p1+k1], W1[k1*n1+i1]

    # For sub-problem 2 (1x2x2) - naive decomposition:
    # c2 = i2*(m2*p2) + j2*p2 + k2 = j2*2 + k2 (since i2 is always 0)
    # c2=0: (0,0,0), c2=1: (0,0,1), c2=2: (0,1,0), c2=3: (0,1,1)
    sub2_map = []
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                sub2_map.append((i2, j2, k2))

    for c1 in range(rank1):
        # Decode c1 into (i1, j1, k1)
        i1 = c1 // (m1 * p1)
        rem1 = c1 % (m1 * p1)
        j1 = rem1 // p1
        k1 = rem1 % p1

        for c2 in range(rank2):
            i2, j2, k2 = sub2_map[c2]

            col_full = c1 * rank2 + c2

            # Calculate full indices
            # i_full = i1*n2 + i2 = i1*1 + i2 = i1 + i2
            # j_full = j1*m2 + j2 = j1*2 + j2
            # k_full = k1*p2 + k2 = k1*2 + k2
            u_idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
            v_idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
            w_idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)

            U[u_idx][col_full] = 1
            V[v_idx][col_full] = 1
            W[w_idx][col_full] = 1

    # Check deadline
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
