"""
Tensor decomposition for multiplying a 2×4 matrix by a 4×6 matrix.

The multiplication tensor M_{2,4,6} can be decomposed using Strassen's
algorithm for 2×2 blocks.  We partition the inner dimension m=4 into two
blocks of size 2 (A[:,0:2], A[:,2:4]) and the output dimension p=6 into
three blocks of size 2 (B[0:2,:], B[2:4,:], B[4:6,:]).  Each 2×2 × 2×2
sub-block multiplication uses Strassen's 7-term scheme, giving a total
rank of 2 × 3 × 7 = 42.

Indexing (0-based):
  U index: i*m + j  where i in [0,2), j in [0,4)   →  8 entries
  V index: j*p + k  where j in [0,4), k in [0,6)   → 24 entries
  W index: k*n + i  where k in [0,6), i in [0,2)   → 12 entries

Trace convention: C[k*n + i, r] = sum_s W[k*n+i, r] * ...
i.e. W is indexed as [k*n + i, r] per the task statement.
"""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6  # A is n×m, B is m×p, C is n×p

    # Strassen's 7 products for 2×2 matrices:
    #   M1 = (A11 + A22)(B11 + B22)
    #   M2 = (A21 + A22)B11
    #   M3 = A11(B21 - B22)
    #   M4 = A22(B12 - B11)
    #   M5 = (A11 + A12)B22
    #   M6 = (A21 - A11)(B11 + B12)
    #   M7 = (A12 - A22)(B21 + B22)
    #
    # For C = A*B (2×2 result):
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # A is indexed as A[i, j_loc] where i in {0,1}, j_loc in {0,1}
    # B is indexed as B[j_loc, k_loc] where j_loc in {0,1}, k_loc in {0,1}
    # C is indexed as C[i, k_loc] where i in {0,1}, k_loc in {0,1}

    # U_coeffs[s]: dict mapping (i, j_loc) -> coefficient for Strassen product s
    U_coeffs = [
        # s=0: (A11 + A22) → (i=0,j=0):1, (i=1,j=1):1
        {(0, 0): 1, (1, 1): 1},
        # s=1: (A21 + A22) → (i=1,j=0):1, (i=1,j=1):1
        {(1, 0): 1, (1, 1): 1},
        # s=2: A11 → (i=0,j=0):1
        {(0, 0): 1},
        # s=3: A22 → (i=1,j=1):1
        {(1, 1): 1},
        # s=4: (A11 + A12) → (i=0,j=0):1, (i=0,j=1):1
        {(0, 0): 1, (0, 1): 1},
        # s=5: (A21 - A11) → (i=1,j=0):1, (i=0,j=0):-1
        {(1, 0): 1, (0, 0): -1},
        # s=6: (A12 - A22) → (i=0,j=1):1, (i=1,j=1):-1
        {(0, 1): 1, (1, 1): -1},
    ]

    # V_coeffs[s]: dict mapping (j_loc, k_loc) -> coefficient
    V_coeffs = [
        # s=0: (B11 + B22) → (0,0):1, (1,1):1
        {(0, 0): 1, (1, 1): 1},
        # s=1: B11 → (0,0):1
        {(0, 0): 1},
        # s=2: (B21 - B22) → (1,0):1, (1,1):-1
        {(1, 0): 1, (1, 1): -1},
        # s=3: (B12 - B11) → (0,1):1, (0,0):-1
        {(0, 1): 1, (0, 0): -1},
        # s=4: B22 → (1,1):1
        {(1, 1): 1},
        # s=5: (B11 + B12) → (0,0):1, (0,1):1
        {(0, 0): 1, (0, 1): 1},
        # s=6: (B21 + B22) → (1,0):1, (1,1):1
        {(1, 0): 1, (1, 1): 1},
    ]

    # W_coeffs[s]: dict mapping (i, k_loc) -> coefficient
    # C11 = M1 + M4 - M5 + M7  → (i=0,k=0): +1 for s in {0,3}, -1 for s=4, +1 for s=6
    # C12 = M3 + M5            → (i=0,k=1): +1 for s in {2,4}
    # C21 = M2 + M4            → (i=1,k=0): +1 for s in {1,3}
    # C22 = M1 - M2 + M3 + M6  → (i=1,k=1): +1 for s in {0,2,5}, -1 for s=1
    W_coeffs = [
        # s=0: contributes to C11(+1) and C22(+1)
        {(0, 0): 1, (1, 1): 1},
        # s=1: contributes to C21(+1) and C22(-1)
        {(1, 0): 1, (1, 1): -1},
        # s=2: contributes to C12(+1) and C22(+1)
        {(0, 1): 1, (1, 1): 1},
        # s=3: contributes to C11(+1) and C21(+1)
        {(0, 0): 1, (1, 0): 1},
        # s=4: contributes to C11(-1) and C12(+1)
        {(0, 0): -1, (0, 1): 1},
        # s=5: contributes to C22(+1)
        {(1, 1): 1},
        # s=6: contributes to C11(+1)
        {(0, 0): 1},
    ]

    num_a_blocks = m // 2  # 2
    num_b_blocks = p // 2  # 3
    rank = num_a_blocks * num_b_blocks * 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for a in range(num_a_blocks):
        for b in range(num_b_blocks):
            for s in range(7):
                # U: row index = i * m + j, where j = a*2 + j_loc
                for (i, j_loc), coeff in U_coeffs[s].items():
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff

                # V: row index = j * p + k, where j = a*2 + j_loc, k = b*2 + k_loc
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff

                # W: row index = k * n + i, where k = b*2 + k_loc
                for (i, k_loc), coeff in W_coeffs[s].items():
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff

                col += 1

    # Deadline guard: if we've exceeded 120s, return immediately (shouldn't happen
    # for this static construction, but per directive)
    if time.time() - start_time > 120:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
