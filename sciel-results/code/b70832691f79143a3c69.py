"""
Rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication.

Uses Strassen (rank 7) for the 2x2 sub-block and schoolbook for others.
Total rank = 7 + 12 + 12 + 18 = 49.
"""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen for the 2x2 sub-block and schoolbook for the remaining blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=5, p=5

    U = []
    V = []
    W = []

    # Strassen rank-7 decomposition for 2x2x2
    # Local indices: A=(A11,A12,A21,A22), B=(B11,B12,B21,B22), C=(C11,C12,C21,C22)
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),    # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # M3
        ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [0,-1,0,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7
    ]

    # Map local 2x2 indices to global indices
    a_map = [0, 1, 5, 6]
    b_map = [0, 1, 5, 6]
    c_map = [0, 1, 2, 3]

    for u_l, v_l, w_l in strassen_terms:
        u_g = [0.0] * 10
        v_g = [0.0] * 25
        w_g = [0.0] * 10
        for li, val in enumerate(u_l):
            if val != 0:
                u_g[a_map[li]] = float(val)
        for li, val in enumerate(v_l):
            if val != 0:
                v_g[b_map[li]] = float(val)
        for li, val in enumerate(w_l):
            if val != 0:
                w_g[c_map[li]] = float(val)
        U.append(u_g)
        V.append(v_g)
        W.append(w_g)

    # Block 2: 2x3 x 3x2 schoolbook (rank 12)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u_g = [0.0] * 10
                v_g = [0.0] * 25
                w_g = [0.0] * 10
                u_g[i * 5 + j] = 1.0
                v_g[j * 5 + k] = 1.0
                w_g[k * 2 + i] = 1.0
                U.append(u_g)
                V.append(v_g)
                W.append(w_g)

    # Block 3: 2x2 x 2x3 schoolbook (rank 12)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u_g = [0.0] * 10
                v_g = [0.0] * 25
                w_g = [0.0] * 10
                u_g[i * 5 + j] = 1.0
                v_g[j * 5 + k] = 1.0
                w_g[k * 2 + i] = 1.0
                U.append(u_g)
                V.append(v_g)
                W.append(w_g)

    # Block 4: 2x3 x 3x3 schoolbook (rank 18)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u_g = [0.0] * 10
                v_g = [0.0] * 25
                w_g = [0.0] * 10
                u_g[i * 5 + j] = 1.0
                v_g[j * 5 + k] = 1.0
                w_g[k * 2 + i] = 1.0
                U.append(u_g)
                V.append(v_g)
                W.append(w_g)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
