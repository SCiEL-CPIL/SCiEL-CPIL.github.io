"""Tensor decomposition for 2x4 * 4x6 matrix multiplication, rank 42 (Strassen-based)."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6

    # Strassen 2x2x2 factor matrices (local, 4 rows x 7 cols)
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0],  # X00
        [0, 0, 0, 0, 1, 0, 1],   # X01
        [0, 1, 0, 0, 0, 1, 0],   # X10
        [1, 1, 0, 1, 0, 0, -1],  # X11
    ]
    V_loc = [
        [1, 1, -1, 0, 0, 1, 0],  # Y00
        [0, 0, 1, 0, 0, 0, 1],   # Y01
        [0, 0, 0, 1, 0, 1, 0],   # Y10
        [1, 0, -1, 0, 1, 0, 1],  # Y11
    ]
    W_loc = [
        [1, 0, 0, 1, -1, 0, 0],  # P00
        [0, 0, 1, 0, 0, 0, 0],   # P01
        [0, 1, 0, 0, 0, 0, 0],   # P10
        [1, 0, 0, 0, 0, 1, 1],   # P11
    ]

    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # For each column-block j_c in {0,1,2} and sub s in {0 (A1,B1j), 1 (A2,B2j)}
    for j_c in range(3):
        for s in range(2):
            for t in range(7):
                col = j_c * 14 + s * 7 + t
                # U: local (i,a) -> global
                for lu in range(4):
                    val = U_loc[lu][t]
                    if val != 0:
                        i = lu // 2
                        a = lu % 2
                        if s == 0:
                            grow = i * m + a        # A1: cols 0,1
                        else:
                            grow = i * m + (a + 2)  # A2: cols 2,3
                        U[grow][col] += val
                # V: local (b,c) -> global
                for lv in range(4):
                    val = V_loc[lv][t]
                    if val != 0:
                        b = lv // 2
                        c = lv % 2
                        if s == 0:
                            grow = b * p + (2 * j_c + c)      # B1j: rows 0,1
                        else:
                            grow = (b + 2) * p + (2 * j_c + c) # B2j: rows 2,3
                        V[grow][col] += val
                # W: local (i,c) -> global, C block cols 2*j_c, 2*j_c+1
                for lw in range(4):
                    val = W_loc[lw][t]
                    if val != 0:
                        i = lw // 2
                        c = lw % 2
                        grow = (2 * j_c + c) * n + i
                        W[grow][col] += val

    # Deadline guard: construction is O(1); return immediately.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
