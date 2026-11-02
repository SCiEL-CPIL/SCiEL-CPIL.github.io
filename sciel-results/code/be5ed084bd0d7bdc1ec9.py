"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import time

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen with 7 factors, integer entries."""
    # U rows: A_11, A_12, A_21, A_22  (4 rows)
    # V rows: B_11, B_12, B_21, B_22  (4 rows)
    # W rows: C_11, C_12, C_21, C_22  (4 rows)
    # 7 factors
    # M1 = A11*(B11+B22): U: A11=1,A22=1 ; V: B11=1,B22=1 ; W: C11=1,C22=1
    # M2 = (A21+A22)*B11: U: A21=1,A22=1 ; V: B11=1 ; W: C21=1
    # M3 = A11*(B12-B22): U: A11=1 ; V: B12=1,B22=-1 ; W: C12=1,C22=-1
    # M4 = (A22-A11)*(B21+B11): U: A22=1,A11=-1 ; V: B21=1,B11=1 ; W: C11=-1,C21=1
    # M5 = (A11+A12)*B22: U: A11=1,A12=1 ; V: B22=1 ; W: C12=1
    # M6 = (A21-A11)*(B11+B12): U: A21=1,A11=-1 ; V: B11=1,B12=1 ; W: C11=-1,C22=1
    # M7 = (A12-A22)*(B21+B22): U: A12=1,A22=-1 ; V: B21=1,B22=1 ; W: C11=1,C22=-1

    U = [
        [1, 0, 1, -1, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],     # A12
        [0, 1, 0, 0, 0, 1, 0],     # A21
        [1, 1, 0, 1, 0, 0, -1],    # A22
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],     # B11
        [0, 0, 1, 0, 0, 1, 0],     # B12
        [0, 0, 0, 1, 0, 0, 1],     # B21
        [1, 0, -1, 0, 1, 0, 1],    # B22
    ]
    W = [
        [1, 0, 0, -1, 0, -1, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],     # C12
        [0, 1, 0, 1, 0, 0, 0],     # C21
        [1, 0, -1, 0, 0, 1, -1],   # C22
    ]
    return U, V, W


def _build_block_decomposition(n, m, p, block_m):
    """
    Block decomposition: split m into (m/block_m) blocks of size block_m.
    Each block uses Strassen 2x2x2 to multiply (n x block_m) by (block_m x p).
    This gives rank = (m/block_m) * 7.
    """
    n_blocks = m // block_m
    rank = n_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    sU, sV, sW = _build_strassen_2x2x2()

    for b in range(n_blocks):
        offset = b * 7
        # For each Strassen factor, assign to the appropriate rows/cols
        for c in range(7):
            # U rows: for each i in range(n), j in range(block_m)
            for i in range(n):
                for j in range(block_m):
                    row_u = i * m + (b * block_m + j)
                    U[row_u][offset + c] = sU[j][c]
            # V rows: for each j in range(block_m), k in range(p)
            for j in range(block_m):
                for k in range(p):
                    row_v = (b * block_m + j) * p + k
                    V[row_v][offset + c] = sV[j][c]
            # W rows: for each k in range(p), i in range(n)
            for k in range(p):
                for i in range(n):
                    row_w = k * n + i
                    W[row_w][offset + c] = sW[i][c]

    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110  # safety margin for 120s limit

    # Try block decomposition with block_m=2 (Strassen)
    U, V, W, rank = _build_block_decomposition(n, m, p, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
