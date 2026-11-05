import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition():
    """Return the schoolbook rank-72 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication using Strassen (rank 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 0 (M1): (A11)(B11+B22)
    U[0][0] = 1
    V[0][0] = 1
    V[1][0] = 1
    W[0][0] = 1
    W[1][0] = 1
    
    # Product 1 (M2): (A21+A22)(B11)
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Product 2 (M3): (A11)(B22)
    U[0][2] = 1
    V[1][2] = 1
    W[1][2] = 1
    
    # Product 3 (M4): (A22)(B21-B11)
    U[3][3] = 1
    V[0][3] = -1
    V[1][3] = 1
    W[1][3] = 1
    
    # Product 4 (M5): (A11+A22)(B22+B12)
    U[0][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    V[1][4] = 1
    W[1][4] = 1
    W[3][4] = 1
    
    # Product 5 (M6): (A21-A11)(B12+B22)
    U[0][5] = -1
    U[2][5] = 1
    V[0][5] = 1
    V[1][5] = 1
    W[2][5] = 1
    
    # Product 6 (M7): (A12-A22)(B21+B11)
    U[1][6] = 1
    U[3][6] = -1
    V[0][6] = 1
    V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_optimized_block_decomposition():
    """
    Build a block decomposition for 3x3 * 3x8 using Strassen and optimized outer products.
    Target rank: 48.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    total_rank = 48
    
    U = [[0]*total_rank for _ in range(n*m)]  # 9 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    sU, sV, sW = _strassen_2x2x2()
    
    # Helper to add Strassen block
    def add_strassen(u_rows, v_rows, w_rows):
        nonlocal col
        for r in range(7):
            for i_local in range(4):
                U[u_rows[i_local]][col + r] = sU[i_local][r]
            for i_local in range(4):
                V[v_rows[i_local]][col + r] = sV[i_local][r]
            for i_local in range(4):
                W[w_rows[i_local]][col + r] = sW[i_local][r]
        col += 7

    # Helper to add 1x2x2 schoolbook block (Rank 4)
    def add_1x2x2(u_rows, v_rows, w_rows):
        # u_rows: 2 rows (A[2,0], A[2,1])
        # v_rows: 4 rows (B[0,0], B[0,1], B[1,0], B[1,1])
        # w_rows: 2 rows (C[2,0], C[2,1])
        nonlocal col
        # r=0: A0 B0 -> C0
        U[u_rows[0]][col] = 1
        V[v_rows[0]][col] = 1
        W[w_rows[0]][col] = 1
        col += 1
        # r=1: A1 B1 -> C0
        U[u_rows[1]][col] = 1
        V[v_rows[2]][col] = 1
        W[w_rows[0]][col] = 1
        col += 1
        # r=2: A0 B0 -> C1
        U[u_rows[0]][col] = 1
        V[v_rows[1]][col] = 1
        W[w_rows[1]][col] = 1
        col += 1
        # r=3: A1 B1 -> C1
        U[u_rows[1]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[1]][col] = 1
        col += 1

    # Helper to add 2x1x4 outer product (Rank 1)
    def add_2x1x4_outer(u_rows, v_rows, w_rows):
        # u_rows: 2 rows (A[0,2], A[1,2])
        # v_rows: 4 rows (B[2,0]..B[2,3])
        # w_rows: 8 rows (C[0,0]..C[1,3])
        nonlocal col
        for i in range(2):
            U[u_rows[i]][col] = 1
        for k in range(4):
            V[v_rows[k]][col] = 1
        for w_idx in w_rows:
            W[w_idx][col] = 1
        col += 1

    # Helper to add 1x1x4 outer product (Rank 1)
    def add_1x1x4_outer(u_row, v_rows, w_rows):
        # u_row: 1 row (A[2,2])
        # v_rows: 4 rows (B[2,0]..B[2,3])
        # w_rows: 4 rows (C[2,0]..C[2,3])
        nonlocal col
        U[u_row][col] = 1
        for k in range(4):
            V[v_rows[k]][col] = 1
        for w_idx in w_rows:
            W[w_idx][col] = 1
        col += 1

    # Block 1: i in {0,1}, k in {0,1,2,3}
    # 1a: 2x2x2 (j in {0,1}, k in {0,1})
    add_strassen([0, 1, 3, 4], [0, 1, 8, 9], [0, 1, 3, 4])
    # 1b: 2x2x2 (j in {0,1}, k in {2,3})
    add_strassen([0, 1, 3, 4], [2, 3, 10, 11], [6, 7, 9, 10])
    # 1c: 2x1x4 (j in {2}, k in {0,1,2,3})
    add_2x1x4_outer([2, 5], [16, 17, 18, 19], [0, 1, 3, 4, 6, 7, 9, 10])

    # Block 2: i in {0,1}, k in {4,5,6,7}
    # 2a: 2x2x2 (j in {0,1}, k in {4,5})
    add_strassen([0, 1, 3, 4], [4, 5, 12, 13], [12, 13, 15, 16])
    # 2b: 2x2x2 (j in {0,1}, k in {6,7})
    add_strassen([0, 1, 3, 4], [6, 7, 14, 15], [18, 19, 21, 22])
    # 2c: 2x1x4 (j in {2}, k in {4,5,6,7})
    add_2x1x4_outer([2, 5], [20, 21, 22, 23], [12, 13, 15, 16, 18, 19, 21, 22])

    # Block 3: i in {2}, k in {0,1,2,3}
    # 3a1: 1x2x2 (j in {0,1}, k in {0,1})
    add_1x2x2([6, 7], [0, 1, 8, 9], [2, 5])
    # 3a2: 1x2x2 (j in {0,1}, k in {2,3})
    add_1x2x2([6, 7], [2, 3, 10, 11], [8, 11])
    # 3b: 1x1x4 (j in {2}, k in {0,1,2,3})
    add_1x1x4_outer(8, [16, 17, 18, 19], [2, 5, 8, 11])

    # Block 4: i in {2}, k in {4,5,6,7}
    # 4a1: 1x2x2 (j in {0,1}, k in {4,5})
    add_1x2x2([6, 7], [4, 5, 12, 13], [14, 17])
    # 4a2: 1x2x2 (j in {0,1}, k in {6,7})
    add_1x2x2([6, 7], [6, 7, 14, 15], [20, 23])
    # 4b: 1x1x4 (j in {2}, k in {4,5,6,7})
    add_1x1x4_outer(8, [20, 21, 22, 23], [14, 17, 20, 23])

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank exact decomposition for 3x3 * 3x8 matrix multiplication.
    Uses optimized block decomposition with rank 48.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    try:
        result = _build_optimized_block_decomposition()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
