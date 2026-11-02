"""Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _make_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] == multiplication tensor."""
    rank = len(U[0])
    total = np.zeros((n, m, p), dtype=np.float64)
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for c in range(rank):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.max(np.abs(total - expected)) < tol


def _strassen_222():
    """
    Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Strassen's algorithm:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Indexing:
    U[i*m+j] where m=2: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
    V[j*p+k] where p=2: V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
    W[k*n+i] where n=2: W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1               # B00
    W[1][1] = 1               # C10
    
    # M3: A00(B10-B11)
    U[0][2] = 1               # A00
    V[2][2] = 1; V[3][2] = -1 # B10 - B11
    W[2][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4: A11(B01-B00)
    U[3][3] = 1               # A11
    V[0][3] = -1; V[1][3] = 1 # B01 - B00
    W[1][3] = 1; W[2][3] = 1  # C10, C01
    
    # M5: (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1  # A00, A10
    V[3][4] = 1               # B11
    W[0][4] = -1; W[2][4] = 1 # -C00, C01
    
    # M6: (A11-A10)(B00+B10)
    U[2][5] = -1; U[3][5] = 1 # -A10, A11
    V[0][5] = 1; V[2][5] = 1  # B00, B10
    W[3][5] = 1               # C11
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1 # A01, -A11
    V[1][6] = 1; V[3][6] = 1  # B01, B11
    W[0][6] = 1               # C00
    
    return {"U": U, "V": V, "W": W, "rank": 7}


def _schoolbook_122():
    """Schoolbook rank-4 decomposition for 1x2x2 matrix multiplication."""
    n, m, p = 1, 2, 2
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _embed_222_into_242(block_decomp, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, full_U, full_V, full_W, col_start):
    """
    Embed a 2x2x2 decomposition into a larger 2x4x2 decomposition.
    
    a_row_offset: row offset in A (2x4 matrix)
    a_col_offset: col offset in A (2x4 matrix)
    b_row_offset: row offset in B (4x2 matrix)
    b_col_offset: col offset in B (4x2 matrix)
    c_row_offset: row offset in C (2x2 matrix)
    c_col_offset: col offset in C (2x2 matrix)
    """
    n_sub, m_sub, p_sub = 2, 2, 2
    for c in range(block_decomp["rank"]):
        col = col_start + c
        # U: A[a_row_offset + i, a_col_offset + j]
        for i in range(n_sub):
            for j in range(m_sub):
                val = block_decomp["U"][i * m_sub + j][c]
                full_U[(a_row_offset + i) * 4 + (a_col_offset + j)][col] = val
        # V: B[b_row_offset + j, b_col_offset + k]
        for j in range(m_sub):
            for k in range(p_sub):
                val = block_decomp["V"][j * p_sub + k][c]
                full_V[(b_row_offset + j) * 2 + (b_col_offset + k)][col] = val
        # W: C[c_row_offset + i, c_col_offset + k]
        for i in range(n_sub):
            for k in range(p_sub):
                val = block_decomp["W"][k * n_sub + i][c]
                full_W[(c_col_offset + k) * 2 + (c_row_offset + i)][col] = val


def _embed_122_into_242(block_decomp, a_row, a_col_offset, b_row_offset, b_col_offset, c_row, c_col_offset, full_U, full_V, full_W, col_start):
    """
    Embed a 1x2x2 decomposition into a larger 2x4x2 decomposition.
    
    a_row: row in A (2x4 matrix)
    a_col_offset: col offset in A (2x4 matrix)
    b_row_offset: row offset in B (4x2 matrix)
    b_col_offset: col offset in B (4x2 matrix)
    c_row: row in C (2x2 matrix)
    c_col_offset: col offset in C (2x2 matrix)
    """
    n_sub, m_sub, p_sub = 1, 2, 2
    for c in range(block_decomp["rank"]):
        col = col_start + c
        # U: A[a_row, a_col_offset + j]
        for j in range(m_sub):
            val = block_decomp["U"][j][c]
            full_U[a_row * 4 + (a_col_offset + j)][col] = val
        # V: B[b_row_offset + j, b_col_offset + k]
        for j in range(m_sub):
            for k in range(p_sub):
                val = block_decomp["V"][j * p_sub + k][c]
                full_V[(b_row_offset + j) * 2 + (b_col_offset + k)][col] = val
        # W: C[c_row, c_col_offset + k]
        for k in range(p_sub):
            val = block_decomp["W"][k][c]
            full_W[(c_col_offset + k) * 2 + c_row][col] = val


def _build_242_strassen():
    """
    Build a 2x4x2 decomposition using Strassen for 2x2x2 sub-blocks.
    
    A is 2x4, B is 4x2, C is 2x2.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split into two 2x2x2 blocks:
    Block 1: A[0:2, 0:2] @ B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    Block 2: A[0:2, 2:4] @ B[2:4, 0:2] -> contributes to C[0:2, 0:2]
    
    Each uses Strassen rank 7, total rank 14.
    """
    n, m, p = 2, 4, 2
    s1 = _strassen_222()
    s2 = _strassen_222()
    total_rank = s1["rank"] + s2["rank"]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: A[0:2, 0:2] @ B[0:2, 0:2]
    _embed_222_into_242(s1, 0, 0, 0, 0, 0, 0, U, V, W, 0)
    
    # Block 2: A[0:2, 2:4] @ B[2:4, 0:2]
    _embed_222_into_242(s2, 0, 2, 2, 0, 0, 0, U, V, W, s1["rank"])
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}


def _build_142_strassen():
    """
    Build a 1x4x2 decomposition using schoolbook for 1x2x2 sub-blocks.
    
    A is 1x4, B is 4x2, C is 1x2.
    C[0,k] = sum_j A[0,j] * B[j,k]
    
    Split into two 1x2x2 blocks:
    Block 1: A[0, 0:2] @ B[0:2, 0:2] -> contributes to C[0, 0:2]
    Block 2: A[0, 2:4] @ B[2:4, 0:2] -> contributes to C[0, 0:2]
    
    Each uses schoolbook rank 4, total rank 8.
    """
    n, m, p = 1, 4, 2
    s1 = _schoolbook_122()
    s2 = _schoolbook_122()
    total_rank = s1["rank"] + s2["rank"]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: A[0, 0:2] @ B[0:2, 0:2]
    _embed_122_into_242(s1, 0, 0, 0, 0, 0, 0, U, V, W, 0)
    
    # Block 2: A[0, 2:4] @ B[2:4, 0:2]
    _embed_122_into_242(s2, 0, 2, 2, 0, 0, 0, U, V, W, s1["rank"])
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}


def _embed_242_into_346(block_decomp, a_row_offset, b_col_offset, c_row_offset, c_col_offset, full_U, full_V, full_W, col_start):
    """
    Embed a 2x4x2 decomposition into the full 3x4x6 decomposition.
    
    a_row_offset: row offset in A (3x4 matrix)
    b_col_offset: col offset in B (4x6 matrix)
    c_row_offset: row offset in C (3x6 matrix)
    c_col_offset: col offset in C (3x6 matrix)
    """
    n_sub, m_sub, p_sub = 2, 4, 2
    for c in range(block_decomp["rank"]):
        col = col_start + c
        # U: A[a_row_offset + i, j]
        for i in range(n_sub):
            for j in range(m_sub):
                val = block_decomp["U"][i * m_sub + j][c]
                full_U[(a_row_offset + i) * 4 + j][col] = val
        # V: B[j, b_col_offset + k]
        for j in range(m_sub):
            for k in range(p_sub):
                val = block_decomp["V"][j * p_sub + k][c]
                full_V[j * 6 + (b_col_offset + k)][col] = val
        # W: C[c_row_offset + i, c_col_offset + k]
        for i in range(n_sub):
            for k in range(p_sub):
                val = block_decomp["W"][k * n_sub + i][c]
                full_W[(c_col_offset + k) * 3 + (c_row_offset + i)][col] = val


def _embed_142_into_346(block_decomp, a_row, b_col_offset, c_row, c_col_offset, full_U, full_V, full_W, col_start):
    """
    Embed a 1x4x2 decomposition into the full 3x4x6 decomposition.
    
    a_row: row in A (3x4 matrix)
    b_col_offset: col offset in B (4x6 matrix)
    c_row: row in C (3x6 matrix)
    c_col_offset: col offset in C (3x6 matrix)
    """
    n_sub, m_sub, p_sub = 1, 4, 2
    for c in range(block_decomp["rank"]):
        col = col_start + c
        # U: A[a_row, j]
        for j in range(m_sub):
            val = block_decomp["U"][j][c]
            full_U[a_row * 4 + j][col] = val
        # V: B[j, b_col_offset + k]
        for j in range(m_sub):
            for k in range(p_sub):
                val = block_decomp["V"][j * p_sub + k][c]
                full_V[j * 6 + (b_col_offset + k)][col] = val
        # W: C[c_row, c_col_offset + k]
        for k in range(p_sub):
            val = block_decomp["W"][k][c]
            full_W[(c_col_offset + k) * 3 + c_row][col] = val


def _build_346_hybrid():
    """
    Build the full 3x4x6 decomposition using hybrid Strassen/schoolbook approach.
    
    Partition:
    - A (3x4) into: A_top = A[0:2, :] (2x4), A_bot = A[2, :] (1x4)
    - B (4x6) into: B_left = B[:, 0:2], B_mid = B[:, 2:4], B_right = B[:, 4:6]
    
    Sub-problems:
    1. C[0:2, 0:2] = A_top @ B_left  (2x4x2, Strassen rank 14)
    2. C[0:2, 2:4] = A_top @ B_mid   (2x4x2, Strassen rank 14)
    3. C[0:2, 4:6] = A_top @ B_right (2x4x2, Strassen rank 14)
    4. C[2, 0:2]   = A_bot @ B_left  (1x4x2, schoolbook rank 8)
    5. C[2, 2:4]   = A_bot @ B_mid   (1x4x2, schoolbook rank 8)
    6. C[2, 4:6]   = A_bot @ B_right (1x4x2, schoolbook rank 8)
    
    Total rank: 3*14 + 3*8 = 42 + 24 = 66
    """
    n, m, p = 3, 4, 6
    
    # Build sub-decompositions
    d242_1 = _build_242_strassen()
    d242_2 = _build_242_strassen()
    d242_3 = _build_242_strassen()
    d142_1 = _build_142_strassen()
    d142_2 = _build_142_strassen()
    d142_3 = _build_142_strassen()
    
    total_rank = (d242_1["rank"] + d242_2["rank"] + d242_3["rank"] +
                  d142_1["rank"] + d142_2["rank"] + d142_3["rank"])
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # 1. C[0:2, 0:2] = A[0:2, :] @ B[:, 0:2]
    _embed_242_into_346(d242_1, 0, 0, 0, 0, U, V, W, col)
    col += d242_1["rank"]
    
    # 2. C[0:2, 2:4] = A[0:2, :] @ B[:, 2:4]
    _embed_242_into_346(d242_2, 0, 2, 0, 2, U, V, W, col)
    col += d242_2["rank"]
    
    # 3. C[0:2, 4:6] = A[0:2, :] @ B[:, 4:6]
    _embed_242_into_346(d242_3, 0, 4, 0, 4, U, V, W, col)
    col += d242_3["rank"]
    
    # 4. C[2, 0:2] = A[2, :] @ B[:, 0:2]
    _embed_142_into_346(d142_1, 2, 0, 2, 0, U, V, W, col)
    col += d142_1["rank"]
    
    # 5. C[2, 2:4] = A[2, :] @ B[:, 2:4]
    _embed_142_into_346(d142_2, 2, 2, 2, 2, U, V, W, col)
    col += d142_2["rank"]
    
    # 6. C[2, 4:6] = A[2, :] @ B[:, 4:6]
    _embed_142_into_346(d142_3, 2, 4, 2, 4, U, V, W, col)
    col += d142_3["rank"]
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s deadline, 20s safety margin
    
    # Start with schoolbook as fallback
    best_result = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid Strassen/schoolbook decomposition
    try:
        hybrid = _build_346_hybrid()
        if _validate(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            if hybrid["rank"] < best_rank:
                best_result = hybrid
                best_rank = hybrid["rank"]
    except Exception:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
