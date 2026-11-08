"""Matrix multiplication 4x4 * 4x9: find rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    a = U[ui][r]
                    if a != 0:
                        b = V[vj][r]
                        if b != 0:
                            s += a * b * W[wk][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x2(U, V, W, n, m, p, i0, j0, k0, rank_offset):
    """
    Add Strassen 2x2*2x2 decomposition for blocks starting at (i0, j0, k0).
    Returns new rank offset.
    """
    A11 = i0 * m + j0
    A12 = i0 * m + j0 + 1
    A21 = (i0 + 1) * m + j0
    A22 = (i0 + 1) * m + j0 + 1
    
    B11 = j0 * p + k0
    B12 = j0 * p + k0 + 1
    B21 = (j0 + 1) * p + k0
    B22 = (j0 + 1) * p + k0 + 1
    
    C11 = k0 * n + i0
    C12 = k0 * n + i0 + 1
    C21 = (k0 + 1) * n + i0
    C22 = (k0 + 1) * n + i0 + 1
    
    r = rank_offset
    
    # M1 = (A11+A22)*(B11+B22)
    U[A11][r] += 1; U[A22][r] += 1
    V[B11][r] += 1; V[B22][r] += 1
    W[C11][r] += 1; W[C22][r] += 1
    r += 1
    
    # M2 = (A21+A22)*B11
    U[A21][r] += 1; U[A22][r] += 1
    V[B11][r] += 1
    W[C21][r] += 1; W[C22][r] -= 1
    r += 1
    
    # M3 = A11*(B12-B22)
    U[A11][r] += 1
    V[B12][r] += 1; V[B22][r] -= 1
    W[C12][r] += 1; W[C22][r] += 1
    r += 1
    
    # M4 = A22*(B21-B11)
    U[A22][r] += 1
    V[B21][r] += 1; V[B11][r] -= 1
    W[C11][r] += 1; W[C21][r] += 1
    r += 1
    
    # M5 = (A11+A12)*B22
    U[A11][r] += 1; U[A12][r] += 1
    V[B22][r] += 1
    W[C11][r] -= 1; W[C12][r] += 1
    r += 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[A21][r] += 1; U[A11][r] -= 1
    V[B11][r] += 1; V[B12][r] += 1
    W[C22][r] += 1
    r += 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[A12][r] += 1; U[A22][r] -= 1
    V[B21][r] += 1; V[B22][r] += 1
    W[C11][r] += 1
    r += 1
    
    return r

def _build_strassen_4x4x8(n, m, p):
    """
    Build Strassen decomposition for 4x4 * 4x8 part (first 8 columns of B).
    2 i-blocks * 2 j-blocks * 4 k-blocks = 16 Strassen 2x2x2 blocks, rank 16*7 = 112.
    """
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for I in range(2):
        for J in range(2):
            for K in range(4):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook_4x4x1(n, m, p):
    """
    Build schoolbook decomposition for 4x4 * 4x1 part (last column of B, k=8).
    Rank = 4*4*1 = 16
    """
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            k = 8
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_split_4x4x9_split1(n, m, p):
    """
    Split k dimension: 4x4*4x4 (Strassen 2x2x2 blocks) + 4x4*4x5
    For 4x4*4x4: 4 Strassen blocks, rank 28
    For 4x4*4x5: schoolbook, rank 80
    Total: 108
    """
    rank = 28 + 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # First 4 columns of B: 2 i-blocks * 2 j-blocks * 2 k-blocks
    for I in range(2):
        for J in range(2):
            for K in range(2):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    # Last 5 columns of B: schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, 9):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _build_split_4x4x9_split2(n, m, p):
    """
    Split k dimension: 4x4*4x6 + 4x4*4x3
    For 4x4*4x6: 2 i-blocks * 2 j-blocks * 3 k-blocks Strassen, rank 42
    For 4x4*4x3: schoolbook, rank 48
    Total: 90
    """
    rank = 42 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # First 6 columns of B: 2 i-blocks * 2 j-blocks * 3 k-blocks
    for I in range(2):
        for J in range(2):
            for K in range(3):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    # Last 3 columns of B: schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(6, 9):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _build_split_4x4x9_split3(n, m, p):
    """
    Split k dimension: 4x4*4x7 + 4x4*4x2
    For 4x4*4x7: 2 i-blocks * 2 j-blocks * 3 k-blocks Strassen + 1 k-block schoolbook
    Actually, let's do 3 full Strassen blocks (6 cols) + 1 col schoolbook for first 7 cols
    Rank for first 7: 42 + 16 = 58
    For 4x4*4x2: schoolbook, rank 32
    Total: 90
    """
    rank = 58 + 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # First 6 columns: Strassen
    for I in range(2):
        for J in range(2):
            for K in range(3):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    # Column 6 (7th col): schoolbook
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Last 2 columns (cols 7,8): schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(7, 9):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _build_split_4x4x9_split4(n, m, p):
    """
    Split k dimension: 4x4*4x8 + 4x4*4x1
    For 4x4*4x8: 16 Strassen blocks, rank 112
    For 4x4*4x1: schoolbook, rank 16
    Total: 128 (same as original hybrid)
    """
    rank = 112 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for I in range(2):
        for J in range(2):
            for K in range(4):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    for i in range(n):
        for j in range(m):
            k = 8
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decs, n, m, p):
    """
    Combine multiple decompositions by concatenating their rank dimensions.
    """
    total_rank = sum(len(d["U"][0]) for d in decs)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for d in decs:
        r = len(d["U"][0])
        for i in range(n * m):
            for c in range(r):
                U[i][offset + c] = d["U"][i][c]
        for i in range(m * p):
            for c in range(r):
                V[i][offset + c] = d["V"][i][c]
        for i in range(p * n):
            for c in range(r):
                W[i][offset + c] = d["W"][i][c]
        offset += r
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns the tensor decomposition for matrix multiplication of dimensions 4x4 * 4x9.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try various split strategies
    candidates = [
        ("split2_6_3", _build_split_4x4x9_split2),
        ("split3_7_2", _build_split_4x4x9_split3),
        ("split1_4_5", _build_split_4x4x9_split1),
        ("split4_8_1", _build_split_4x4x9_split4),
    ]
    
    for name, builder in candidates:
        if not _time_ok(deadline):
            break
        try:
            result = builder(n, m, p)
            r = len(result["U"][0])
            if r < best_rank:
                if _validate(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END
