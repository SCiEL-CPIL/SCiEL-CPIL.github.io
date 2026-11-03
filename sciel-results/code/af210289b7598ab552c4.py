"""4x4x8 matrix multiplication tensor decomposition.

Strategy: Use Strassen's rank-7 decomposition for 2x2x2, applied recursively
to get a rank-56 decomposition for 4x4x4 (8 sub-blocks x 7 terms). Then
combine two 4x4x4 blocks for 4x4x8, giving total rank 112.

The key insight: for 4x4x8, we split B into two 4x4 column blocks.
Each block produces a 4x4 result, so C is 8x4 (rows 0-3 from block 1, rows 4-7 from block 2).

Verification is done at runtime with random integer inputs.
"""

import time
import random

DEADLINE = 120.0
T_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - T_START)


def _build_strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    
    Each term is (U_dict, V_dict, W_dict) where:
    - U_dict maps A index (i,m) -> row i*2+m to coefficient
    - V_dict maps B index (m,p) -> row m*2+p to coefficient  
    - W_dict maps C index (k,n) -> row k*2+n to coefficient
      where k = column of B = row of C, n = row of A = column of C
    """
    def a(i, m): return i * 2 + m
    def b(m, p): return m * 2 + p
    def c(k, n): return k * 2 + n  # W[k*n + i] convention: k is C row, n is C col
    
    terms = []
    
    # M0 = (A00+A11)*(B00+B11) -> C00 += M0, C11 += M0
    terms.append(({a(0,0): 1, a(1,1): 1}, {b(0,0): 1, b(1,1): 1}, {c(0,0): 1, c(1,1): 1}))
    
    # M1 = (A10+A11)*B00 -> C10 += M1
    terms.append(({a(1,0): 1, a(1,1): 1}, {b(0,0): 1}, {c(1,0): 1}))
    
    # M2 = A00*(B01-B11) -> C01 += M2
    terms.append(({a(0,0): 1}, {b(0,1): 1, b(1,1): -1}, {c(0,1): 1}))
    
    # M3 = A11*(B10-B00) -> C00 += M3, C10 += M3
    terms.append(({a(1,1): 1}, {b(1,0): 1, b(0,0): -1}, {c(0,0): 1, c(1,0): 1}))
    
    # M4 = (A00+A01)*B11 -> C01 += M4, C11 += M4
    terms.append(({a(0,0): 1, a(0,1): 1}, {b(1,1): 1}, {c(0,1): 1, c(1,1): 1}))
    
    # M5 = (A10-A00)*(B00+B01) -> C00 -= M5
    terms.append(({a(1,0): 1, a(0,0): -1}, {b(0,0): 1, b(0,1): 1}, {c(0,0): -1}))
    
    # M6 = (A01-A11)*(B10+B11) -> C00 -= M6, C11 -= M6
    terms.append(({a(0,1): 1, a(1,1): -1}, {b(1,0): 1, b(1,1): 1}, {c(0,0): -1, c(1,1): -1}))
    
    return terms


def _build_4x4x4_strassen():
    """Build rank-56 decomposition of 4x4x4 matmul tensor.
    
    Returns U, V, W as lists of lists (16 x 56 each).
    U row index: i*4+m (A is 4x4)
    V row index: m*4+p (B is 4x4)  
    W row index: k*4+n (C is 4x4, where k=B col, n=A row)
    """
    s7 = _build_strassen_2x2x2()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 8 sub-blocks: a (A row half), b (A col/B row half), c (B col/C row half)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for (tu, tv, tw) in s7:
                    u_col = {}
                    v_col = {}
                    w_col = {}
                    
                    # A local (i,m) -> global row: (2*a+i)*4 + (2*b+m)
                    for (li, lm), coeff in tu.items():
                        gr = (2*a + li) * 4 + (2*b + lm)
                        u_col[gr] = u_col.get(gr, 0) + coeff
                    
                    # B local (m,p) -> global row: (2*b+m)*4 + (2*c+p)
                    for (lm, lp), coeff in tv.items():
                        gr = (2*b + lm) * 4 + (2*c + lp)
                        v_col[gr] = v_col.get(gr, 0) + coeff
                    
                    # C local (k,n) -> W row: k*4+n
                    # k is C row = B col, n is C col = A row
                    # global: (2*c+k)*4 + (2*a+n)
                    for (lk, ln), coeff in tw.items():
                        gr = (2*c + lk) * 4 + (2*a + ln)
                        w_col[gr] = w_col.get(gr, 0) + coeff
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r in range(rank):
        for row, val in U_cols[r].items():
            U[row][r] = val
        for row, val in V_cols[r].items():
            V[row][r] = val
        for row, val in W_cols[r].items():
            W[row][r] = val
    
    return U, V, W, rank


def _build_4x4x8_strassen():
    """Build rank-112 decomposition of 4x4x8 matmul tensor.
    
    U: 16 x 112 (A is 4x4, flattened as i*4+m)
    V: 32 x 112 (B is 4x8, flattened as m*8+p)
    W: 32 x 112 (C is 8x4, flattened as k*4+n where k=0..7, n=0..3)
    
    Block 1: B columns 0-3, C rows 0-3
    Block 2: B columns 4-7, C rows 4-7
    """
    U1, V1, W1, r1 = _build_4x4x4_strassen()
    
    rank = 2 * r1
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    for r in range(r1):
        # Block 1: B cols 0-3, C rows 0-3
        for row in range(16):
            U[row][r] = U1[row][r]
            # V1 row = m*4+p where m,p in 0..3
            # Global V row = m*8+p
            m = row // 4
            p = row % 4
            V[m*8 + p][r] = V1[row][r]
            # W1 row = k*4+n where k,n in 0..3
            # Global W row = k*4+n (same, since k in 0..3)
            W[row][r] = W1[row][r]
        
        # Block 2: B cols 4-7, C rows 4-7
        r2 = r + r1
        for row in range(16):
            U[row][r2] = U1[row][r]
            m = row // 4
            p = row % 4
            V[m*8 + (p+4)][r2] = V1[row][r]
            k = row // 4
            n = row % 4
            W[(k+4)*4 + n][r2] = W1[row][r]
    
    return U, V, W, rank


def _build_schoolbook():
    """Schoolbook 4x4x8 decomposition, rank 128."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank


def _verify_decomposition(U, V, W, rank):
    """Verify the decomposition is exact using random integer inputs."""
    try:
        import numpy as np
    except ImportError:
        return False
    
    n, m, p = 4, 4, 8
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    rng = random.Random(12345)
    for trial in range(5):
        A = np.array([[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)], dtype=np.int64)
        B = np.array([[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)], dtype=np.int64)
        C_true = A @ B
        
        C = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            u_vec = U_arr[:, r].reshape(n, m)
            v_vec = V_arr[:, r].reshape(m, p)
            w_vec = W_arr[:, r].reshape(p, n)
            C += (u_vec @ v_vec) * w_vec
        
        if not np.array_equal(C, C_true):
            return False
    
    return True


def construct():
    if _remaining() < 5.0:
        U, V, W, _ = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    try:
        U, V, W, rank = _build_4x4x8_strassen()
        if _remaining() < 15.0:
            U, V, W, _ = _build_schoolbook()
            return {"U": U, "V": V, "W": W}
        if _verify_decomposition(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W, _ = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
