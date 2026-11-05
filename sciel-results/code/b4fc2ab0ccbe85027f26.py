"""Matrix multiplication 3x4x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _build_rank36():
    """
    Build a rank-36 decomposition using:
    - Split j as (3+1): 3x3x4 + 3x1x4
    - 3x3x4: split k as (2+2): two 3x3x2 blocks
    - Each 3x3x2: split j as (1+1+1): three 3x1x2 blocks (schoolbook, rank 4 each)
    - 3x1x4: schoolbook, rank 12
    Total: 2*3*4 + 12 = 24 + 12 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 12 x rank
    V = [[0]*rank for _ in range(m*p)]  # 16 x rank
    W = [[0]*rank for _ in range(p*n)]  # 12 x rank
    col = 0

    def add(u, v, w):
        nonlocal col
        for (ij, c) in u.items():
            U[ij][col] += c
        for (jk, c) in v.items():
            V[jk][col] += c
        for (ik, c) in w.items():
            W[ik][col] += c
        col += 1

    def ui(i, j): return i*m + j
    def vi(j, k): return j*p + k
    def wi(i, k): return k*n + i  # W[k*n+i]

    # 3x1x2 schoolbook: A is 3x1, B is 1x2, C is 3x2
    # C[i,k] = A[i,0] * B[0,k]
    # Terms: (i=0,1,2) x (k=0,1) = 6 terms
    def schoolbook_3x1x2(j, k0, k1):
        """3x1x2: A rows 0,1,2 col j; B row j cols k0,k1; C rows 0,1,2 cols k0,k1"""
        for i in range(3):
            for k in (k0, k1):
                add({ui(i,j):1}, {vi(j,k):1}, {wi(i,k):1})

    # 3x3x2: split j as (1+1+1), each 3x1x2
    def block_3x3x2(j0, j1, j2, k0, k1):
        """3x3x2: A rows 0,1,2 cols j0,j1,j2; B rows j0,j1,j2 cols k0,k1; C rows 0,1,2 cols k0,k1"""
        for j in (j0, j1, j2):
            schoolbook_3x1x2(j, k0, k1)

    # 3x1x4 schoolbook: A is 3x1 (col j), B is 1x4 (row j, cols 0,1,2,3), C is 3x4
    # Terms: (i=0,1,2) x (k=0,1,2,3) = 12 terms
    def schoolbook_3x1x4(j):
        for i in range(3):
            for k in range(4):
                add({ui(i,j):1}, {vi(j,k):1}, {wi(i,k):1})

    # 3x3x4: split k as (2+2): two 3x3x2 blocks
    # Block 1: j in {0,1,2}, k in {0,1}
    block_3x3x2(0, 1, 2, 0, 1)  # 3*4 = 12 terms
    # Block 2: j in {0,1,2}, k in {2,3}
    block_3x3x2(0, 1, 2, 2, 3)  # 3*4 = 12 terms

    # 3x1x4: j=3, k in {0,1,2,3}
    schoolbook_3x1x4(3)  # 3*4 = 12 terms

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _build_rank44():
    """Fallback: rank-44 decomposition using Strassen."""
    n, m, p = DIMENSIONS  # 3,4,4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x rank
    V = [[0]*rank for _ in range(m*p)]  # 16 x rank
    W = [[0]*rank for _ in range(p*n)]  # 12 x rank
    col = 0

    def add(u, v, w):
        nonlocal col
        for (ij, c) in u.items():
            U[ij][col] += c
        for (jk, c) in v.items():
            V[jk][col] += c
        for (ik, c) in w.items():
            W[ik][col] += c
        col += 1

    def ui(i, j): return i*m + j
    def vi(j, k): return j*p + k
    def wi(i, k): return k*n + i  # W[k*n+i]

    def strassen_block(i0, i1, j0, j1, k0, k1):
        def Uv(a00, a01, a10, a11):
            return {ui(i0,j0): a00, ui(i0,j1): a01, ui(i1,j0): a10, ui(i1,j1): a11}
        def Vv(b00, b01, b10, b11):
            return {vi(j0,k0): b00, vi(j0,k1): b01, vi(j1,k0): b10, vi(j1,k1): b11}
        def Wv(c00, c01, c10, c11):
            return {wi(i0,k0): c00, wi(i0,k1): c01, wi(i1,k0): c10, wi(i1,k1): c11}
        terms = [
            (Uv(1,0,0,1), Vv(1,0,0,1), Wv(1,0,0,1)),
            (Uv(0,0,1,1), Vv(1,0,0,0), Wv(0,0,1,-1)),
            (Uv(1,0,0,0), Vv(0,1,0,-1), Wv(0,1,0,1)),
            (Uv(0,0,0,1), Vv(-1,0,1,0), Wv(1,0,1,0)),
            (Uv(1,1,0,0), Vv(0,0,0,1), Wv(-1,1,0,0)),
            (Uv(-1,0,1,0), Vv(1,1,0,0), Wv(0,0,0,1)),
            (Uv(0,1,0,-1), Vv(0,0,1,1), Wv(1,0,0,0)),
        ]
        for u, v, w in terms:
            add(u, v, w)

    def schoolbook_1x2x2(i, j0, j1, k0, k1):
        for j in (j0, j1):
            for k in (k0, k1):
                add({ui(i,j):1}, {vi(j,k):1}, {wi(i,k):1})

    for (j0, j1) in [(0,1), (2,3)]:
        for (k0, k1) in [(0,1), (2,3)]:
            strassen_block(0, 1, j0, j1, k0, k1)
            schoolbook_1x2x2(2, j0, j1, k0, k1)
    assert col == rank, col
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for im_j in range(n*m):
        i, j = divmod(im_j, m)
        for jp_k in range(m*p):
            j2, k = divmod(jp_k, p)
            if j2 != j:
                val = float(np.einsum('c,c,c->', Ua[im_j], Va[jp_k], Wa[k*n+i]))
                if abs(val) > 1e-9:
                    return False
                continue
            kn_i = k*n + i
            val = float(np.einsum('c,c,c->', Ua[im_j], Va[jp_k], Wa[kn_i]))
            if abs(val - 1) > 1e-9:
                return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = t0 + 120
    
    # Try rank-36 first
    try:
        cand = _build_rank36()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return _schoolbook()
    
    # Try rank-44
    try:
        cand = _build_rank44()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return _schoolbook()
    
    return _schoolbook()
# EVOLVE-BLOCK-END
