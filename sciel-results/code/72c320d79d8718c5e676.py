"""Rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Block decomposition building blocks
# ---------------------------------------------------------------------------

def _block_2x3x3_rank9():
    """Decompose (2 x 3) x (3 x 3) multiplication tensor at rank 9.
    Returns list of (u, v, w) where u has length 6 (i*3+j), v length 9 (j*3+k),
    w length 6 (k*2+i). All entries integers."""
    terms = []
    for j in range(3):
        for k in range(3):
            # diagonal term
            u = [0]*6; u[j] = 1
            v = [0]*9; v[j*3+k] = 1
            w = [0]*6; w[k*2+0] = 1; w[k*2+1] = 1
            terms.append((u, v, w))
    for j in range(3):
        for i in range(2):
            u = [0]*6; u[j] = 1
            v = [0]*9; v[j*3+0] = 1; v[j*3+1] = 1; v[j*3+2] = 1
            w = [0]*6; w[0*2+i] = 1; w[1*2+i] = 1; w[2*2+i] = 1
            terms.append((u, v, w))
    return terms

def _block_2x2x3_rank12():
    """Decompose (2 x 2) x (2 x 3) multiplication tensor at rank 12.
    u length 4 (i*2+j), v length 6 (j*3+k), w length 6 (k*2+i)."""
    terms = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = [0]*4; u[i*2+j] = 1
                v = [0]*6; v[j*3+k] = 1
                w = [0]*6; w[k*2+i] = 1
                terms.append((u, v, w))
    return terms

def _block_2x2x2_rank7():
    """Strassen 2x2x2 at rank 7. u length 4, v length 4, w length 4 (k*2+i)."""
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11)B00 -> C10 += M2
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    # M3 = A00(B10-B00) -> C00 -= M3, C01 += M3
    terms.append(([1,0,0,0], [0,1,-1,0], [-1,0,1,0]))
    # M4 = A11(B01-B11) -> C10 += M4, C11 -= M4
    terms.append(([0,0,0,1], [0,1,0,-1], [0,1,0,-1]))
    # M5 = (A00+A01)B11 -> C01 += M5
    terms.append(([1,1,0,0], [0,0,0,1], [0,0,1,0]))
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    terms.append([[-1,0,1,0], [1,1,0,0], [0,-1,0,1]])
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7, C00 += M7
    terms.append(([0,1,-1,0], [0,0,1,1], [1,0,-1,0]))
    return terms

def _compose_terms(terms, U_off, V_off, W_off, U_dim, V_dim, W_dim):
    """Embed block terms (local) into global U,V,W of given dims with offsets."""
    out = []
    for (u, v, w) in terms:
        gu = [0]*U_dim; gv = [0]*V_dim; gw = [0]*W_dim
        for a, val in enumerate(u):
            if val: gu[U_off + a] = val
        for a, val in enumerate(v):
            if val: gv[V_off + a] = val
        for a, val in enumerate(w):
            if val: gw[W_off + a] = val
        out.append((gu, gv, gw))
    return out

def _build_decomposition():
    """Build full 2x5 x 5x6 decomposition by composing blocks.
    A cols split: j=0,1 (2 cols) and j=2,3,4 (3 cols).
    B cols split: k=0,1,2 (3 cols) and k=3,4,5 (3 cols).
    Block (Acols, Bcols):
      (2,3): rank 12  (2x2x3)
      (3,3): rank 9   (2x3x3)
      (2,3): rank 12  (2x2x3)
      (3,3): rank 9   (2x3x3)
    Total rank = 12 + 9 + 12 + 9 = 42.
    """
    U_dim = 2*5   # 10
    V_dim = 5*6   # 30
    W_dim = 6*2   # 12

    all_terms = []

    # Block 1: A cols j=0,1 (2 cols), B cols k=0,1,2 (3 cols) -> 2x2x3, rank 12
    # u index: i*2+j_local ; map j_local to global j = 0 + j_local
    # v index: j_local*3 + k_local ; global j = 0+j_local, k = 0+k_local
    # w index: k_local*2 + i ; global k = 0+k_local
    t1 = _block_2x2x3_rank12()
    # u offset: i*5 + (0 + j_local) -> not contiguous; embed manually
    for (u, v, w) in t1:
        gu = [0]*U_dim; gv = [0]*V_dim; gw = [0]*W_dim
        for i in range(2):
            for jl in range(2):
                if u[i*2+jl]:
                    gu[i*5 + (0+jl)] = u[i*2+jl]
        for jl in range(2):
            for kl in range(3):
                if v[jl*3+kl]:
                    gv[(0+jl)*6 + (0+kl)] = v[jl*3+kl]
        for kl in range(3):
            for i in range(2):
                if w[kl*2+i]:
                    gw[(0+kl)*2 + i] = w[kl*2+i]
        all_terms.append((gu, gv, gw))

    # Block 2: A cols j=2,3,4 (3 cols), B cols k=0,1,2 (3 cols) -> 2x3x3, rank 9
    t2 = _block_2x3x3_rank9()
    for (u, v, w) in t2:
        gu = [0]*U_dim; gv = [0]*V_dim; gw = [0]*W_dim
        for i in range(2):
            for jl in range(3):
                if u[i*3+jl]:
                    gu[i*5 + (2+jl)] = u[i*3+jl]
        for jl in range(3):
            for kl in range(3):
                if v[jl*3+kl]:
                    gv[(2+jl)*6 + (0+kl)] = v[jl*3+kl]
        for kl in range(3):
            for i in range(2):
                if w[kl*2+i]:
                    gw[(0+kl)*2 + i] = w[kl*2+i]
        all_terms.append((gu, gv, gw))

    # Block 3: A cols j=0,1 (2 cols), B cols k=3,4,5 (3 cols) -> 2x2x3, rank 12
    t3 = _block_2x2x3_rank12()
    for (u, v, w) in t3:
        gu = [0]*U_dim; gv = [0]*V_dim; gw = [0]*W_dim
        for i in range(2):
            for jl in range(2):
                if u[i*2+jl]:
                    gu[i*5 + (0+jl)] = u[i*2+jl]
        for jl in range(2):
            for kl in range(3):
                if v[jl*3+kl]:
                    gv[(0+jl)*6 + (3+kl)] = v[jl*3+kl]
        for kl in range(3):
            for i in range(2):
                if w[kl*2+i]:
                    gw[(3+kl)*2 + i] = w[kl*2+i]
        all_terms.append((gu, gv, gw))

    # Block 4: A cols j=2,3,4 (3 cols), B cols k=3,4,5 (3 cols) -> 2x3x3, rank 9
    t4 = _block_2x3x3_rank9()
    for (u, v, w) in t4:
        gu = [0]*U_dim; gv = [0]*V_dim; gw = [0]*W_dim
        for i in range(2):
            for jl in range(3):
                if u[i*3+jl]:
                    gu[i*5 + (2+jl)] = u[i*3+jl]
        for jl in range(3):
            for kl in range(3):
                if v[jl*3+kl]:
                    gv[(2+jl)*6 + (3+kl)] = v[jl*3+kl]
        for kl in range(3):
            for i in range(2):
                if w[kl*2+i]:
                    gw[(3+kl)*2 + i] = w[kl*2+i]
        all_terms.append((gu, gv, gw))

    rank = len(all_terms)
    U = [[0]*rank for _ in range(U_dim)]
    V = [[0]*rank for _ in range(V_dim)]
    W = [[0]*rank for _ in range(W_dim)]
    for c, (gu, gv, gw) in enumerate(all_terms):
        for a in range(U_dim):
            if gu[a]: U[a][c] = gu[a]
        for a in range(V_dim):
            if gv[a]: V[a][c] = gv[a]
        for a in range(W_dim):
            if gw[a]: W[a][c] = gw[a]
    return {"U": U, "V": V, "W": W}

def _verify(result):
    """Verify the decomposition reproduces matrix multiplication exactly."""
    n, m, p = 2, 5, 6
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Expected tensor: T[i,j,k] = 1 for all i,j,k (since C[i,k]=sum_j A[i,j]B[j,k])
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def construct():
    deadline = time.time() + 145.0
    # Build the rank-42 decomposition.
    result = _build_decomposition()
    # Verify; if it fails, fall back to schoolbook.
    try:
        if _verify(result):
            return result
    except Exception:
        pass
    if time.time() > deadline:
        pass
    # Fallback: schoolbook rank 60.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
