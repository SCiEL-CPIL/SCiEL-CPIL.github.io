"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def _build_4x4x3_block(i_offset, j_offset, k_offset, base_col, U, V, W):
    """
    Build one 4x4x3 sub-tensor (schoolbook) into U,V,W starting at base_col.
    U indexed by (i*4+j), V by (j*3+k), W by (k*4+i) -- but we embed into the
    full 4x4x9 layout. For the full layout:
        U row  = i*m + j   (m=4)
        V row  = j*p + k   (p=9)
        W row  = k*n + i   (n=4)
    For a column-block b (0..2), the 3 columns are k = 3*b .. 3*b+2.
    """
    n, m, p = 4, 4, 9
    b = k_offset  # which 3-column block (0,1,2)
    col = base_col
    for i in range(n):
        for j in range(m):
            for kk in range(3):
                k = 3 * b + kk
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return col

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_c U[u,c]*V[v,c]*W[w,c] equals the matmul tensor.
    T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i.
    Returns True if exact (within tol).
    """
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # For each (i,j,k) check the triple product sum equals 1; for others 0.
    # Efficient: for each c, the outer structure. Check all (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = 0.0
                # s = sum_c U[u,c] V[v,c] W[w,c]
                uc = Ua[u]
                vc = Va[v]
                wc = Wa[w]
                s = float(np.dot(uc * vc, wc))
                if abs(s - 1.0) > tol:
                    return False
    # Spot-check a few off-diagonal triples must be 0.
    checks = [
        (0 * m + 0, 0 * p + 1, 1 * n + 0),  # i=0,j=0,k=1 -> w=1*4+0=4, but u=0,v=1
        (1 * m + 1, 0 * p + 0, 0 * n + 0),
        (2 * m + 3, 3 * p + 8, 8 * n + 2),
    ]
    for (u, v, w) in checks:
        uc = Ua[u]; vc = Va[v]; wc = Wa[w]
        s = float(np.dot(uc * vc, wc))
        if abs(s) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s (and 150s kill)
    n, m, p = DIMENSIONS  # 4, 4, 9

    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Structured Kronecker column-block decomposition:
    # 4x4x9 = sum over 3 column-blocks b of an independent 4x4x3 sub-tensor.
    base = 0
    for b in range(3):
        base = _build_4x4x3_block(0, 0, b, base, U, V, W)
    assert base == rank

    # Verification pass (cheap: 4*4*9 = 144 triple products of length 144).
    if time.time() < deadline:
        ok = _verify(U, V, W, n, m, p)
        if not ok:
            # Fallback: rebuild via direct schoolbook to guarantee correctness.
            for col in range(rank):
                k = col % p
                tmp = col // p
                j = tmp % m
                i = tmp // m
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
