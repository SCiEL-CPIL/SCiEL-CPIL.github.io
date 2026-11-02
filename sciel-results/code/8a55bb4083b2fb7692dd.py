"""Combined program: structured Strassen construction + verification machinery."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W decompose the matmul tensor exactly (and are half-integers)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # half-integer check
    for M in (U, V, W):
        if np.max(np.abs(M * 2 - np.round(M * 2))) > tol:
            return False
    # tensor check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k]) * 1.0
                val = np.einsum('c,c,c->', U[i * m + j], V[j * p + k], W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Rank n*m*p decomposition (always valid)."""
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


def _to_int_lists(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    def conv(M):
        M = np.round(M * 2) / 2.0
        out = []
        for row in M:
            out.append([float(x) for x in row])
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _strassen_matrix_block(n, m, p):
    """Build a low-rank decomposition of the full (n*m) x (m*p) x (n*p) tensor
    by linear-combining the schoolbook basis terms with half-integer coefficients.
    Returns (U,V,W) numpy arrays or None if not exactly valid."""
    n, m, p = int(n), int(m), int(p)
    rank = n * m * p
    # schoolbook basis term index t for (i,j,k)
    def tidx(i, j, k):
        return (i * m + j) * p + k
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = tidx(i, j, k)
                U[i * m + j, t] = 1.0
                V[j * p + k, t] = 1.0
                W[k * n + i, t] = 1.0
    return U, V, W


def _try_strassen_4x4x4():
    """Construct the 2-level Strassen rank-49 decomposition for 4x4 x 4x4.
    Uses the standard 2x2 Strassen 7-term formulas applied recursively to 2x2 blocks.
    Returns (U,V,W) arrays or None."""
    n = m = p = 4
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = (i * m + j) * p + k
                U[i * m + j, t] = 1.0
                V[j * p + k, t] = 1.0
                W[k * n + i, t] = 1.0

    # 2x2 Strassen 7-term formulas.
    # Each term is a triple (u, v, w) of length-4 vectors over the 4 entries of a 2x2 block,
    # indexed in row-major: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    # The 7 terms (u,v,w) such that sum u[i]*v[j]*w[k] reproduces the 2x2x2 matmul tensor.
    # Standard Strassen (with A=[[a,b],[c,d]], B=[[e,f],[g,h]]):
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)(e)
    #   M3 = (a)(f-h)
    #   M4 = (d)(g-c)
    #   M5 = (a+b)(h)
    #   M6 = (c-a)(e+g)
    #   M7 = (b-d)(f+h)
    # The rank-1 tensor terms (u,v,w) for 2x2x2 that sum to the all-ones tensor:
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 1, 0]),  # M6
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),  # M7
    ]
    # Verify the 2x2 Strassen terms reproduce the 2x2x2 all-ones tensor.
    T2 = np.zeros((2, 2, 2))
    for u, v, w in S2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    T2[a, b, c] += u[a * 2 + b] * v[b * 2 + c] * w[c * 2 + a]
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        # formulas wrong; fall back to None
        return None

    # Now build 4x4x4 by 2-level block Strassen.
    # Level 1: 4x4 split into 2x2 blocks of 2x2 sub-blocks.
    # We apply the 2x2 Strassen at the "block" level (7 terms), and each block term
    # itself is a 2x2 Strassen (7 terms) -> 49 total rank-1 terms.
    # Represent each rank-1 term as (u4, v4, w4) length-16 vectors (row-major over 4x4).
    # A block-level term (ub, vb, wb) where ub,vb,wb are length-4 over the 4 blocks.
    # Each block is 2x2; within a block we use the 2x2 Strassen (7 terms).
    # So the full rank-1 term = sum over the 7 inner terms of (ub[b]*u_inner, vb[b]*v_inner, wb[b]*w_inner).

    terms = []
    for ub, vb, wb in S2:
        # ub, vb, wb are length-4 (over blocks). For each block b with nonzero coeff,
        # add 7 inner terms scaled.
        for b in range(4):
            if ub[b] == 0 or vb[b] == 0 or wb[b] == 0:
                continue
            scale = ub[b] * vb[b] * wb[b]
            for ui, vi, wi in S2:
                u4 = np.zeros(16)
                v4 = np.zeros(16)
                w4 = np.zeros(16)
                # block b occupies rows b//2, cols b%2 of the 2x2 block grid
                # In 4x4 row-major, block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
                br, bc = b // 2, b % 2
                for a in range(2):
                    for c2 in range(2):
                        for cc in range(2):
                            # inner index a*2+c2 within block maps to global row 2*br+a, col 2*bc+c2
                            row = 2 * br + a
                            col = 2 * bc + c2
                            u4[row * 4 + col] += scale * ui[a * 2 + c2]
                            # v: inner (c2, cc) -> global row 2*bc+c2, col 2*cc (B is 4x4)
                            v4[(2 * bc + c2) * 4 + 2 * cc] += scale * vi[c2 * 2 + cc]
                            # w: inner (cc, a) -> global row 2*cc, col 2*br+a (C is 4x4)
                            w4[(2 * cc) * 4 + 2 * br + a] += scale * wi[cc * 2 + a]
                terms.append((u4, v4, w4))

    r = len(terms)
    Uf = np.zeros((16, r))
    Vf = np.zeros((16, r))
    Wf = np.zeros((16, r))
    for c, (u4, v4, w4) in enumerate(terms):
        Uf[:, c] = u4
        Vf[:, c] = v4
        Wf[:, c] = w4
    return Uf, Vf, Wf


def _try_strassen_4x4x5():
    """Build 4x4 x 4x5 decomposition: Strassen 4x4x4 (rank 49) + 4 extra terms for the 5th column.
    Returns (U,V,W) arrays or None."""
    U44, V44, W44 = _try_strassen_4x4x4()
    if U44 is None:
        return None
    # Extend to p=5. The 4x4x4 decomposition handles k in 0..3.
    # For k=4 (the 5th column of B / 5th column of C), we need the standard 4x4 x 4x1 = 16 terms.
    # But we can reuse: the 4x4x4 part already covers k=0..3. We add terms for k=4.
    # For k=4: C[i,4] = sum_j A[i,j] B[j,4]. Schoolbook gives 16 terms.
    # So total rank = 49 + 16 = 65. That's worse than we want but valid.
    # Actually we want lower. Let's just do 49 + 16 = 65 first (valid), then try to merge.
    n, m = 4, 4
    p4 = 4
    p = 5
    r4 = U44.shape[1]
    Uf = np.zeros((n * m, r4 + n * m))
    Vf = np.zeros((m * p, r4 + n * m))
    Wf = np.zeros((p * n, r4 + n * m))
    # copy 4x4x4 part (k=0..3)
    Uf[:, :r4] = U44
    # V: 4x4x4 V is (m*p4)=16 rows; we need (m*p)=20 rows. Copy first 16, last 4 for k=4.
    Vf[:m * p4, :r4] = V44
    Wf[:p4 * n, :r4] = W44
    # add k=4 terms
    for i in range(n):
        for j in range(m):
            c = r4 + i * m + j
            Uf[i * m + j, c] = 1.0
            Vf[j * p + 4, c] = 1.0
            Wf[4 * n + i, c] = 1.0
    return Uf, Vf, Wf


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    # Fallback: schoolbook (rank 80, always valid).
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    candidates = []
    # 1) 2-level Strassen 4x4x4 + 16 extra = rank 65
    if time.time() < deadline - 5:
        try:
            cand = _try_strassen_4x4x5()
            if cand is not None:
                candidates.append(cand)
        except Exception:
            pass

    # Verify each candidate and keep the best valid one.
    for Uf, Vf, Wf in candidates:
        if time.time() > deadline - 2:
            break
        try:
            if _verify(Uf, Vf, Wf, n, m, p):
                r = Uf.shape[1]
                if r < best_rank:
                    best = _to_int_lists(Uf, Vf, Wf)
                    best_rank = r
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
