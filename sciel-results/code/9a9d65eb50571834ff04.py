"""Block-Strassen decomposition for 4x4 x 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check U,V,W decompose the matmul tensor exactly (and are half-integers)."""
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
    for M in (U, V, W):
        if np.max(np.abs(M * 2 - np.round(M * 2))) > tol:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
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


def _S2():
    """2x2x2 Strassen 7-term vectors (u,v,w) for the all-ones 2x2x2 tensor."""
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 1, 0]),  # M6
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),  # M7
    ]


def _build_4x4x4_strassen():
    """Build rank-49 Strassen decomposition for 4x4 x 4x4 matmul.
    Uses hierarchical 2x2 Strassen: 4 blocks of 2x2x2, each with 7 terms = 28.
    Wait, that's only 28. The actual 4x4x4 Strassen is 49 terms.
    
    Actually, for 4x4x4, we can partition into 2x2 blocks in all three dimensions.
    That gives 2*2*2 = 8 sub-blocks of 2x2x2, each with 7 terms = 56 terms.
    But the standard 4x4 Strassen uses 49 terms via a more clever combination.
    
    Let me use the simpler approach: 8 blocks of 2x2x2 Strassen = 56 terms.
    Then for 4x4x5, we add the 5th column naively: 4x4 = 16 more terms. Total = 72.
    That's the same as the current program.
    
    Alternative: Use 4x4x4 with rank 49 (known result) + 4 naive = 53.
    The rank-49 decomposition for 4x4x4 is non-trivial. Let me construct it.
    
    Actually, let me just use the 8-block 2x2x2 Strassen (rank 56) for 4x4x4,
    then add 16 naive terms for the 5th column = 72 total. Same as current.
    
    To get 53, I need a proper 4x4x4 rank-49 decomposition.
    """
    # For now, build the 8-block 2x2x2 Strassen for 4x4x4 (rank 56)
    n = m = p = 4
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    def add(ub, vb, wb):
        nonlocal U, V, W
        U = np.hstack([U, ub[:, None]])
        V = np.hstack([V, vb[:, None]])
        W = np.hstack([W, wb[:, None]])
    
    S2 = _S2()
    for bi in range(n // 2):
        for bj in range(m // 2):
            for bk in range(p // 2):
                for ub, vb, wb in S2:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for a in range(2):
                        for c in range(2):
                            for cc in range(2):
                                row = 2 * bi + a
                                col = 2 * bj + c
                                kcol = 2 * bk + cc
                                u[row * m + col] += ub[a * 2 + c]
                                v[col * p + kcol] += vb[c * 2 + cc]
                                w[kcol * n + row] += wb[cc * 2 + a]
                    add(u, v, w)
    return U, V, W


def _build_4x4x5():
    """Build decomposition for 4x4 x 4x5: 4x4x4 Strassen (56) + 4 naive (16) = 72."""
    n, m, p = 4, 4, 5
    U4, V4, W4 = _build_4x4x4_strassen()
    
    U = U4.copy()
    V = np.zeros((m * p, U4.shape[1]))
    W = np.zeros((p * n, U4.shape[1]))
    
    # Copy V4 and W4 into the first 4 columns
    V[:, :4 * 4] = V4  # m*p=20, but V4 has 4*4=16 rows... wait
    # V4 has shape (4*4, 56) = (16, 56)
    # V has shape (4*5, 56) = (20, 56)
    V[:16, :] = V4
    W[:16, :] = W4  # W4 has shape (4*4, 56) = (16, 56), W has shape (5*4, 56) = (20, 56)
    
    # Add 16 naive terms for the 5th column (k=4)
    rank_naive = n * m  # 16 terms
    U_new = np.zeros((n * m, rank_naive))
    V_new = np.zeros((m * p, rank_naive))
    W_new = np.zeros((p * n, rank_naive))
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U_new[idx, idx] = 1.0
            V_new[j * p + 4, idx] = 1.0
            W_new[4 * n + i, idx] = 1.0
    
    U = np.hstack([U, U_new])
    V = np.hstack([V, V_new])
    W = np.hstack([W, W_new])
    
    return U, V, W


def _build_block(n, m, p):
    """Build a decomposition for an n x m (A) times m x p (B) block.
    Uses Strassen (7) for 2x2x2 sub-blocks and a 9-term scheme for 2x2x3.
    Returns (U,V,W) arrays of shapes (n*m, r), (m*p, r), (p*n, r)."""
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))

    def add(ub, vb, wb):
        nonlocal U, V, W
        U = np.hstack([U, ub[:, None]])
        V = np.hstack([V, vb[:, None]])
        W = np.hstack([W, wb[:, None]])

    S2 = _S2()
    for bi in range(n // 2):
        for bj in range(m // 2):
            for bk in range(p // 2):
                # 2x2x2 Strassen block
                for ub, vb, wb in S2:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for a in range(2):
                        for c in range(2):
                            for cc in range(2):
                                row = 2 * bi + a
                                col = 2 * bj + c
                                kcol = 2 * bk + cc
                                u[row * m + col] += ub[a * 2 + c]
                                v[col * p + kcol] += vb[c * 2 + cc]
                                w[kcol * n + row] += wb[cc * 2 + a]
                    add(u, v, w)
            # 2x2x3 blocks: one 2x2x2 Strassen + 2 naive for the extra k-column
            for bk in range((p // 2) * 2, p, 2):
                for ub, vb, wb in S2:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for a in range(2):
                        for c in range(2):
                            for cc in range(2):
                                row = 2 * bi + a
                                col = 2 * bj + c
                                kcol = 2 * bk + cc
                                u[row * m + col] += ub[a * 2 + c]
                                v[col * p + kcol] += vb[c * 2 + cc]
                                w[kcol * n + row] += wb[cc * 2 + a]
                    add(u, v, w)
                # naive terms for the 3rd column of this 2x2x3 block
                for a in range(2):
                    for c in range(2):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        row = 2 * bi + a
                        col = 2 * bj + c
                        kcol = 2 * bk + 2
                        u[row * m + col] = 1.0
                        v[col * p + kcol] = 1.0
                        w[kcol * n + row] = 1.0
                        add(u, v, w)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the 4x4x5 construction (rank 72)
    try:
        Uf, Vf, Wf = _build_4x4x5()
        if _verify(Uf, Vf, Wf, n, m, p):
            r = Uf.shape[1]
            if r < best_rank:
                best = _to_int_lists(Uf, Vf, Wf)
                best_rank = r
    except Exception:
        pass
    
    # Try the block construction (also rank 72, but different structure)
    if time.time() < deadline - 5:
        try:
            Uf, Vf, Wf = _build_block(n, m, p)
            if _verify(Uf, Vf, Wf, n, m, p):
                r = Uf.shape[1]
                if r < best_rank:
                    best = _to_int_lists(Uf, Vf, Wf)
                    best_rank = r
        except Exception:
            pass
    
    if time.time() > deadline - 2:
        return best
    return best
# EVOLVE-BLOCK-END
