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
        return None

    # Build 4x4x4 by 2-level block Strassen.
    terms = []
    for ub, vb, wb in S2:
        for b in range(4):
            if ub[b] == 0 or vb[b] == 0 or wb[b] == 0:
                continue
            scale = ub[b] * vb[b] * wb[b]
            for ui, vi, wi in S2:
                u4 = np.zeros(16)
                v4 = np.zeros(16)
                w4 = np.zeros(16)
                br, bc = b // 2, b % 2
                for a in range(2):
                    for c2 in range(2):
                        for cc in range(2):
                            row = 2 * br + a
                            col = 2 * bc + c2
                            u4[row * 4 + col] += scale * ui[a * 2 + c2]
                            v4[(2 * bc + c2) * 4 + 2 * cc] += scale * vi[c2 * 2 + cc]
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
    n, m = 4, 4
    p4 = 4
    p = 5
    r4 = U44.shape[1]
    Uf = np.zeros((n * m, r4 + n * m))
    Vf = np.zeros((m * p, r4 + n * m))
    Wf = np.zeros((p * n, r4 + n * m))
    Uf[:, :r4] = U44
    Vf[:m * p4, :r4] = V44
    Wf[:p4 * n, :r4] = W44
    for i in range(n):
        for j in range(m):
            c = r4 + i * m + j
            Uf[i * m + j, c] = 1.0
            Vf[j * p + 4, c] = 1.0
            Wf[4 * n + i, c] = 1.0
    return Uf, Vf, Wf


def _try_strassen_2x2x5():
    """Build a 2x2 x 2x5 decomposition using Strassen for the 2x2 part and schoolbook for the extra column.
    Returns (U,V,W) arrays or None."""
    n, m = 2, 2
    p = 5
    # 2x2 Strassen terms
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
    ]
    # Verify 2x2 part
    T2 = np.zeros((2, 2, 2))
    for u, v, w in S2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    T2[a, b, c] += u[a * 2 + b] * v[b * 2 + c] * w[c * 2 + a]
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        return None

    # Build 2x2 x 2x5: Strassen for k=0,1 (7 terms) + schoolbook for k=2,3,4 (6 terms)
    # Total rank = 7 + 6 = 13
    r = 13
    Uf = np.zeros((n * m, r))
    Vf = np.zeros((m * p, r))
    Wf = np.zeros((p * n, r))

    # Add Strassen terms for k=0,1
    for c, (u, v, w) in enumerate(S2):
        for a in range(2):
            for b in range(2):
                Uf[a * 2 + b, c] = u[a * 2 + b]
        for b in range(2):
            for k in range(2):
                Vf[b * 5 + k, c] = v[b * 2 + k]
        for k in range(2):
            for a in range(2):
                Wf[k * 2 + a, c] = w[k * 2 + a]

    # Add schoolbook terms for k=2,3,4
    for i in range(n):
        for j in range(m):
            for k in range(2, 5):
                c = 7 + (i * m + j) * 3 + (k - 2)
                Uf[i * m + j, c] = 1.0
                Vf[j * p + k, c] = 1.0
                Wf[k * n + i, c] = 1.0

    return Uf, Vf, Wf


def _try_strassen_4x4x5_block():
    """Build 4x4 x 4x5 by blocking: split into 2x2 blocks and use Strassen 2x2x5 for each block column.
    This gives rank = 4 * 13 = 52 (4 block columns, each with 13 terms).
    Returns (U,V,W) arrays or None."""
    U2, V2, W2 = _try_strassen_2x2x5()
    if U2 is None:
        return None

    n, m, p = 4, 4, 5
    r_block = U2.shape[1]  # 13
    r_total = 4 * r_block  # 52

    Uf = np.zeros((n * m, r_total))
    Vf = np.zeros((m * p, r_total))
    Wf = np.zeros((p * n, r_total))

    # For each block column (bc in 0..1), use the 2x2x5 Strassen decomposition
    # Block (br, bc) covers rows 2*br..2*br+1 of A, cols 2*bc..2*bc+1 of B
    for bc in range(2):
        for br in range(2):
            c_offset = (bc * 2 + br) * r_block
            for c in range(r_block):
                for a in range(2):
                    for b in range(2):
                        # U: A block (br, bc) entry (a, b) -> global (2*br+a, 2*bc+b)
                        Uf[(2 * br + a) * 4 + (2 * bc + b), c_offset + c] = U2[a * 2 + b, c]
                for b in range(2):
                    for k in range(5):
                        # V: B block (bc, *) entry (b, k) -> global (2*bc+b, k)
                        Vf[(2 * bc + b) * 5 + k, c_offset + c] = V2[b * 5 + k, c]
                for k in range(5):
                    for a in range(2):
                        # W: C block (*, br) entry (k, a) -> global (k, 2*br+a)
                        Wf[k * 4 + (2 * br + a), c_offset + c] = W2[k * 2 + a, c]

    return Uf, Vf, Wf


def _try_strassen_4x4x5_mixed():
    """Build 4x4 x 4x5 by using Strassen 4x4x4 for k=0..3 and Strassen 2x2x1 for k=4.
    For k=4, split into 2x2 blocks: rank = 2 * 7 = 14 (two 2x2 Strassen blocks for the 4x1 column).
    Total rank = 49 + 14 = 63.
    Returns (U,V,W) arrays or None."""
    U44, V44, W44 = _try_strassen_4x4x4()
    if U44 is None:
        return None

    n, m, p = 4, 4, 5
    r4 = U44.shape[1]  # 49
    r_k4 = 14  # 2 blocks * 7 Strassen terms
    r_total = r4 + r_k4  # 63

    Uf = np.zeros((n * m, r_total))
    Vf = np.zeros((m * p, r_total))
    Wf = np.zeros((p * n, r_total))

    Uf[:, :r4] = U44
    Vf[:16, :r4] = V44
    Wf[:16, :r4] = W44

    # For k=4: C[:, 4] = A @ B[:, 4]
    # Split A into 2x2 blocks, B[:, 4] into 2x1 vectors
    # Use Strassen 2x2x1: M1 = (a+d)(e), M2 = (c+d)(e), M3 = a(f), M4 = d(g), M5 = (a+b)(f), M6 = (c-a)(e), M7 = (b-d)(f)
    # Wait, for 2x2 x 2x1, the output is 2x1, so we need 2 outputs.
    # Let's just use schoolbook for k=4: 16 terms. That's simpler and valid.
    # Actually, let's use the 2x2 Strassen structure properly.
    # For 2x2 x 2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Schoolbook: 4 terms. Strassen doesn't help here since p=1.
    # So just use schoolbook: 16 terms for 4x4 x 4x1.
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

    # 2) Block Strassen 2x2x5: rank 52
    if time.time() < deadline - 5:
        try:
            cand = _try_strassen_4x4x5_block()
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
