"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search.

Strategy: Split the 8 output columns into two 4-column blocks (3x4 sub-problems).
Each 3x4 block: C = A_left(2x2)*B_left(2x4) + A_right(2x1)*B_right(1x4).
- A_left*B_left: M(2,2,4) = 2 x M(2,2,2) = 2 x Strassen(7) = 14
- A_right*B_right: M(2,1,4) = 4 x rank-1 = 4
Total per 3x4 block: 18, grand total: 36.

Also try the 4-block approach (4 x M(2,3,2)) targeting rank 10 per block = 40 total.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = 2, 3, 8

_TARGET = np.zeros((N * M, M * P, P * N), dtype=bool)
for _i in range(N):
    for _j in range(M):
        for _k in range(P):
            _TARGET[_i * M + _j, _j * P + _k, _k * N + _i] = True


def _verify(U, V, W, tol=1e-9):
    """Verify U, V, W exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                e = abs(T[a, b, c] - 1.0)
                if e > max_err:
                    max_err = e
    non_target_err = float(np.max(np.abs(T[~_TARGET]))) if (~_TARGET).any() else 0.0
    return max_err <= tol and non_target_err <= tol


def _round_and_verify(U, V, W, tol=1e-6):
    """Try rounding to half-integers / integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for scale, name in ((2, 'half'), (1, 'int')):
        Uh = np.round(U_arr * scale) / scale
        Vh = np.round(V_arr * scale) / scale
        Wh = np.round(W_arr * scale) / scale
        if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist(), tol=tol):
            return Uh.tolist(), Vh.tolist(), Wh.tolist()
    return None


def _strassen_2x2(A_indices, B_indices, C_indices, U, V, W, col):
    """Add a Strassen decomposition for a 2x2x2 sub-problem.
    
    A_indices: 4 indices into U (a00, a01, a10, a11)
    B_indices: 4 indices into V (b00, b01, b10, b11)  
    C_indices: 4 indices into W (c00, c01, c10, c11)
    """
    a00, a01, a10, a11 = A_indices
    b00, b01, b10, b11 = B_indices
    c00, c01, c10, c11 = C_indices
    
    # m1 = (a00+a11)(b00+b11) -> c00 += m1, c01 -= m1, c10 -= m1, c11 += m1
    U[a00, col] += 1; U[a11, col] += 1
    V[b00, col] += 1; V[b11, col] += 1
    W[c00, col] += 1; W[c01, col] -= 1; W[c10, col] -= 1; W[c11, col] += 1
    col += 1
    
    # m2 = (a10+a11)b00 -> c10 += m2, c11 -= m2
    U[a10, col] += 1; U[a11, col] += 1
    V[b00, col] += 1
    W[c10, col] += 1; W[c11, col] -= 1
    col += 1
    
    # m3 = a00(b01-b11) -> c00 -= m3, c01 += m3, c11 += m3
    U[a00, col] += 1
    V[b01, col] += 1; V[b11, col] -= 1
    W[c00, col] -= 1; W[c01, col] += 1; W[c11, col] += 1
    col += 1
    
    # m4 = a11(b10-b00) -> c00 += m4, c10 += m4
    U[a11, col] += 1
    V[b10, col] += 1; V[b00, col] -= 1
    W[c00, col] += 1; W[c10, col] += 1
    col += 1
    
    # m5 = (a00+a01)b11 -> c01 += m5, c11 -= m5
    U[a00, col] += 1; U[a01, col] += 1
    V[b11, col] += 1
    W[c01, col] += 1; W[c11, col] -= 1
    col += 1
    
    # m6 = (a10-a00)(b00+b01) -> c10 += m6
    U[a10, col] += 1; U[a00, col] -= 1
    V[b00, col] += 1; V[b01, col] += 1
    W[c10, col] += 1
    col += 1
    
    # m7 = (a01-a11)(b10+b11) -> c01 += m7
    U[a01, col] += 1; U[a11, col] -= 1
    V[b10, col] += 1; V[b11, col] += 1
    W[c01, col] += 1
    col += 1
    
    return col


def _build_rank36_decomposition():
    """Build rank-36 decomposition using 3x4 block structure.
    
    Split B's 8 columns into two groups of 4: cols 0-3 and cols 4-7.
    For each group, split A into A_left (cols 0,1) and A_right (col 2).
    A_left * B_left: M(2,2,4) = 2 Strassen blocks = 14
    A_right * B_right: M(2,1,4) = 4 rank-1 terms
    Total per group: 18, grand total: 36
    """
    n, m, p = N, M, P
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for group in range(2):
        # B columns for this group: 4*group to 4*group+3
        b_start = 4 * group
        
        # A_left: rows 0,1 of A (indices 0,1,3,4 in U)
        # A_right: row 2 of A (indices 2,5 in U)
        
        # B_left: rows 0,1 of B (indices 0-7 and 8-15 in V for cols b_start to b_start+3)
        # B_right: row 2 of B (indices 16-23 in V for cols b_start to b_start+3)
        
        # For A_left * B_left: M(2,2,4)
        # Split into 2 x M(2,2,2)
        for sub in range(2):
            k0 = b_start + 2 * sub
            k1 = b_start + 2 * sub + 1
            
            # A indices: a00=U[0], a01=U[1], a10=U[3], a11=U[4]
            # B indices: b00=V[k0], b01=V[k1], b10=V[8+k0], b11=V[8+k1]
            # C indices: c00=W[2*k0], c01=W[2*k1], c10=W[2*k0+1], c11=W[2*k1+1]
            col = _strassen_2x2(
                (0, 1, 3, 4),
                (k0, k1, 8 + k0, 8 + k1),
                (2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1),
                U, V, W, col
            )
        
        # For A_right * B_right: M(2,1,4)
        # 4 rank-1 terms: for each k in b_start to b_start+3
        for kk in range(4):
            kv = b_start + kk
            # A[0,2] * B[2,kv] -> C[0,kv]
            U[2, col] = 1
            V[16 + kv, col] = 1
            W[kv * 2 + 0, col] = 1
            col += 1
            # A[1,2] * B[2,kv] -> C[1,kv]
            U[5, col] = 1
            V[16 + kv, col] = 1
            W[kv * 2 + 1, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U.tolist(), V.tolist(), W.tolist()


def _build_rank44_decomposition():
    """Explicit rank-44 decomposition: 4 blocks x (Strassen 7 + outer 4)."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    col = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for A_left (cols 0,1) * B_top (rows 0,1)
        col = _strassen_2x2(
            (0, 1, 3, 4),
            (k0, k1, 8 + k0, 8 + k1),
            (2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1),
            U, V, W, col
        )
        
        # Outer terms for A_right (col 2) * B_bottom (row 2)
        for ii, kv in ((0, k0), (1, k0), (0, k1), (1, k1)):
            U[ii * 3 + 2, col] = 1
            V[16 + kv, col] = 1
            W[kv * 2 + ii, col] = 1
            col += 1
    
    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()


def _als_block(rank, block, max_iters=100, deadline=None, seed=None):
    """ALS on a single 2x3x2 block to find a rank-`rank` scheme."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    U = rng.randn(6, rank) * 0.5
    V = rng.randn(24, rank) * 0.5
    W = rng.randn(16, rank) * 0.5
    k0 = 2 * block
    k1 = 2 * block + 1
    prev = float('inf')
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            return None
        
        # Update U
        for a in range(6):
            i = a // 3
            j = a % 3
            Mm = np.zeros((16, rank))
            row = 0
            for kk in (k0, k1):
                Mm[row, :] = V[j * 8 + kk, :] * W[kk * 2 + i, :]
                row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                U[a, :] = x
            except Exception:
                pass
        
        # Update V
        for b in range(24):
            j = b // 8
            k = b % 8
            if k not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for ii in range(2):
                for kk in (k0, k1):
                    Mm[row, :] = U[ii * 3 + j, :] * W[kk * 2 + ii, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                V[b, :] = x
            except Exception:
                pass
        
        # Update W
        for c in range(16):
            kk = c // 2
            i = c % 2
            if kk not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for ii in range(2):
                for j in range(3):
                    Mm[row, :] = U[ii * 3 + j, :] * V[j * 8 + kk, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                W[c, :] = x
            except Exception:
                pass
        
        # Residual
        T = np.zeros((6, 24, 16))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = 0.0
        for i in range(2):
            for j in range(3):
                for k in (k0, k1):
                    e = abs(T[i * 3 + j, j * 8 + k, k * 2 + i] - 1.0)
                    if e > err:
                        err = e
        
        if err < 1e-9:
            return U, V, W
        if err > prev * 1.5:
            break
        prev = err
    
    return None


def _polish_block(U, V, W, block, max_iters=50, deadline=None):
    """High-precision polish for a found block scheme."""
    k0 = 2 * block
    k1 = 2 * block + 1
    rank = U.shape[1]
    prev = float('inf')
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        for a in range(6):
            i = a // 3
            j = a % 3
            Mm = np.zeros((16, rank))
            row = 0
            for kk in (k0, k1):
                Mm[row, :] = V[j * 8 + kk, :] * W[kk * 2 + i, :]
                row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                U[a, :] = x
            except Exception:
                pass
        
        for b in range(24):
            j = b // 8
            k = b % 8
            if k not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for ii in range(2):
                for kk in (k0, k1):
                    Mm[row, :] = U[ii * 3 + j, :] * W[kk * 2 + ii, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                V[b, :] = x
            except Exception:
                pass
        
        for c in range(16):
            kk = c // 2
            i = c % 2
            if kk not in (k0, k1):
                continue
            Mm = np.zeros((16, rank))
            row = 0
            for ii in range(2):
                for j in range(3):
                    Mm[row, :] = U[ii * 3 + j, :] * V[j * 8 + kk, :]
                    row += 1
            try:
                x, *_ = np.linalg.lstsq(Mm, np.ones(16), rcond=None)
                W[c, :] = x
            except Exception:
                pass
        
        T = np.zeros((6, 24, 16))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = 0.0
        for i in range(2):
            for j in range(3):
                for k in (k0, k1):
                    e = abs(T[i * 3 + j, j * 8 + k, k * 2 + i] - 1.0)
                    if e > err:
                        err = e
        
        if err < 1e-12:
            break
        if err > prev * 1.01:
            break
        prev = err
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank

    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill

    # 1) Try explicit rank-36 (3x4 block structure).
    try:
        U36, V36, W36 = _build_rank36_decomposition()
        if _verify(U36, V36, W36, tol=1e-6):
            rounded = _round_and_verify(U36, V36, W36)
            best_U, best_V, best_W = rounded if rounded else (U36, V36, W36)
            best_rank = 36
    except Exception:
        pass

    # 2) Try explicit rank-44 (Strassen + outer) as a safe fallback.
    if best_rank > 44:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            if _verify(U44, V44, W44, tol=1e-6):
                rounded = _round_and_verify(U44, V44, W44)
                best_U, best_V, best_W = rounded if rounded else (U44, V44, W44)
                best_rank = 44
        except Exception:
            pass

    # 3) If we only have 44+, search for rank-10 per-block scheme via ALS.
    if best_rank > 40 and time.time() < deadline:
        block_schemes = []
        for block in range(4):
            found = None
            for seed in range(20):
                if time.time() > deadline:
                    break
                res = _als_block(10, block, max_iters=100, deadline=deadline, seed=seed)
                if res is not None:
                    U, V, W = res
                    # Polish
                    U, V, W = _polish_block(U, V, W, block, max_iters=30, deadline=deadline)
                    # Verify local residual
                    T = np.zeros((6, 24, 16))
                    for r in range(10):
                        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    err = 0.0
                    k0 = 2 * block
                    k1 = 2 * block + 1
                    for i in range(2):
                        for j in range(3):
                            for k in (k0, k1):
                                e = abs(T[i * 3 + j, j * 8 + k, k * 2 + i] - 1.0)
                                if e > err:
                                    err = e
                    if err < 1e-7:
                        found = (U, V, W)
                        break
            if found is None:
                block_schemes = None
                break
            block_schemes.append(found)
        
        if block_schemes is not None:
            U = np.zeros((6, 40))
            V = np.zeros((24, 40))
            W = np.zeros((16, 40))
            col = 0
            for (Ub, Vb, Wb) in block_schemes:
                U[:, col:col + 10] = Ub
                V[:, col:col + 10] = Vb
                W[:, col:col + 10] = Wb
                col += 10
            if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                rounded = _round_and_verify(U, V, W)
                best_U, best_V, best_W = rounded if rounded else (U.tolist(), V.tolist(), W.tolist())
                best_rank = 40

    return {"U": best_U, "V": best_V, "W": best_W}
