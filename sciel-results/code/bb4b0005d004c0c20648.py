"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search.

Strategy: Decompose into four 2x3x2 blocks.
1. Explicit rank-11 decomposition per block (Total 44) using Strassen + outer terms.
2. Search for rank-10 decomposition per block (Total 40) using HOOI.
3. Search for rank-43 decomposition (e.g., three rank-11 and one rank-10 blocks).
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = 2, 3, 8

# Precompute the target mask and index structures once.
_TARGET = np.zeros((N * M, M * P, P * N), dtype=bool)
for _i in range(N):
    for _j in range(M):
        for _k in range(P):
            _TARGET[_i * M + _j, _j * P + _k, _k * N + _i] = True


def _verify(U, V, W, tol=1e-9):
    """Verify U, V, W (lists of lists) exactly decompose the tensor."""
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


def _strassen_block(block):
    """Return (U,V,W) local factor matrices (6x7, 24x7, 16x7) for a 2x2*2x2
    Strassen block acting on B columns (k0,k1), A cols 0,1."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 7))
    V = np.zeros((24, 7))
    W = np.zeros((16, 7))
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[4, 0] += 1
    V[k0, 0] += 1; V[8 + k1, 0] += 1
    W[2 * k0, 0] += 1; W[2 * k1 + 1, 0] += 1
    # m2 = (a10+a11)b00
    U[3, 1] += 1; U[4, 1] += 1
    V[k0, 1] += 1
    W[2 * k0 + 1, 1] += 1; W[2 * k1 + 1, 1] += -1
    # m3 = a00(b01-b11)
    U[0, 2] += 1
    V[k1, 2] += 1; V[8 + k1, 2] += -1
    W[2 * k1, 2] += 1; W[2 * k1 + 1, 2] += 1
    # m4 = a11(b10-b00)
    U[4, 3] += 1
    V[8 + k0, 3] += 1; V[k0, 3] += -1
    W[2 * k0, 3] += 1; W[2 * k0 + 1, 3] += 1
    # m5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[8 + k1, 4] += 1
    W[2 * k0, 4] += -1; W[2 * k1, 4] += 1
    # m6 = (a10-a00)(b00+b01)
    U[3, 5] += 1; U[0, 5] += -1
    V[k0, 5] += 1; V[k1, 5] += 1
    W[2 * k1 + 1, 5] += 1
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[4, 6] += -1
    V[8 + k0, 6] += 1; V[8 + k1, 6] += 1
    W[2 * k0, 6] += 1
    return U, V, W


def _outer_block(block):
    """4 rank-1 terms for A_right (col 2) * B_bottom (row 2)."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 4))
    V = np.zeros((24, 4))
    W = np.zeros((16, 4))
    t = 0
    for ii in range(2):
        for kk in range(2):
            kv = k0 if kk == 0 else k1
            U[ii * 3 + 2, t] = 1
            V[16 + kv, t] = 1
            W[kv * 2 + ii, t] = 1
            t += 1
    return U, V, W


def _build_rank44_decomposition():
    """Explicit rank-44 decomposition: 4 blocks x (Strassen 7 + outer 4)."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    col = 0
    for block in range(4):
        Us, Vs, Ws = _strassen_block(block)
        Uo, Vo, Wo = _outer_block(block)
        U[:, col:col + 7] = Us
        V[:, col:col + 7] = Vs
        W[:, col:col + 7] = Ws
        col += 7
        U[:, col:col + 4] = Uo
        V[:, col:col + 4] = Vo
        W[:, col:col + 4] = Wo
        col += 4
    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()


def _hooi_block(rank, block, max_iters=20, deadline=None, seed=None):
    """HOOI on a single 2x3x2 block to find a rank-`rank` scheme."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    U = rng.randn(6, rank)
    V = rng.randn(24, rank)
    W = rng.randn(16, rank)
    
    # Orthonormalize initial guess
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    W, _ = np.linalg.qr(W)
    
    k0 = 2 * block
    k1 = 2 * block + 1
    
    # Precompute indices for the local tensor
    # Local tensor T_local[i, j, k] = 1 if (i, j, k) is a valid multiplication term
    # i in 0..1, j in 0..2, k in {k0, k1}
    # But we work with global indices for the factors.
    # U has size 6, V has size 24, W has size 16.
    # We only care about entries where V's k-index is k0 or k1.
    
    # Target vector for the local block
    # We want to match the tensor T such that T[i, j, k] = 1 for valid i,j,k
    # In the local block, the "valid" terms are:
    # i in 0..1, j in 0..2, k in {k0, k1}
    # The global indices are:
    # u_idx = i * 3 + j
    # v_idx = j * 8 + k
    # w_idx = k * 2 + i
    
    # We construct a "flattened" target matrix for each mode update.
    
    prev_err = float('inf')
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            return None
            
        # Update U
        # For each u_idx, we want to solve for U[u_idx, :] such that
        # sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r] = Target[u_idx, v_idx, w_idx]
        # This is equivalent to:
        # U[u_idx, :] * (V[:, r] * W[:, r])^T = Target_u
        # Where Target_u is the slice of the target tensor for fixed u_idx.
        
        U_new = np.zeros_like(U)
        for u_idx in range(6):
            # Target vector for this u_idx
            target = np.zeros(16) # 16 entries in W dimension? No, 24*16?
            # Actually, we fix u_idx. The equation is:
            # sum_r U[u_idx, r] * (V[:, r] * W[:, r]) = Target_slice_u
            # Target_slice_u is a vector of length 24*16? No.
            # The tensor is T[u, v, w]. Fixing u, we have a matrix T_u of size 24x16.
            # We want U[u, :] such that sum_r U[u, r] V[:, r] W[:, r]^T = T_u
            # This is a linear system in U[u, :].
            # Let M_u be a matrix of size (24*16, rank).
            # M_u[flat_idx, r] = V[flat_idx // 16, r] * W[flat_idx % 16, r]
            # b_u = T_u.flatten()
            
            T_u = np.zeros((24, 16))
            # Fill T_u with 1s where the local term is valid
            # i = u_idx // 3, j = u_idx % 3
            i = u_idx // 3
            j = u_idx % 3
            for k in (k0, k1):
                v_idx = j * 8 + k
                w_idx = k * 2 + i
                T_u[v_idx, w_idx] = 1.0
            
            M_u = np.zeros((24 * 16, rank))
            for v_idx in range(24):
                for w_idx in range(16):
                    M_u[v_idx * 16 + w_idx, :] = V[v_idx, :] * W[w_idx, :]
            
            b_u = T_u.flatten()
            try:
                x, *_ = np.linalg.lstsq(M_u, b_u, rcond=None)
                U_new[u_idx, :] = x
            except Exception:
                U_new[u_idx, :] = U[u_idx, :]
        
        # Update V
        V_new = np.zeros_like(V)
        for v_idx in range(24):
            j = v_idx // 8
            k = v_idx % 8
            if k not in (k0, k1):
                continue
            T_v = np.zeros((6, 16))
            for i in range(2):
                u_idx = i * 3 + j
                w_idx = k * 2 + i
                T_v[u_idx, w_idx] = 1.0
            
            M_v = np.zeros((6 * 16, rank))
            for u_idx in range(6):
                for w_idx in range(16):
                    M_v[u_idx * 16 + w_idx, :] = U[u_idx, :] * W[w_idx, :]
            
            b_v = T_v.flatten()
            try:
                x, *_ = np.linalg.lstsq(M_v, b_v, rcond=None)
                V_new[v_idx, :] = x
            except Exception:
                V_new[v_idx, :] = V[v_idx, :]
        
        # Update W
        W_new = np.zeros_like(W)
        for w_idx in range(16):
            k = w_idx // 2
            i = w_idx % 2
            if k not in (k0, k1):
                continue
            T_w = np.zeros((6, 24))
            for j in range(3):
                u_idx = i * 3 + j
                v_idx = j * 8 + k
                T_w[u_idx, v_idx] = 1.0
            
            M_w = np.zeros((6 * 24, rank))
            for u_idx in range(6):
                for v_idx in range(24):
                    M_w[u_idx * 24 + v_idx, :] = U[u_idx, :] * V[v_idx, :]
            
            b_w = T_w.flatten()
            try:
                x, *_ = np.linalg.lstsq(M_w, b_w, rcond=None)
                W_new[w_idx, :] = x
            except Exception:
                W_new[w_idx, :] = W[w_idx, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        # Compute residual for the local block
        err = 0.0
        for i in range(2):
            for j in range(3):
                for k in (k0, k1):
                    u_idx = i * 3 + j
                    v_idx = j * 8 + k
                    w_idx = k * 2 + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    err = max(err, abs(val - 1.0))
        
        if err < 1e-9:
            return U, V, W
        
        if err > prev_err * 1.1:
            break
        prev_err = err
        
    return None


def _build_rank40_decomposition(deadline=None):
    """Try to build a rank-40 decomposition using HOOI for rank-10 per block."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 40))
    V = np.zeros((m * p, 40))
    W = np.zeros((p * n, 40))
    
    for block in range(4):
        if deadline and time.time() > deadline:
            return None
        found = None
        for seed in range(20):
            if deadline and time.time() > deadline:
                return None
            result = _hooi_block(10, block, max_iters=30, deadline=deadline, seed=seed)
            if result is not None:
                found = result
                break
        if found is None:
            return None
        Ub, Vb, Wb = found
        col = block * 10
        U[:, col:col + 10] = Ub
        V[:, col:col + 10] = Vb
        W[:, col:col + 10] = Wb
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_rank43_decomposition(deadline=None):
    """Try to build a rank-43 decomposition: 3 blocks of rank 11, 1 block of rank 10."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 43))
    V = np.zeros((m * p, 43))
    W = np.zeros((p * n, 43))
    
    # We need to assign ranks to blocks. Let's say block 3 gets rank 10, others 11.
    # Or try all permutations? No, just try one configuration.
    # If rank 10 is found for any block, we can use it.
    
    # First, find a rank 10 block
    rank10_block = None
    rank10_factors = None
    for b in range(4):
        if deadline and time.time() > deadline:
            return None
        for seed in range(10):
            result = _hooi_block(10, b, max_iters=20, deadline=deadline, seed=seed)
            if result is not None:
                rank10_block = b
                rank10_factors = result
                break
        if rank10_block is not None:
            break
            
    if rank10_block is None:
        return None
        
    # Now build the rest with rank 11 (using explicit construction)
    col = 0
    for b in range(4):
        if b == rank10_block:
            Ub, Vb, Wb = rank10_factors
            U[:, col:col+10] = Ub
            V[:, col:col+10] = Vb
            W[:, col:col+10] = Wb
            col += 10
        else:
            Us, Vs, Ws = _strassen_block(b)
            Uo, Vo, Wo = _outer_block(b)
            U[:, col:col+7] = Us
            V[:, col:col+7] = Vs
            W[:, col:col+7] = Ws
            col += 7
            U[:, col:col+4] = Uo
            V[:, col:col+4] = Vo
            W[:, col:col+4] = Wo
            col += 4
            
    if col != 43:
        return None
        
    return U.tolist(), V.tolist(), W.tolist()


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
    deadline = start + 140

    # 1) Try rank-43 via HOOI (1 block rank 10, 3 blocks rank 11)
    try:
        result43 = _build_rank43_decomposition(deadline=deadline)
        if result43 is not None:
            U43, V43, W43 = result43
            if _verify(U43, V43, W43, tol=1e-6):
                rounded = _round_and_verify(U43, V43, W43)
                best_U, best_V, best_W = rounded if rounded else (U43, V43, W43)
                best_rank = 43
    except Exception:
        pass

    # 2) Try rank-40 via HOOI
    if best_rank > 40:
        try:
            result40 = _build_rank40_decomposition(deadline=deadline)
            if result40 is not None:
                U40, V40, W40 = result40
                if _verify(U40, V40, W40, tol=1e-6):
                    rounded = _round_and_verify(U40, V40, W40)
                    best_U, best_V, best_W = rounded if rounded else (U40, V40, W40)
                    best_rank = 40
        except Exception:
            pass

    # 3) Try explicit rank-44 (Strassen + outer) as a safe fallback.
    if best_rank > 44:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            if _verify(U44, V44, W44, tol=1e-6):
                rounded = _round_and_verify(U44, V44, W44)
                best_U, best_V, best_W = rounded if rounded else (U44, V44, W44)
                best_rank = 44
        except Exception:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
