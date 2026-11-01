import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _build_rank28():
    """Build the rank-28 Strassen-block decomposition. Returns (U, V, W) as int arrays."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def _search_rank27(U28, V28, W28):
    """
    Search for a rank-27 decomposition by finding a linear combination
    of the 28 rank-1 terms that cancels one term.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Build the 3-tensor for each rank-1 term
    T_flat = np.zeros((rank, n * m * p), dtype=float)
    for i in range(rank):
        u_vec = U28[:, i].astype(float)
        v_vec = V28[:, i].astype(float)
        w_vec = W28[:, i].astype(float)
        tensor = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        T_flat[i, :] = tensor.reshape(-1)
    
    # Find null space of T_flat (shape 28 x 32)
    try:
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=True)
        tol = 1e-10
        null_indices = np.where(s < tol)[0]
        if len(null_indices) > 0:
            for idx in null_indices:
                v = Vt[idx, :]
                nz = np.where(np.abs(v) > 1e-8)[0]
                if len(nz) >= 2:
                    j = nz[0]
                    if abs(v[j]) < 1e-10:
                        continue
                    U27 = np.delete(U28, j, axis=1)
                    V27 = np.delete(V28, j, axis=1)
                    W27 = np.delete(W28, j, axis=1)
                    if _verify(U27, V27, W27):
                        return U27, V27, W27
    except np.linalg.LinAlgError:
        pass
    
    return None

def _try_2x2x4_decomposition():
    """
    Try to exploit the 2x2x4 structure more deeply.
    
    The 2x4x4 multiplication can be viewed as computing two 2x2 blocks of the result:
    C = [A1, A2] * [B11 B12; B21 B22] = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each 2x2 product takes rank 7 (Strassen). Total: 4 * 7 = 28.
    
    However, we might be able to share some computations between the two halves.
    For example, if we compute:
    M1 = (A1 + A2) * (B11 + B21)  -- rank 7
    M2 = A1 * B11                 -- rank 7  
    M3 = A2 * B21                 -- rank 7
    Then A1*B11 + A2*B21 = M1 - M2 - M3 + (A1*B21 + A2*B11)
    
    This doesn't immediately help. Let's try a different approach.
    
    Actually, for 2x2x4 (multiplying 2x2 by 2x4), the rank is known to be 11.
    We can view our problem as two such multiplications:
    - Left half: [A1 A2] * [B11; B21] where A1,A2 are 2x2, B11,B21 are 2x2
      This gives a 2x2 result.
    - Right half: [A1 A2] * [B12; B22] where B12,B22 are 2x2
      This gives another 2x2 result.
    
    If each 2x2x4 takes rank 11, total would be 22, which is better than 28.
    But we need to verify this is actually possible and find the decomposition.
    
    For now, this is a placeholder for future work.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition
    U28, V28, W28 = _build_rank28()
    
    # Verify rank-28 works
    if not _verify(U28, V28, W28):
        # Fallback: return rank-28 anyway (shouldn't happen)
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to find a rank-27 decomposition
    if time.time() < _DEADLINE - 10.0:
        result = _search_rank27(U28, V28, W28)
        if result is not None:
            U27, V27, W27 = result
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Try the 2x2x4 decomposition approach
    if time.time() < _DEADLINE - 5.0:
        result2 = _try_2x2x4_decomposition()
        if result2 is not None:
            U_new, V_new, W_new = result2
            if _verify(U_new, V_new, W_new):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
