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
        # M1 = (X00+X11)(Y00+Y11) -> P00+=M1, P11+=M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2 = (X10+X11)Y00 -> P10+=M2, P11-=M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3 = X00(Y01-Y11) -> P01+=M3, P11+=M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4 = X11(Y10-Y00) -> P00+=M4, P10+=M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5 = (X00+X01)Y11 -> P00-=M5, P01+=M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6 = (X10-X00)(Y00+Y01) -> P11+=M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7 = (X01-X11)(Y10+Y11) -> P00+=M7
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

def _verify(U, V, W, trials=3, seed=42):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    for trial in range(trials):
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
    Search for a rank-27 decomposition.
    
    Approach: The rank-28 decomposition has 4 independent Strassen blocks.
    We look for a way to merge two blocks or find a linear dependence.
    
    Key idea: Try to find if one rank-1 term can be expressed as a linear
    combination of others. Since the tensor space is 32-dim and we have 28
    terms, they're likely independent. But we can try to find a 27-term
    decomposition by allowing the 28th term to be absorbed.
    
    Alternative: Directly search over small integer perturbations of the
    rank-28 decomposition to find a rank-27 solution.
    
    Most promising: Use the fact that the 4 Strassen blocks are applied to
    different parts of the matrices. Check if there's a global linear relation
    by examining the structure more carefully.
    
    Practical approach: Try removing each of the 28 terms and see if the
    remaining 27 can be adjusted (by adding one new term) to still decompose
    the tensor. This is equivalent to checking if the removed term lies in
    the span of the other 27 in the tensor space.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Build tensor representations of each rank-1 term
    # T_i[a, b, c] = U[a,i] * V[b,i] * W[c,i]
    # Flatten to vector of length n*m*p = 32
    
    T_flat = np.zeros((rank, n * m * p), dtype=float)
    for i in range(rank):
        u_vec = U28[:, i].astype(float)
        v_vec = V28[:, i].astype(float)
        w_vec = W28[:, i].astype(float)
        tensor = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        T_flat[i, :] = tensor.reshape(-1)
    
    # For each term j, check if T_j is in the span of the other 27 terms
    # This is equivalent to checking if the rank of [T_0...T_{j-1}, T_{j+1}...T_27]
    # equals the rank of [T_0...T_27]
    
    # First, find the rank of all 28 terms
    U_all, s_all, Vt_all = np.linalg.svd(T_flat, full_matrices=False)
    tol = 1e-10
    rank_all = np.sum(s_all > tol)
    
    if rank_all < rank:
        # There's a linear dependence, so we can potentially reduce rank
        # Find the null space
        null_indices = np.where(s_all <= tol)[0]
        if len(null_indices) > 0:
            null_space = Vt_all[null_indices, :]
            for idx in null_indices:
                v = null_space[idx]
                nz = np.where(np.abs(v) > 1e-8)[0]
                if len(nz) >= 2:
                    j = nz[0]
                    U27 = np.delete(U28, j, axis=1)
                    V27 = np.delete(V28, j, axis=1)
                    W27 = np.delete(W28, j, axis=1)
                    if _verify(U27, V27, W27):
                        return U27, V27, W27
    
    # If no direct linear dependence, try a more sophisticated search
    # Look for a 27-term decomposition by allowing one term to be a
    # linear combination of others
    
    # Try each pair of terms to see if they can be merged
    for j in range(rank):
        if time.time() > _DEADLINE - 5.0:
            break
        # Remove term j
        U27 = np.delete(U28, j, axis=1)
        V27 = np.delete(V28, j, axis=1)
        W27 = np.delete(W28, j, axis=1)
        
        # Check if the remaining 27 terms still decompose the tensor
        if _verify(U27, V27, W27):
            return U27, V27, W27
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition
    U28, V28, W28 = _build_rank28()
    
    # Verify rank-28 works
    if not _verify(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to find a rank-27 decomposition
    if time.time() < _DEADLINE - 10.0:
        result = _search_rank27(U28, V28, W28)
        if result is not None:
            U27, V27, W27 = result
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
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
