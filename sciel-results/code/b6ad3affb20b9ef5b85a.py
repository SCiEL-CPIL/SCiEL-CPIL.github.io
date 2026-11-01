import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _try_rank(R, T, U0, V0, W0, deadline):
    """Try to find a rank-R decomposition by zeroing out columns.
    Start from a rank-R0 solution and try to eliminate columns."""
    n, m, p = DIMENSIONS
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # Flatten the tensor
    T_flat = T.flatten()
    
    # Build the linear system: for each (i,j,k), T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # This is bilinear in U,V,W, so we can't directly solve linearly.
    # Instead, fix U and V, solve for W, then fix U and W, solve for V, etc.
    
    # Alternative: use the fact that we have a candidate from rank 28.
    # Try to find a subset of columns of size R that spans the tensor.
    
    # For each subset of R columns from the 28-column solution,
    # check if the tensor can be represented.
    
    # This is combinatorial: C(28, R). For R=27, that's 28 choices.
    # For R=26, C(28,26)=378. For R=25, C(28,25)=3276.
    
    # A column r can be removed if the corresponding rank-1 tensor is in the span
    # of the others. We can check this by seeing if the tensor can be reconstructed
    # without that column, adjusting the others.
    
    # Simpler approach: for each column to remove, set it to zero and see if the
    # remaining columns can be adjusted to still represent T.
    # This requires solving a linear system for the remaining coefficients.
    
    # Let's try a different approach: use the structure.
    # The 2x4x4 matmul tensor has rank at most 28 (from 2x2x2 Strassen blocks).
    # Can we do better?
    
    # Known results: rank(2x4x4) is between 24 and 28.
    # Let's try to find rank 27 by removing one column.
    
    R0 = U0.shape[1]
    
    if R0 <= R:
        return U0, V0, W0, True
    
    # Try removing (R0 - R) columns
    # For each set of columns to keep, check if the tensor is in the span
    
    # Efficient check: for a given set of columns S, the tensor T must satisfy
    # T[i,j,k] = sum_{r in S} U[i,r]*V[j,r]*W[k,r]
    # This means the "residual" T - sum_{r in S} U[i,r]*V[j,r]*W[k,r] must be zero.
    
    # But we can also adjust U,V,W for the kept columns.
    # So we need: exists U',V',W' with support on S such that T = sum_{r in S} U'[i,r]*V'[j,r]*W'[k,r]
    
    # This is equivalent to: the tensor T has CP rank <= |S| with factors supported on S.
    
    # Hard problem. Let's use a heuristic: for each column r, remove it and try to
    # "absorb" its contribution into other columns by adjusting them.
    
    # Actually, let's just try all single-column removals for R = R0 - 1.
    
    if R == R0 - 1:
        for r_remove in range(R0):
            if time.time() > deadline:
                break
            # Keep all columns except r_remove
            keep = [r for r in range(R0) if r != r_remove]
            U_keep = U0[:, keep]
            V_keep = V0[:, keep]
            W_keep = W0[:, keep]
            
            # Check if this subset can represent T
            # Compute the residual
            Tc = np.zeros_like(T)
            for idx, r in enumerate(keep):
                Tc += np.einsum('i,j,k->ijk', U_keep[:,idx], V_keep[:,idx], W_keep[:,idx])
            
            if np.allclose(Tc, T, atol=1e-10):
                return U_keep, V_keep, W_keep, True
    
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Build the rank-28 Strassen construction
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    S = [
        ([[(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]],),
        ([[(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]],),
        ([[(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]],),
        ([[(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]],),
        ([[(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]],),
        ([[(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]],),
        ([[(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]],),
    ]

    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        if time.time() > deadline:
            break
        for j, (parts,) in enumerate(S):
            ci = base + j
            u_entries, v_entries, w_entries = parts
            for li, cf in u_entries:
                U[mA(li, a_off), ci] += cf
            for li, cf in v_entries:
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_entries:
                W[mC(li, c_co), ci] += cf

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Verify the construction
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook rank 32
        R = 32
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = R

    # Try to reduce rank to 27 by removing one column
    if time.time() < deadline:
        U27, V27, W27, ok = _try_rank(27, T, best_U, best_V, best_W, deadline)
        if ok:
            best_U, best_V, best_W = U27, V27, W27
            best_rank = 27

    # Try to reduce rank to 26 by removing two columns
    if best_rank > 26 and time.time() < deadline:
        R0 = best_rank
        for r1 in range(R0):
            for r2 in range(r1+1, R0):
                if time.time() > deadline:
                    break
                keep = [r for r in range(R0) if r != r1 and r != r2]
                U_keep = best_U[:, keep]
                V_keep = best_V[:, keep]
                W_keep = best_W[:, keep]
                
                Tc = np.zeros_like(T)
                for idx in range(len(keep)):
                    Tc += np.einsum('i,j,k->ijk', U_keep[:,idx], V_keep[:,idx], W_keep[:,idx])
                
                if np.allclose(Tc, T, atol=1e-10):
                    best_U, best_V, best_W = U_keep, V_keep, W_keep
                    best_rank = 26
                    break
            if best_rank == 26:
                break

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

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
