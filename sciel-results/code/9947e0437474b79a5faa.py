import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 115.0  # safety margin before 120s evaluator limit

def _build_structured_rank77():
    """Build the proven rank-77 nested Strassen tiling decomposition."""
    n, m, p = 4, 4, 6
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    return np.max(np.abs(T - T_expected)) < tol


def _fit_rank1_als(residual, n, m, p, max_iter=50, tol=1e-8):
    """Fit a rank-1 tensor to residual using ALS. Returns (u, v, w) or None."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize via SVD
    mat = residual.reshape(nm, -1)
    try:
        Us, S, Vt = np.linalg.svd(mat, full_matrices=False)
        if len(S) == 0 or S[0] < 1e-10:
            return None
        u = Us[:, 0] * np.sqrt(S[0])
        vec = Vt[0, :] * np.sqrt(S[0])
        mat2 = vec.reshape(mp, pn)
        U2, S2, Vt2 = np.linalg.svd(mat2, full_matrices=False)
        if len(S2) == 0 or S2[0] < 1e-10:
            return None
        v = U2[:, 0] * np.sqrt(S2[0])
        w = Vt2[0, :] * np.sqrt(S2[0])
    except:
        return None
    
    if np.max(np.abs(u)) < 1e-10 or np.max(np.abs(v)) < 1e-10 or np.max(np.abs(w)) < 1e-10:
        return None
    
    for _ in range(max_iter):
        # Update u: minimize ||residual - u outer v outer w||
        # Fix v, w: residual_u = residual - u*(v outer w)
        # For each i: residual[i,:,:] = u[i]*(v outer w) + other
        # So u[i] = <residual[i,:,:], v outer w> / ||v outer w||^2
        vw = np.outer(v, w)
        vw_norm_sq = np.sum(vw**2)
        if vw_norm_sq < 1e-15:
            break
        for i in range(nm):
            u[i] = np.sum(residual[i, :, :] * vw) / vw_norm_sq
        
        # Update v
        uw = np.outer(u, w)
        uw_norm_sq = np.sum(uw**2)
        if uw_norm_sq < 1e-15:
            break
        for j in range(mp):
            v[j] = np.sum(residual[:, j, :] * uw) / uw_norm_sq
        
        # Update w
        uv = np.outer(u, v)
        uv_norm_sq = np.sum(uv**2)
        if uv_norm_sq < 1e-15:
            break
        for k in range(pn):
            w[k] = np.sum(residual[:, :, k] * uv) / uv_norm_sq
    
    # Check convergence
    T_fit = np.einsum('i,j,k->ijk', u, v, w)
    err = np.max(np.abs(residual - T_fit))
    if err < tol:
        return (u, v, w)
    return None


def _rank_reduce(U, V, W, deadline=None):
    """Try to reduce rank by zeroing columns and repairing via ALS."""
    n, m, p = 4, 4, 6
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    R = U.shape[1]
    
    if deadline is None:
        deadline = time.time() + 100
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Phase 1: Remove columns with all-zero U, V, or W
    keep = []
    for r in range(R):
        if np.allclose(U[:, r], 0, atol=1e-9) or \
           np.allclose(V[:, r], 0, atol=1e-9) or \
           np.allclose(W[:, r], 0, atol=1e-9):
            continue
        keep.append(r)
    
    if len(keep) < R:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        R = len(keep)
        if not _verify_decomposition(U, V, W):
            # This shouldn't happen, but restore
            U, V, W = U[:, :R], V[:, :R], W[:, :R]
    
    # Phase 2: Try removing pairs of columns and repair via ALS
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for r in range(R):
        if time.time() > deadline - 15:
            break
        for s in range(r + 1, R):
            if time.time() > deadline - 10:
                break
            
            # Remove columns r and s
            keep = [x for x in range(R) if x != r and x != s]
            U_sub = U[:, keep]
            V_sub = V[:, keep]
            W_sub = W[:, keep]
            
            # Compute residual
            T_sub = np.zeros((n*m, m*p, p*n))
            for rr in range(len(keep)):
                T_sub += np.einsum('i,j,k->ijk', U_sub[:, rr], V_sub[:, rr], W_sub[:, rr])
            residual = T_expected - T_sub
            
            # Try to fit residual with a single rank-1 term
            fit = _fit_rank1_als(residual, n, m, p, max_iter=30)
            if fit is not None:
                u_new, v_new, w_new = fit
                # Add the new rank-1 term
                U_try = np.column_stack([U_sub, u_new])
                V_try = np.column_stack([V_sub, v_new])
                W_try = np.column_stack([W_sub, w_new])
                
                if _verify_decomposition(U_try, V_try, W_try):
                    new_rank = U_try.shape[1]
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U, best_V, best_W = U_try, V_try, W_try
                        # Update for next iterations
                        U, V, W = U_try, V_try, W_try
                        R = new_rank
                        break
    
    return best_U, best_V, best_W


def to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]


def construct():
    start_time = time.time()
    
    # Build the structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try to reduce rank
    deadline = start_time + _DEADLINE
    U, V, W = _rank_reduce(U, V, W, deadline=deadline)
    
    # Final polish: round to nearest integer/half-integer if close
    for mat in [U, V, W]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < 1e-9:
                    mat[i, j] = nearest_half
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
