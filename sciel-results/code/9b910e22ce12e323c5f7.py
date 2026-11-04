import numpy as np
import time
import itertools

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _block_specs():
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return False
    return True

def _build_tensor_matrix(U, V, W, n, m, p):
    """Build the tensor matrix M where M[idx, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    R = U.shape[1]
    dim = n * p * m
    M = np.zeros((dim, R))
    for r in range(R):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx = i * p * m + k * m + j
                    M[idx, r] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    return M

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0

    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"

    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]

    # Build tensor vector matrix
    M = _build_tensor_matrix(U, V, W, n, m, p)

    U_final = U
    V_final = V
    W_final = W

    try:
        # Compute SVD to find numerical rank
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        max_sv = s_vals[0] if len(s_vals) > 0 else 0.0
        tol = 1e-10 * max_sv * max(M.shape)
        rank_svd = int(np.sum(s_vals > tol))

        if rank_svd < R and rank_svd > 0:
            # Forward selection: pick columns that maximize residual norm
            selected = []
            remaining = set(range(R))
            col_norms = np.linalg.norm(M, axis=0)

            for step in range(rank_svd):
                if not remaining:
                    break
                if time.time() > deadline - 30.0:
                    break

                if not selected:
                    best_col = max(remaining, key=lambda c: col_norms[c])
                    best_norm = col_norms[best_col]
                else:
                    M_sel = M[:, selected]
                    rem_list = sorted(remaining)
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(M_sel, M[:, rem_list], rcond=None)
                        residuals = M[:, rem_list] - M_sel @ coeffs
                        res_norms = np.linalg.norm(residuals, axis=0)
                        best_local = np.argmax(res_norms)
                        best_col = rem_list[best_local]
                        best_norm = res_norms[best_local]
                    except np.linalg.LinAlgError:
                        break

                if best_norm < 1e-8:
                    break

                selected.append(best_col)
                remaining.discard(best_col)

            if len(selected) == rank_svd:
                keep = sorted(selected)
                U_final = U[:, keep]
                V_final = V[:, keep]
                W_final = W[:, keep]
    except Exception:
        pass

    # Try additional rank reduction by checking if we can remove more columns
    # using a more aggressive approach
    try:
        if time.time() < deadline - 20.0:
            R_cur = U_final.shape[1]
            if R_cur < R:
                M_cur = _build_tensor_matrix(U_final, V_final, W_final, n, m, p)
                # Check if any column is a linear combination of others
                # Use QR with column pivoting to find a minimal spanning set
                Q, R_qr, piv = np.linalg.qr(M_cur, mode='complete', pivoting=True)
                # The rank is the number of non-zero diagonal entries in R_qr
                diag = np.abs(np.diag(R_qr))
                tol2 = 1e-10 * diag[0] if len(diag) > 0 else 0
                effective_rank = int(np.sum(diag > tol2))

                if effective_rank < R_cur:
                    # Keep only the first effective_rank pivoted columns
                    keep2 = sorted(piv[:effective_rank].tolist())
                    U_try = U_final[:, keep2]
                    V_try = V_final[:, keep2]
                    W_try = W_final[:, keep2]
                    if _verify(U_try, V_try, W_try, n, m, p):
                        U_final = U_try
                        V_final = V_try
                        W_final = W_try
    except Exception:
        pass

    # Final verification
    if time.time() < deadline - 2.0:
        if not _verify(U_final, V_final, W_final, n, m, p):
            U_final = U
            V_final = V
            W_final = W

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
