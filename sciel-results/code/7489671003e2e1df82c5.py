# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0

    nm = n * m
    mp = m * p
    pn = p * n

    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition(U, V, W, atol=1e-9):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=atol)

    def is_half_integer(A, tol=1e-9):
        doubled = A * 2.0
        rounded = np.round(doubled)
        return np.all(np.abs(doubled - rounded) < tol)

    def snap_to_half_integer(A):
        return np.round(A * 2.0) / 2.0

    def clean_half_int(lst):
        out = []
        for row in lst:
            new_row = []
            for v in row:
                v2 = round(float(v) * 2) / 2.0
                if v2 == int(v2):
                    new_row.append(int(v2))
                else:
                    new_row.append(float(v2))
            out.append(new_row)
        return out

    # Schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    rng = np.random.RandomState(42)
    
    # Try ranks from 60 to 89
    for r in range(60, 90):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            W = rng.randn(pn, r)
            
            # Normalize
            for c in range(r):
                U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-9)
                V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-9)
                W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-9)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # A = V outer W -> (mp*pn, r)
                A = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
                T_flat_T = T.transpose(1, 2, 0).reshape(mp*pn, nm) # (mp*pn, nm)
                # Solve A * X = T_flat_T where X = U^T (r, nm)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A, T_flat_T, rcond=None)
                    U = U_T.T
                except:
                    break
                    
                # Update V
                # B = U outer W -> (nm*pn, r)
                B = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
                T_flat_b = T.transpose(0, 2, 1).reshape(nm*pn, mp) # (nm*pn, mp)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(B, T_flat_b, rcond=None)
                    V = V_T.T
                except:
                    break
                    
                # Update W
                # C = U outer V -> (nm*mp, r)
                C = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
                T_flat_c = T.transpose(0, 1, 2).reshape(nm*mp, pn) # (nm*mp, pn)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(C, T_flat_c, rcond=None)
                    W = W_T.T
                except:
                    break
                    
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_rec - T)
            
            if err < 1e-8:
                U_h = snap_to_half_integer(U)
                V_h = snap_to_half_integer(V)
                W_h = snap_to_half_integer(W)
                
                if verify_decomposition(U_h, V_h, W_h, atol=1e-8):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_h.copy(), V_h.copy(), W_h.copy()
                    break

        if r < best_rank:
            break

    U_out = clean_half_int(best_U.tolist())
    V_out = clean_half_int(best_V.tolist())
    W_out = clean_half_int(best_W.tolist())

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
