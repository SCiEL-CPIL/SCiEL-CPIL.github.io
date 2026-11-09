import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    start_time = time.time()
    time_limit = 140  # Leave 10s margin for safety

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Precompute tensor slices for faster ALS
    T_flat_a = T.reshape(nm, mp * pn)  # (6, 486)
    T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # (27, 108)
    T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # (18, 162)

    # Try ranks from 36 down to 30 with efficient ALS
    ranks_to_try = [36, 35, 34, 33, 32, 31, 30]
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break

        found = False
        max_restarts = 25
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break

            # Vary initialization strategy
            if restart < 5:
                # Structured initialization based on schoolbook
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
                # Add small random perturbation
                np.random.seed(12345 + restart * 1000 + rank)
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            else:
                np.random.seed(12345 + restart * 1000 + rank)
                scale = 0.3 + (restart % 10) * 0.1
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale

            max_iter = 300
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit:
                    break

                # Solve for U: minimize ||VW - T||_F
                VW = np.einsum('br,cr->bcr', V, W)
                VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break

                # Solve for V: minimize ||UW - T||_F
                UW = np.einsum('ar,cr->acr', U, W)
                UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break

                # Solve for W: minimize ||UV - T||_F
                UV = np.einsum('ar,br->abr', U, V)
                UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break

                if iteration % 5 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break
                    if error > prev_error * 1.05 and iteration > 30:
                        break
                    prev_error = error

            if found:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break

        if found:
            continue

    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
