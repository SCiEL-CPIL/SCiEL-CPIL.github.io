"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # safety margin before 120s kill

    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    mask_1 = np.zeros(T.shape, dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mask_1[i * m + j, j * p + k, k * n + i] = True

    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        if not np.allclose(T_recon[mask_1], 1.0, atol=tol):
            return False
        if not np.allclose(T_recon[~mask_1], 0.0, atol=tol):
            return False
        return True

    def polish(U, V, W, r, max_iters=300):
        """High-precision polish: tighten the alternating minimization with
        a tighter tolerance and more iterations, then try rounding."""
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Update U
            P = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, P)
            G_U = np.einsum('abc,bcd,bce->ade', T, P, P)
            for a in range(n * m):
                try:
                    U[a, :] = np.linalg.solve(G_U[a], h_U[a])
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(G_U[a], h_U[a], rcond=None)[0]

            # Update V
            Q = U[:, None, :] * W[None, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, Q)
            G_V = np.einsum('abc,acd,ace->bde', T, Q, Q)
            for b in range(m * p):
                try:
                    V[b, :] = np.linalg.solve(G_V[b], h_V[b])
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(G_V[b], h_V[b], rcond=None)[0]

            # Update W
            R = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, R)
            G_W = np.einsum('abc,abd,abe->cde', T, R, R)
            for c in range(p * n):
                try:
                    W[c, :] = np.linalg.solve(G_W[c], h_W[c])
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(G_W[c], h_W[c], rcond=None)[0]

            # Periodic rounding check
            if it % 30 == 0:
                for scale in [1, 2, 4]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if check_exact(U_r, V_r, W_r):
                        return (r, U_r, V_r, W_r)

        # Final rounding checks
        for scale in [1, 2, 4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r):
                return (r, U_r, V_r, W_r)

        return None

    def try_rank(r, seed=42, max_iters=400, init_scale=0.5):
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)

        # Scaled random initialization for better convergence
        U = np.random.randn(n * m, r) * init_scale
        V = np.random.randn(m * p, r) * init_scale
        W = np.random.randn(p * n, r) * init_scale

        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None

            # Update U
            P = V[:, None, :] * W[None, :, :]
            h_U = np.einsum('abc,bcd->ad', T, P)
            G_U = np.einsum('abc,bcd,bce->ade', T, P, P)
            for a in range(n * m):
                try:
                    U[a, :] = np.linalg.solve(G_U[a], h_U[a])
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(G_U[a], h_U[a], rcond=None)[0]

            # Update V
            Q = U[:, None, :] * W[None, :, :]
            h_V = np.einsum('abc,acd->bd', T, Q)
            G_V = np.einsum('abc,acd,ace->bde', T, Q, Q)
            for b in range(m * p):
                try:
                    V[b, :] = np.linalg.solve(G_V[b], h_V[b])
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(G_V[b], h_V[b], rcond=None)[0]

            # Update W
            R = U[:, None, :] * V[None, :, :]
            h_W = np.einsum('abc,abd->cd', T, R)
            G_W = np.einsum('abc,abd,abe->cde', T, R, R)
            for c in range(p * n):
                try:
                    W[c, :] = np.linalg.solve(G_W[c], h_W[c])
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(G_W[c], h_W[c], rcond=None)[0]

            # Periodic rounding check
            if it % 40 == 0:
                for scale in [1, 2]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if check_exact(U_r, V_r, W_r):
                        # Polish for higher precision
                        polished = polish(U_r, V_r, W_r, r, max_iters=200)
                        if polished is not None:
                            return polished
                        return (r, U_r, V_r, W_r)

        # Final rounding checks
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r):
                polished = polish(U_r, V_r, W_r, r, max_iters=200)
                if polished is not None:
                    return polished
                return (r, U_r, V_r, W_r)

        return None

    # Try ranks from a tighter lower bound.
    # For 3x3x3x5, known results suggest rank around 20-30.
    # Start from 20 to be aggressive but safe.
    ranks_to_try = list(range(20, rank_sb))

    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break

        # Try multiple seeds with varying initialization scales
        seeds_configs = [
            (0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5),
            (7, 1.0), (13, 1.0), (42, 0.3), (99, 0.7),
            (100, 0.5), (200, 0.5), (300, 1.0), (400, 0.3),
        ]

        for seed, scale in seeds_configs:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, seed, max_iters=400, init_scale=scale)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a rank, try next rank

    return best_result
# EVOLVE-BLOCK-END
