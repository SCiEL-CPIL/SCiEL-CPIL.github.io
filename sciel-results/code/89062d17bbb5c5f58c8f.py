"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 48

    # Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute reshaped views for ALS
    T_nm = T.reshape(nm, mp * pn)
    T_mp = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_pn = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    deadline = time.time() + 110

    def check_exact(U, V, W, rank):
        """Check if U,V,W give exact decomposition with integer or half-integer entries."""
        # Try integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if (np.max(np.abs(U - U_r)) < 1e-5 and np.max(np.abs(V - V_r)) < 1e-5 and
            np.max(np.abs(W - W_r)) < 1e-5):
            recon = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
            if np.max(np.abs(recon - T)) < 1e-5:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        # Try half-integer
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if (np.max(np.abs(U - U_h)) < 1e-5 and np.max(np.abs(V - V_h)) < 1e-5 and
            np.max(np.abs(W - W_h)) < 1e-5):
            U2 = (U_h * 2).astype(np.float64)
            V2 = (V_h * 2).astype(np.float64)
            W2 = (W_h * 2).astype(np.float64)
            recon = np.einsum('ac,bc,dc->abd', U2, V2, W2)
            if np.max(np.abs(recon - T * 8)) < 1e-5:
                return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}

        return None

    def als(rank, max_iters=60, n_trials=4):
        best_err = float('inf')
        best_UVW = None

        for trial in range(n_trials):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5

            for it in range(max_iters):
                if time.time() > deadline:
                    break

                # Update U
                for c in range(rank):
                    vw = np.outer(V[:, c], W[:, c]).ravel()
                    d = vw @ vw
                    if d > 1e-12:
                        U[:, c] = (T_nm @ vw) / d

                # Update V
                for c in range(rank):
                    uw = np.outer(U[:, c], W[:, c]).ravel()
                    d = uw @ uw
                    if d > 1e-12:
                        V[:, c] = (T_mp @ uw) / d

                # Update W
                for c in range(rank):
                    uv = np.outer(U[:, c], V[:, c]).ravel()
                    d = uv @ uv
                    if d > 1e-12:
                        W[:, c] = (T_pn @ uv) / d

                # Compute error periodically
                if it % 10 == 0 or it == max_iters - 1:
                    recon = np.einsum('ac,bc,dc->abd', U, V, W)
                    err = np.linalg.norm(recon - T)
                    if err < best_err:
                        best_err = err
                        best_UVW = (U.copy(), V.copy(), W.copy())
                    if err < 1e-10:
                        break

        if best_UVW is not None:
            U_b, V_b, W_b = best_UVW
            return check_exact(U_b, V_b, W_b, rank)
        return None

    # Search for lower rank decompositions
    for rank in range(47, 40, -1):
        if time.time() > deadline:
            break
        result = als(rank, max_iters=50, n_trials=3)
        if result is not None:
            best_result = result
            break

    return best_result
# EVOLVE-BLOCK-END

