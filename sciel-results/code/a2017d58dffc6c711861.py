import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    DEADLINE = 125.0  # leave 5s margin for safety

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 21, 21

    # Build the multiplication tensor T[a,b,c]
    # a = k*n+i (W index), b = i*m+j (U index), c = j*p+k (V index)
    T = np.zeros((pn, nm, mp), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    # Schoolbook decomposition (rank 63)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose T."""
        acc = np.zeros((pn, nm, mp), dtype=np.float64)
        for r in range(U.shape[1]):
            acc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(acc, T, atol=tol)

    def round_half(x):
        return np.round(x * 2.0) / 2.0

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to find lower-rank decomposition via ALS
    # The tensor is sparse: only 63 non-zeros out of 3969
    # Use structured initialization based on the sparsity pattern

    # Strategy: try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(pn, nm, mp) = 21, but realistically much higher
    # Try from 35 upward

    for target_rank in range(35, rank_sb):
        if time.time() - start_time > DEADLINE:
            break

        found_valid = False
        for trial in range(3):
            if time.time() - start_time > DEADLINE:
                break

            # Initialize using a structured approach
            # Start from schoolbook and randomly project to lower rank
            if target_rank <= rank_sb:
                # Random projection of schoolbook
                P = np.random.randn(rank_sb, target_rank) * 0.5
                U_t = U_sb @ P
                V_t = V_sb @ P
                W_t = W_sb @ P
            else:
                U_t = np.random.randn(nm, target_rank) * 0.3
                V_t = np.random.randn(mp, target_rank) * 0.3
                W_t = np.random.randn(pn, target_rank) * 0.3

            # ALS iterations
            for it in range(40):
                if time.time() - start_time > DEADLINE:
                    break

                # Update U: for each row i, T[:,i,:] = sum_r U[i,r] * outer(V[:,r], W[:,r])
                # Design matrix D_U: (pn*mp) x target_rank
                D_U = np.zeros((pn * mp, target_rank))
                for r in range(target_rank):
                    D_U[:, r] = np.outer(V_t[:, r], W_t[:, r]).flatten()
                for i in range(nm):
                    rhs = T[:, i, :].flatten()
                    U_t[i, :], _, _, _ = np.linalg.lstsq(D_U, rhs, rcond=None)

                # Update V: for each col j, T[:,:,j] transposed = sum_r V[j,r] * outer(W[:,r], U[:,r])
                D_V = np.zeros((pn * nm, target_rank))
                for r in range(target_rank):
                    D_V[:, r] = np.outer(W_t[:, r], U_t[:, r]).flatten()
                for j in range(mp):
                    # T[a, b, j] for all a,b -> reshape to (pn*nm,)
                    rhs = T[:, :, j].transpose(1, 0).flatten()
                    V_t[j, :], _, _, _ = np.linalg.lstsq(D_V, rhs, rcond=None)

                # Update W: for each row k, T[k,:,:] = sum_r W[k,r] * outer(U[:,r], V[:,r])
                D_W = np.zeros((nm * mp, target_rank))
                for r in range(target_rank):
                    D_W[:, r] = np.outer(U_t[:, r], V_t[:, r]).flatten()
                for k in range(pn):
                    rhs = T[k, :, :].flatten()
                    W_t[k, :], _, _, _ = np.linalg.lstsq(D_W, rhs, rcond=None)

            if time.time() - start_time > DEADLINE:
                break

            # Check if decomposition is exact
            if verify(U_t, V_t, W_t, tol=1e-9):
                # Try rounding to half-integers
                U_r = round_half(U_t)
                V_r = round_half(V_t)
                W_r = round_half(W_t)
                if verify(U_r, V_r, W_r, tol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found_valid = True
                    break
                else:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_t, V_t, W_t
                    found_valid = True
                    break

        if found_valid:
            break

    # Safety: verify our best result
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
