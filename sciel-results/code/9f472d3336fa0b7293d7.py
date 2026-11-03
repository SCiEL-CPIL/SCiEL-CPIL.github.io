"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill

    # 1. Build the exact schoolbook (rank n*m*p) tensor
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = nm * mp * pn
    U0 = np.zeros((nm, rank_sb))
    V0 = np.zeros((mp, rank_sb))
    W0 = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    def tensor_residual(U, V, W):
        # Compute ||U*V^T (elementwise W^T)||_F^2 as sum over c of || (U_c * V_c) - W_c * T ||^2
        # Actually the tensor T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
        # Residual = sum_{a,b,c} (T_{a,b,c} - sum_r U[a,r]V[b,r]W[c,r])^2
        # = ||T||^2 - 2*sum_r <T, U_r V_r W_r> + sum_r ||U_r V_r W_r||^2
        # For speed, we just compute the full tensor for small sizes
        T = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        # The target tensor
        Tt = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Tt[i * m + j, j * p + k, k * n + i] = 1
        return np.sum((T - Tt) ** 2)

    def round_to_halfint(A):
        return np.round(A * 2) / 2.0

    def verify(U, V, W):
        """Check exact decomposition. Returns True if residual is 0 (within tiny tol)."""
        T = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        Tt = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Tt[i * m + j, j * p + k, k * n + i] = 1
        return np.max(np.abs(T - Tt)) < 1e-9

    def try_rank(r, max_iters=2000, tol=1e-12):
        """Try to find rank-r decomposition via ALS + rounding."""
        # Initialize with random matrices
        rng = np.random.RandomState(42)
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None

        for trial in range(3):
            if time.time() > deadline:
                break
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            W = rng.randn(pn, r)
            # Scale
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)

            prev_res = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                # U[:,r] should minimize sum_c ||W[c,r] * T[:, :, c] - U[:,r] * (V[:,r] * W[:,r])||^2
                # More precisely: for each r, U[:,r] = (sum_c W[c,r]^2 * (T[:,:,c] * V[:,r])) / sum_c W[c,r]^2 * V[:,r]^2
                # This is getting complex. Let's use a simpler approach:
                # The bilinear form: T(a,b,c) = sum_r U(a,r) V(b,r) W(c,r)
                # Fix V, W: U(a,r) = <T[:,b,c] * V(b,r) * W(c,r), 1> / sum_{b,c} V(b,r)^2 W(c,r)^2
                # Actually: minimize sum_{a,b,c} (T(a,b,c) - sum_s U(a,s)V(b,s)W(c,s))^2 over U
                # = sum_{a,r} [sum_{b,c} V(b,r)W(c,r) T(a,b,c) - U(a,r) sum_{b,c} V(b,r)^2 W(c,r)^2]^2 / ...
                # Optimal U(a,r) = sum_{b,c} V(b,r) W(c,r) T(a,b,c) / sum_{b,c} V(b,r)^2 W(c,r)^2

                # Precompute for each r: VR = V[:,r], WR = W[:,r]
                # U[:,r] = (VR * WR) contracted with T over (b,c)
                # T is (nm, mp, pn)
                # For each r: U[:,r] = einsum('b,c,abc->a', VR, WR, T) / (VR**2 * WR**2).sum()

                for r in range(r):
                    VR = V[:, r]
                    WR = W[:, r]
                    num = np.einsum('b,c,abc->a', VR, WR, Tt)  # Use target T
                    den = np.sum(VR ** 2 * WR ** 2)
                    if den > 1e-15:
                        U[:, r] = num / den

                # Update V similarly
                for r in range(r):
                    UR = U[:, r]
                    WR = W[:, r]
                    num = np.einsum('a,c,abc->b', UR, WR, Tt)
                    den = np.sum(UR ** 2 * WR ** 2)
                    if den > 1e-15:
                        V[:, r] = num / den

                # Update W
                for r in range(r):
                    UR = U[:, r]
                    VR = V[:, r]
                    num = np.einsum('a,b,abc->c', UR, VR, Tt)
                    den = np.sum(UR ** 2 * VR ** 2)
                    if den > 1e-15:
                        W[:, r] = num / den

                # Compute residual
                if it % 50 == 0 or it == max_iters - 1:
                    T_approx = np.zeros((nm, mp, pn))
                    for rr in range(r):
                        T_approx += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
                    res = np.sum((T_approx - Tt) ** 2)
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if res < tol:
                        break
                    if abs(prev_res - res) < tol * 0.1:
                        break
                    prev_res = res

            # Try rounding the best found
            if best_U is not None:
                U_r = round_to_halfint(best_U)
                V_r = round_to_halfint(best_V)
                W_r = round_to_halfint(best_W)
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r

        return None

    # Try ranks from low to high
    # Known: Strassen for 4x4x4 is rank 49 or so. For 4x4x5, let's try from 40 up.
    # Actually, let's start with a reasonable lower bound.
    # The trivial bound is 80. Strassen-like could be lower.
    # Let's try a range of ranks.

    for r in range(40, 81):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: return schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
