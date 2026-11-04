"""Matrix multiplication 3,3,5 - low rank decomposition via structured search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 3x3 x 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build the target multiplication tensor
    # Shape: (n*m, m*p, p*n) = (9, 15, 15)
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition(U, V, W, target_tensor, tol=1e-9):
        """Verify that U, V, W decompose the target tensor."""
        rank = U.shape[1]
        reconstructed = np.zeros_like(target_tensor)
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, target_tensor, atol=tol)

    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1

        return U, V, W, rank

    def try_rank_als(rank, max_steps=200, tol=1e-7, seed=42):
        """
        Use ALS to find a rank-r decomposition.
        Returns (U, V, W) if successful, None otherwise.
        """
        if time.time() > deadline - 10:
            return None

        rng = np.random.RandomState(seed)

        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3

        prev_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # --- Update U ---
            # For each a, target[a,:,:].flatten() = sum_r U[a,r] * (V[:,r] ⊗ W[:,r]).flatten()
            # So: target.reshape(n*m, m*p*p*n) = U @ M_U^T where M_U[r, :] = (V[:,r] ⊗ W[:,r]).flatten()
            M_U = np.zeros((rank, m * p * p * n))
            for r in range(rank):
                M_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()

            T_flat = target.reshape(n * m, m * p * p * n)
            # T_flat = U @ M_U^T, so U = T_flat @ M_U^T (M_U @ M_U^T)^{-1}
            try:
                # Solve: M_U @ U.T = T_flat.T
                U.T = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0]
            except:
                return None

            # --- Update V ---
            M_V = np.zeros((rank, n * m * p * n))
            for r in range(rank):
                M_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()

            T_flat_v = np.zeros((m * p, n * m * p * n))
            for b in range(m * p):
                T_flat_v[b, :] = target[:, b, :].flatten()

            try:
                V.T = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)[0]
            except:
                return None

            # --- Update W ---
            M_W = np.zeros((rank, n * m * m * p))
            for r in range(rank):
                M_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()

            T_flat_w = np.zeros((p * n, n * m * m * p))
            for c in range(p * n):
                T_flat_w[c, :] = target[:, :, c].flatten()

            try:
                W.T = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)[0]
            except:
                return None

            # Compute error periodically
            if step % 5 == 0 or step == max_steps - 1:
                recon = np.zeros_like(target)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(recon - target)

                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < prev_err - 1e-6:
                    prev_err = err
                elif err < tol:
                    break
                elif prev_err - err < 1e-10 and step > 30:
                    break

        # Use best found
        U, V, W = best_U, best_V, best_W

        # Verify
        if verify_decomposition(U, V, W, target, tol=1e-5):
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify_decomposition(U_r, V_r, W_r, target, tol=1e-5):
                return U_r, V_r, W_r

            # Try rounding to integers
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if verify_decomposition(U_r, V_r, W_r, target, tol=1e-5):
                return U_r, V_r, W_r

            # Return as-is
            return U, V, W

        return None

    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # Try to find better decompositions
    # The lower bound is at least 27 (n*m*p for 3x3x3 is 27, and 3x3x5 should be at least that)
    # Try ranks from 27 upward
    for rank in range(27, best_rank):
        if time.time() > deadline - 15:
            break

        # Try multiple seeds for this rank
        found = False
        for seed in [42 + rank, 100 + rank, 200 + rank, 300 + rank, 500 + rank]:
            if time.time() > deadline - 15:
                break

            result = try_rank_als(rank, max_steps=150, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break

        if found:
            break  # Found a valid decomposition at this rank

    U, V, W = best_result

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
