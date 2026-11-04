"""Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication.

Uses numerical ALS optimization to find low-rank decompositions.
Index conventions:
  U has shape (n*m, rank) = (12, rank), indexed by [i*m + j]
  V has shape (m*p, rank) = (24, rank), indexed by [j*p + k]
  W has shape (p*n, rank) = (18, rank), indexed by [k*n + i]

Tensor: T[i*m+j, j'*p+k', k''*n+i''] = 1 iff j==j' and i==i' and k==k'
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 115  # safety margin before 120s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=tol)

    def round_to_half(U, V, W):
        """Round to integers/half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-8):
            return U_r, V_r, W_r
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_step(U, V, W, rank, T_A, T_B, T_C, n, m, p):
        """One ALS iteration. Returns updated U, V, W."""
        # Update U: solve for U given V, W
        MW = np.einsum('br,cr->bcr', V, W)
        MW_flat = MW.transpose(2, 0, 1).reshape(rank, m * p * p * n)
        G = MW_flat @ MW_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            U_new = T_A @ MW_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U, V, W

        # Update V: solve for V given U, W
        UW = np.einsum('ar,cr->acr', U_new, W)
        UW_flat = UW.transpose(2, 0, 1).reshape(rank, n * m * p * n)
        G = UW_flat @ UW_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            V_new = T_B @ UW_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U_new, V, W

        # Update W: solve for W given U, V
        UV = np.einsum('ar,br->abr', U_new, V_new)
        UV_flat = UV.transpose(2, 0, 1).reshape(rank, n * m * m * p)
        G = UV_flat @ UV_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            W_new = T_C @ UV_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U_new, V_new, W

        return U_new, V_new, W_new

    def compute_residual(U, V, W):
        """Compute the residual norm."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_computed - T)

    # Try to find lower-rank decompositions
    # Start with promising ranks based on known bounds
    target_ranks = [42, 48, 54, 60, 66]

    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        # Multiple random initializations with different strategies
        for seed in range(10):
            if time.time() > deadline:
                break

            np.random.seed(seed * 1000 + 42)

            # Strategy 1: Random Gaussian
            if seed % 3 == 0:
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            # Strategy 2: Sparse-like initialization
            elif seed % 3 == 1:
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Pick random positions
                    ui = np.random.randint(0, n * m)
                    vj = np.random.randint(0, m * p)
                    wk = np.random.randint(0, p * n)
                    U_init[ui, r] = 1.0
                    V_init[vj, r] = 1.0
                    W_init[wk, r] = 1.0
            # Strategy 3: Structured from schoolbook (perturbed)
            else:
                U_init = np.random.randn(n * m, target_rank) * 0.5
                V_init = np.random.randn(m * p, target_rank) * 0.5
                W_init = np.random.randn(p * n, target_rank) * 0.5

            # Run ALS with convergence checking
            U_cur, V_cur, W_cur = U_init, V_init, W_init
            prev_res = np.inf
            converged = False

            for iteration in range(150):
                if time.time() > deadline:
                    break

                U_cur, V_cur, W_cur = als_step(
                    U_cur, V_cur, W_cur, target_rank, T_A, T_B, T_C, n, m, p
                )

                res = compute_residual(U_cur, V_cur, W_cur)

                # Check convergence
                if prev_res - res < 1e-14:
                    converged = True
                    break
                prev_res = res

                # Early stop if residual is very small
                if res < 1e-10:
                    converged = True
                    break

            # Final verification and rounding
            if converged or res < 1e-8:
                rounded = round_to_half(U_cur, V_cur, W_cur)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if verify(U_r, V_r, W_r, tol=1e-8):
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        break  # Found valid decomposition for this rank

        if best_rank < rank_sch:
            # Found improvement, try to go lower
            continue
        else:
            # Didn't find anything at this rank, move to next
            continue

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
