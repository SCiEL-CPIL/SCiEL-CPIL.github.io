import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10

    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol, rtol=0)

    def build_strassen_based(rank=46):
        """
        Build a rank-46 solution using Strassen's algorithm for 2x2x2 blocks
        plus schoolbook for the remaining parts.
        
        Structure:
        - A is 2x5, B is 5x5, C is 2x5
        - Split B into 4 blocks of 2x2 (covering cols 0-3) and 1 block of 1x1 (col 4)
        - For each 2x2 block of B, use Strassen (7 mults) for the 2x2x2 multiplication
        - For the 1x1 block, use schoolbook (2 mults)
        - Plus handle the last column of A (j=4) with schoolbook
        """
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(j_off, k_off, col_start):
            """Add 7 Strassen terms for a 2x2x2 block at A[:, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]"""
            # P1 = (A00 + A11)(B00 + B11)
            c = col_start
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

            return c

        # Use Strassen for 2x2 blocks of B covering columns 0-3 (j_off in {0, 2}, k_off in {0, 2})
        # A[:, 0:2] * B[0:2, 0:2]
        col = add_strassen(0, 0, col)
        # A[:, 0:2] * B[0:2, 2:4]
        col = add_strassen(0, 2, col)
        # A[:, 2:4] * B[2:4, 0:2]
        col = add_strassen(2, 0, col)
        # A[:, 2:4] * B[2:4, 2:4]
        col = add_strassen(2, 2, col)

        # Now handle j=4 (last column of A) with schoolbook
        # A[:, 4] * B[4, 0:5] -> contributes to C[:, 0:5]
        for k in range(p):
            U[0 * m + 4][col] = 1
            U[1 * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1

        # Handle j in {0,1,2,3} with k=4 (last column of B)
        for j in range(4):
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        return U[:, :col], V[:, :col], W[:, :col]

    # Build the rank-46 solution
    U46, V46, W46 = build_strassen_based(46)
    actual_rank = U46.shape[1]

    best_result = {
        "U": U46.tolist(),
        "V": V46.tolist(),
        "W": W46.tolist()
    }
    best_rank = actual_rank

    # Verify the rank-46 solution
    if not verify(U46, V46, W46, tol=1e-9):
        # Fallback to schoolbook if Strassen construction has a bug
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_result = {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
        best_rank = rank_sb
        return best_result

    deadline = time.time() + 130  # Safety margin before 150s limit

    def try_rank_optimization(target_rank, U_init, V_init, W_init, max_iters=30):
        """
        Try to find a rank-target_rank decomposition starting from an initial guess.
        Uses gradient descent with adaptive learning rate.
        """
        for restart in range(max_iters):
            if time.time() > deadline - 15:
                return None

            np.random.seed(42 + restart * 7)

            # Start from the initial guess with perturbation
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()

            # If we need fewer terms, randomly drop some and re-optimize
            if target_rank < U.shape[1]:
                idx = np.random.choice(U.shape[1], size=target_rank, replace=False)
                U = U[:, idx].copy()
                V = V[:, idx].copy()
                W = W[:, idx].copy()
            elif target_rank > U.shape[1]:
                # Add random extra terms
                extra = target_rank - U.shape[1]
                U_extra = np.random.randn(nm, extra) * 0.1
                V_extra = np.random.randn(mp, extra) * 0.1
                W_extra = np.random.randn(pn, extra) * 0.1
                U = np.hstack([U, U_extra])
                V = np.hstack([V, V_extra])
                W = np.hstack([W, W_extra])

            # Add small random perturbation
            U += np.random.randn(nm, target_rank) * 0.05
            V += np.random.randn(mp, target_rank) * 0.05
            W += np.random.randn(pn, target_rank) * 0.05

            # Gradient descent
            lr = 0.005
            for step in range(200):
                if time.time() > deadline - 10:
                    return None

                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - T_approx
                res = np.sum(R ** 2)

                if res < 1e-14:
                    break

                # Gradients
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)

                # Adaptive step
                grad_norm = np.sqrt(np.sum(grad_U ** 2) + np.sum(grad_V ** 2) + np.sum(grad_W ** 2))
                if grad_norm < 1e-16:
                    break

                # Line search-like adaptation
                U_new = U - lr * grad_U
                V_new = V - lr * grad_V
                W_new = W - lr * grad_W

                T_approx_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                res_new = np.sum((T - T_approx_new) ** 2)

                if res_new < res:
                    U, V, W = U_new, V_new, W_new
                    lr *= 1.1
                else:
                    lr *= 0.7

                if lr < 1e-6:
                    break

            # Try to snap to rational values
            for scale in [1, 2, 4, 8]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                if verify(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r

            # Check without snapping
            if verify(U, V, W, tol=1e-9):
                return U, V, W

        return None

    # Try to find rank 45 or lower
    # Start from the rank-46 solution and try to remove one term
    for target_rank in range(best_rank - 1, max(30, best_rank - 6), -1):
        if time.time() > deadline - 25:
            break

        # Try removing each term from the current best
        for remove_idx in range(best_rank):
            if time.time() > deadline - 20:
                break

            idx_keep = [i for i in range(best_rank) if i != remove_idx]
            U_init = U46[:, idx_keep]
            V_init = V46[:, idx_keep]
            W_init = W46[:, idx_keep]

            result = try_rank_optimization(target_rank, U_init, V_init, W_init, max_iters=5)
            if result is not None:
                U_r, V_r, W_r = result
                if verify(U_r, V_r, W_r):
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    best_rank = target_rank
                    # Update the base for further reductions
                    U46 = U_r
                    V46 = V_r
                    W46 = W_r
                    break

    return best_result
