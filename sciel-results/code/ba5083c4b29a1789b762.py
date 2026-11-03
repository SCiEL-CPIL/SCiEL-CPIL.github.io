"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False

    valid = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True

    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False

    return True


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Schoolbook fallback
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
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Strategy: For each candidate rank r, the decomposition is equivalent to
    # finding a rank-r tensor completion of a partially-specified tensor.
    #
    # We use an alternating least-squares (ALS) approach:
    #   - Fix V and W, solve for U (linear)
    #   - Fix U and W, solve for V (linear)
    #   - Fix U and V, solve for W (linear)
    #
    # Each step is a least-squares problem. We iterate until convergence or
    # the objective drops below tolerance.
    #
    # To make this tractable, we exploit the fact that the target tensor has
    # only n*m*p = 63 non-zero entries. We can set up the system using only
    # those entries.

    # For each rank r, we have r unknowns per row of U, V, W.
    # The constraint is: for each (a,b,c) with T_target[a,b,c]=1,
    #   sum_d U[a,d]*V[b,d]*W[c,d] = 1
    # And for all other (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 0
    #
    # This is a system of nonlinear equations. ALS linearizes each step.

    # For a given rank r and fixed V, W:
    #   For each row a: U[a,:] * (V[b,:] .* W[c,:])^T = T_target[a,b,c]
    #   This is a linear system in U[a,:].

    # We'll use a different approach: direct nonlinear optimization with
    # scipy's trust-region or L-BFGS-B, but with better initialization.

    from scipy.optimize import minimize

    # The key insight for 3x3x7: we can try to find a decomposition where
    # the rank is at most 3*m*p or similar structured bound.
    # Actually, the schoolbook is 27 (3*3*7/3 = 21? No, 3*3*7 = 63, but
    # with the structure it's n*m*p = 63... wait no.
    # For matrix multiplication (n x m) * (m x p), the schoolbook rank is n*m*p.
    # But with the standard indexing, it's actually n*m*p terms.
    # 3*3*7 = 63.
    #
    # However, we can do better. The tensor has a special structure.
    # For 3x3x3, the best known rank is 20 (Strassen-like).
    # For 3x3x7, we can use 3 copies of 3x3x3 (rank 3*20=60) or find something better.
    #
    # Actually, a better bound: we can multiply 3x3 by 3x7 as:
    #   [A1 A2 A3] where each Ai is 3x3, and the result is 3x7.
    #   This is 3 independent 3x3 multiplications, so rank <= 3 * R(3,3,3) = 3*20 = 60.
    #   But schoolbook is 63, so this is only marginally better.
    #
    # A better approach: use the fact that we're multiplying by a 3x7 matrix.
    # We can view this as 3x3 * 3x7 = (3x3 * 3x3) stacked with (3x3 * 3x4) etc.
    #
    # Let's just try to find any rank < 63 using optimization.

    # Try ranks from 20 upward
    rank_targets = list(range(20, 64))

    # For each target rank, try multiple random restarts with ALS
    for target_rank in rank_targets:
        if time.time() > deadline - 20.0:
            break
        if target_rank >= best_rank:
            break

        max_restarts = 5
        found = False

        for restart in range(max_restarts):
            if time.time() > deadline - 15.0:
                break

            # Random initialization
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5

            # ALS iterations
            def compute_objective(U, V, W):
                T = np.einsum('ad,bd,cd->abc', U, V, W)
                return np.sum((T - T_target) ** 2)

            obj = compute_objective(U, V, W)
            if obj < 1e-10:
                found = True
                break

            for iteration in range(200):
                if time.time() > deadline - 10.0:
                    break

                # Update U: for each row a, solve linear system
                # For each (a, b, c) with T_target[a,b,c] != 0:
                #   U[a,:] . (V[b,:] * W[c,:]) = T_target[a,b,c]
                # Plus constraints for zero entries.
                #
                # Actually, let's do a proper least-squares update.
                # For row a of U, we need:
                #   sum over (b,c) in S_a of U[a,:] . (V[b,:] * W[c,:]) * w_bc = obj_a
                # where S_a is the set of (b,c) pairs.
                #
                # This is getting complex. Let's use a simpler approach:
                # Just do gradient descent on the objective.

                # Compute gradient
                # d_obj/dU[a,d] = 2 * sum_{b,c} (T[a,b,c] - T_target[a,b,c]) * V[b,d] * W[c,d]
                T = np.einsum('ad,bd,cd->abc', U, V, W)
                residual = T - T_target

                dU = 2.0 * np.einsum('abc,bd,cd->ad', residual, V, W)
                dV = 2.0 * np.einsum('abc,ad,cd->bd', residual, U, W)
                dW = 2.0 * np.einsum('abc,ad,bd->cd', residual, U, V)

                # Gradient step with adaptive learning rate
                lr = 0.01
                # Scale learning rate by the norm of the gradient
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 0:
                    lr = lr / grad_norm

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                new_obj = compute_objective(U, V, W)
                if new_obj > obj * 1.01:
                    # Bad step, revert
                    U += lr * dU
                    V += lr * dV
                    W += lr * dW
                    lr *= 0.5
                else:
                    obj = new_obj
                    if obj < 1e-10:
                        found = True
                        break

                if obj < 1e-8:
                    found = True
                    break

            if found:
                break

        if found:
            # Try to snap to half-integers
            def snap(val):
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-4:
                    return nearest
                return val

            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]

            if _verify(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break

            U_snap = [[snap(v) for v in row] for row in U_list]
            V_snap = [[snap(v) for v in row] for row in V_list]
            W_snap = [[snap(v) for v in row] for row in W_list]

            if _verify(U_snap, V_snap, W_snap, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
