"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def loss_and_grads(x, target_rank):
        """Compute loss and gradients for L-BFGS-B."""
        nm = n * m
        mp = m * p
        pn = p * n

        U = x[:nm*target_rank].reshape(n, m, target_rank)
        V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(m, p, target_rank)
        W = x[nm*target_rank+mp*target_rank:].reshape(p, n, target_rank)

        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = T - 1.0
        loss = np.sum(diff**2)

        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U, V)

        grads = np.concatenate([
            (2.0 * grad_U).flatten(),
            (2.0 * grad_V).flatten(),
            (2.0 * grad_W).flatten()
        ])

        return loss, grads

    def try_rank_optimization(target_rank, num_trials=5, max_iter=300):
        """Try to find a decomposition of given rank using L-BFGS-B."""
        if time.time() > deadline:
            return None

        nm = n * m
        mp = m * p
        pn = p * n
        total_dim = (nm + mp + pn) * target_rank

        best_result = None
        best_loss = float('inf')

        for trial in range(num_trials):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if trial == 0:
                x0 = np.random.randn(total_dim) * 0.1
            elif trial == 1:
                x0 = np.random.randn(total_dim) * 0.3
            elif trial == 2:
                x0 = np.random.randn(total_dim) * 0.5
            elif trial == 3:
                # Sparse init
                x0 = np.random.randn(total_dim) * 0.2
                x0[np.random.rand(total_dim) < 0.7] = 0
            else:
                x0 = np.random.randn(total_dim) * 0.4

            try:
                result = minimize(
                    loss_and_grads,
                    x0,
                    args=(target_rank,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
            except Exception:
                continue

            if time.time() > deadline:
                break

            if result.fun < best_loss:
                best_loss = result.fun
                x_best = result.x

                U = x_best[:nm*target_rank].reshape(n, m, target_rank)
                V = x_best[nm*target_rank:nm*target_rank+mp*target_rank].reshape(m, p, target_rank)
                W = x_best[nm*target_rank+mp*target_rank:].reshape(p, n, target_rank)

                U_flat = U.flatten()
                V_flat = V.flatten()
                W_flat = W.flatten()

                # Try rounding to half-integers
                U_rounded = np.round(2 * U_flat) / 2.0
                V_rounded = np.round(2 * V_flat) / 2.0
                W_rounded = np.round(2 * W_flat) / 2.0

                U_r = U_rounded.reshape(n, m, target_rank)
                V_r = V_rounded.reshape(m, p, target_rank)
                W_r = W_rounded.reshape(p, n, target_rank)

                T_check = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                if np.allclose(T_check, np.ones((n, m, p)), atol=1e-6):
                    return (U_r.flatten().reshape(nm, target_rank),
                            V_r.flatten().reshape(mp, target_rank),
                            W_r.flatten().reshape(pn, target_rank))

                # Try integer rounding
                U_int = np.round(U_flat)
                V_int = np.round(V_flat)
                W_int = np.round(W_flat)

                U_ir = U_int.reshape(n, m, target_rank)
                V_ir = V_int.reshape(m, p, target_rank)
                W_ir = W_int.reshape(p, n, target_rank)

                T_check_int = np.einsum('ijc,jkc,kic->ijk', U_ir, V_ir, W_ir)
                if np.allclose(T_check_int, np.ones((n, m, p)), atol=1e-6):
                    return (U_ir.flatten().reshape(nm, target_rank),
                            V_ir.flatten().reshape(mp, target_rank),
                            W_ir.flatten().reshape(pn, target_rank))

                best_result = (U.flatten().reshape(nm, target_rank),
                               V.flatten().reshape(mp, target_rank),
                               W.flatten().reshape(pn, target_rank))

        # Check if best continuous solution is valid
        if best_result is not None:
            U, V, W = best_result
            if check_decomposition(U, V, W, tol=1e-4):
                return best_result

        return None

    # Try various ranks - start from lower and go up
    # Lower bound is at least max(n*m, m*p, p*n) = 25 for 3x5x5
    # Known results suggest rank around 25-35 for this size
    ranks_to_try = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break

        result = try_rank_optimization(target_rank, num_trials=3, max_iter=200)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found valid decomposition at this rank

    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
