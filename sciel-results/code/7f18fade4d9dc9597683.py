"""Combined program for 3x3 x 3x7 matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    hard_limit = 145.0  # Keep below 150s evaluator kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Precompute the target tensor T_target with shape (n*m, m*p, p*n)
    # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k), else 0
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Fallback schoolbook decomposition
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1

    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)

    def compute_tensor(U, V, W):
        """Compute the tensor from U, V, W using einsum."""
        return np.einsum('ad,bd,cd->abc', U, V, W)

    def objective_and_grad(params, n, m, p, r):
        """Objective function and analytic gradient for L-BFGS-B."""
        U = params[0:r*n*m].reshape(n*m, r)
        V = params[r*n*m:2*r*n*m + r*m*p].reshape(m*p, r)
        W = params[2*r*n*m + r*m*p:].reshape(p*n, r)

        T = compute_tensor(U, V, W)
        diff = T - T_target
        obj = np.sum(diff * diff)

        # Analytic gradients
        # d/dU[a,d] = 2 * sum_{b,c} diff[a,b,c] * V[b,d] * W[c,d]
        # We can compute this efficiently:
        # grad_U = 2 * np.einsum('abc,bd,cd->ad', diff, V, W)
        # grad_V = 2 * np.einsum('abc,ad,cd->bd', diff, U, W)
        # grad_W = 2 * np.einsum('abc,ad,bd->cd', diff, U, V)

        grad_U = 2.0 * np.einsum('abc,bd,cd->ad', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ad,cd->bd', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ad,bd->cd', diff, U, V)

        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad

    def try_rank(r, max_restarts=5, max_iter=200):
        """Try to find a valid decomposition of rank r."""
        if time.time() - start_time > hard_limit - 10:
            return None

        for restart in range(max_restarts):
            if time.time() - start_time > hard_limit - 5:
                return None

            # Random initialization
            np.random.seed(restart * 1000 + r)
            scale = 1.0 / np.sqrt(r)
            U0 = np.random.randn(n*m, r) * scale
            V0 = np.random.randn(m*p, r) * scale
            W0 = np.random.randn(p*n, r) * scale

            # Add a bias to help break symmetry
            U0 += 0.5
            V0 += 0.5
            W0 += 0.5

            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            try:
                from scipy.optimize import minimize
                res = minimize(objective_and_grad, params0, args=(n, m, p, r),
                              method='L-BFGS-B', jac=True,
                              options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})

                obj_val = res.fun
                if obj_val < 1e-8:
                    params_opt = res.x
                    U_opt = params_opt[0:r*n*m].reshape(n*m, r)
                    V_opt = params_opt[r*n*m:2*r*n*m + r*m*p].reshape(m*p, r)
                    W_opt = params_opt[2*r*n*m + r*m*p:].reshape(p*n, r)

                    # Try to snap to integer/half-integer values
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2

                    # Check if snapped values are close enough
                    if (np.allclose(U_opt, U_snap, atol=1e-4) and
                        np.allclose(V_opt, V_snap, atol=1e-4) and
                        np.allclose(W_opt, W_snap, atol=1e-4)):
                        U_opt, V_opt, W_opt = U_snap, V_snap, W_snap

                    # Verify exactness
                    T_check = compute_tensor(U_opt, V_opt, W_opt)
                    if np.allclose(T_check, T_target, atol=1e-6):
                        return (U_opt, V_opt, W_opt)

                    # If not snapped, check if the un-snapped solution is valid
                    # (but we need integer/half-integer entries)
                    # So we must snap and verify

            except Exception:
                pass

        return None

    # Try ranks from 62 down to 40
    candidate_ranks = list(range(62, 39, -1))

    for r in candidate_ranks:
        if time.time() - start_time > hard_limit - 10:
            break

        solution = try_rank(r)
        if solution is not None:
            best_rank = r
            best_solution = solution
            break  # Found a valid lower rank, stop

    U_res, V_res, W_res = best_solution

    def to_list(arr):
        return arr.tolist()

    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
# EVOLVE-BLOCK-END
