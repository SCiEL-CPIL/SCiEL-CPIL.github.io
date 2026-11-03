import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10

    # Build the target tensor M of shape (dim_u, dim_v, dim_w)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    # Fallback: Schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify_solution(U, V, W, tol=1e-12):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff = np.max(np.abs(T - target))
        return max_diff < tol

    def snap_and_verify(U, V, W):
        """Snap to nearest 0.5 and verify."""
        U_s = np.round(U * 2.0) / 2.0
        V_s = np.round(V * 2.0) / 2.0
        W_s = np.round(W * 2.0) / 2.0
        if verify_solution(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    # ALS solvers - solve for one factor given the other two
    def solve_U(V, W, R):
        U = np.zeros((dim_u, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return U

    def solve_V(U, W, R):
        V = np.zeros((dim_v, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return V

    def solve_W(U, V, R):
        W = np.zeros((dim_w, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return W

    def check_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        # Also check zero entries
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = max(max_err, np.max(np.abs(T - target)))
        return max_err

    def polish(U, V, W, R, max_iter=100, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            U = solve_U(V, W, R)
            V = solve_V(U, W, R)
            W = solve_W(U, V, R)

            err = check_error(U, V, W)

            if err < tol:
                return U, V, W, err

            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err

        return U, V, W, err

    def try_rank_als(R, n_restarts=5, max_iter=200, tol=1e-9):
        """Try to find a rank-R decomposition using ALS."""
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                return None

            np.random.seed(12345 + R * 100 + restart)

            # Different initialization strategies
            if restart == 0:
                # Random small
                U = np.random.randn(dim_u, R) * 0.1
                V = np.random.randn(dim_v, R) * 0.1
                W = np.random.randn(dim_w, R) * 0.1
            elif restart == 1:
                # Medium random
                U = np.random.randn(dim_u, R) * 0.3
                V = np.random.randn(dim_v, R) * 0.3
                W = np.random.randn(dim_w, R) * 0.3
            elif restart == 2:
                # Sparse random
                U = np.random.randn(dim_u, R) * 0.5
                V = np.random.randn(dim_v, R) * 0.5
                W = np.random.randn(dim_w, R) * 0.5
                U[np.random.rand(dim_u, R) < 0.3] = 0
                V[np.random.rand(dim_v, R) < 0.3] = 0
                W[np.random.rand(dim_w, R) < 0.3] = 0
            else:
                # Very small
                U = np.random.randn(dim_u, R) * 0.05
                V = np.random.randn(dim_v, R) * 0.05
                W = np.random.randn(dim_w, R) * 0.05

            # ALS iterations
            prev_err = np.inf
            converged = False
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break

                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)

                err = check_error(U, V, W)

                if err < tol:
                    converged = True
                    break

                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err

            # If converged or close, try snap and verify
            if converged or err < 1e-6:
                result = snap_and_verify(U, V, W)
                if result is not None:
                    return result

                # Try polish
                if time.time() < deadline - 20:
                    U_p, V_p, W_p, final_err = polish(U, V, W, R, max_iter=50, tol=1e-12)
                    if final_err < 1e-9:
                        result = snap_and_verify(U_p, V_p, W_p)
                        if result is not None:
                            return result

        return None

    def try_rank_lbgfs(R, n_restarts=10, max_iter=500):
        """Try to find a rank-R decomposition using L-BFGS-B."""
        from scipy.optimize import minimize

        def loss_and_grad(params):
            u_flat = params[:dim_u * R]
            v_flat = params[dim_u * R: dim_u * R + dim_v * R]
            w_flat = params[dim_u * R + dim_v * R:]

            U = u_flat.reshape(dim_u, R)
            V = v_flat.reshape(dim_v, R)
            W = w_flat.reshape(dim_w, R)

            T = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - target
            loss = np.sum(diff * diff)

            grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)

            grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
            return loss, grad

        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                return None

            np.random.seed(54321 + R * 100 + restart)
            scale = 0.1 if restart < 3 else (0.2 if restart < 6 else 0.3)
            u0 = np.random.randn(dim_u * R) * scale
            v0 = np.random.randn(dim_v * R) * scale
            w0 = np.random.randn(dim_w * R) * scale
            params = np.concatenate([u0, v0, w0])

            try:
                res = minimize(loss_and_grad, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})

                if res.fun < 1e-10:
                    u_final = res.x[:dim_u * R].reshape(dim_u, R)
                    v_final = res.x[dim_u * R: dim_u * R + dim_v * R].reshape(dim_v, R)
                    w_final = res.x[dim_u * R + dim_v * R:].reshape(dim_w, R)

                    result = snap_and_verify(u_final, v_final, w_final)
                    if result is not None:
                        return result

                    # Try polish
                    if time.time() < deadline - 20:
                        U_p, V_p, W_p, final_err = polish(u_final, v_final, w_final, R, max_iter=50, tol=1e-12)
                        if final_err < 1e-9:
                            result = snap_and_verify(U_p, V_p, W_p)
                            if result is not None:
                                return result

            except Exception:
                continue

        return None

    # Main search: try ranks from 49 down to 30
    # Use both ALS and L-BFGS for each rank
    best_result = None

    for target_rank in range(49, 29, -1):
        if time.time() > deadline - 10:
            break

        # Try ALS first (faster per restart)
        result = try_rank_als(target_rank, n_restarts=5, max_iter=200, tol=1e-9)
        if result is not None:
            best_result = result
            break

        # Try L-BFGS if time permits
        if time.time() < deadline - 30:
            result = try_rank_lbgfs(target_rank, n_restarts=10, max_iter=500)
            if result is not None:
                best_result = result
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
