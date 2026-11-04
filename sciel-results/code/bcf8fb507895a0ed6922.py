import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1

    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank_found = schoolbook_rank

    def check_exact(U, V, W):
        """Check if U,V,W exactly decompose T"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-10)

    def try_integer_rounding(U, V, W):
        """Try to round to integers or half-integers and check exactness"""
        # Try integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_exact(U_int, V_int, W_int):
            return U_int, V_int, W_int

        # Try half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_exact(U_half, V_half, W_half):
            return U_half, V_half, W_half

        return None

    def objective_and_grad(params, target_rank):
        """Objective and gradient for L-BFGS-B"""
        U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)

        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        diff = T - T_approx
        loss = np.sum(diff**2)

        grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
        grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
        grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)

        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    # Lower bound is max(n*m, m*p, p*n) = 15
    lower_bound = max(n*m, m*p, p*n)

    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break

        # Increase restarts for lower ranks where solutions are rarer but more valuable
        max_restarts = 20 if target_rank < 25 else 10

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Choose initialization strategy
            if restart == 0:
                # Structured initialization: seed with schoolbook columns
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for idx in range(min(target_rank, schoolbook_rank)):
                    U_init[:, idx] = U_sch[:, idx]
                    V_init[:, idx] = V_sch[:, idx]
                    W_init[:, idx] = W_sch[:, idx]
                # Add small random noise to break symmetry
                U_init += np.random.randn(*U_init.shape) * 0.1
                V_init += np.random.randn(*V_init.shape) * 0.1
                W_init += np.random.randn(*W_init.shape) * 0.1
            elif restart == 1:
                # Slightly larger random init
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            else:
                # Pure random
                U_init = np.random.randn(n*m, target_rank) * 0.3
                V_init = np.random.randn(m*p, target_rank) * 0.3
                W_init = np.random.randn(p*n, target_rank) * 0.3

            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])

            # Use L-BFGS-B for smooth optimization
            try:
                result = minimize(objective_and_grad, params_init, method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 500, 'maxfun': 2000, 'gtol': 1e-7})

                if result.fun < 1e-8:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)

                    rounded = try_integer_rounding(U_r, V_r, W_r)
                    if rounded is not None:
                        return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}
                        
                    # If rounding fails, try a few perturbations and re-optimize
                    for _ in range(5):
                         U_p = U_r + np.random.randn(*U_r.shape) * 0.01
                         V_p = V_r + np.random.randn(*V_r.shape) * 0.01
                         W_p = W_r + np.random.randn(*W_r.shape) * 0.01
                         params_p = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
                         res_p = minimize(objective_and_grad, params_p, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'gtol': 1e-8})
                         if res_p.fun < 1e-9:
                             U_rp = res_p.x[:n*m*target_rank].reshape(n*m, target_rank)
                             V_rp = res_p.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                             W_rp = res_p.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                             rounded_p = try_integer_rounding(U_rp, V_rp, W_rp)
                             if rounded_p is not None:
                                 return {"U": rounded_p[0].tolist(), "V": rounded_p[1].tolist(), "W": rounded_p[2].tolist()}

            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END
