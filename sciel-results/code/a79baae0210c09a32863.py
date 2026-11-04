"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from 59 down
    for rank in range(best_rank - 1, max(best_rank - 30, 10), -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank <= 40:
                break

    return best_result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: sum over r
    # U[i,j,r] * V[j,k,r] -> (n, m, p, rank) then multiply by W[k,i,r]
    # Better: use einsum directly
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T


def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of 0.5*sum(T-1)^2 w.r.t. U, V, W"""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    diff = T - 1.0  # (n, m, p)
    
    # grad_U[i,j,r] = sum_k diff[i,j,k] * V[j,k,r] * W[k,i,r]
    # grad_V[j,k,r] = sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
    # grad_W[k,i,r] = sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
    
    grad_U = np.einsum('ijk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
    grad_V = np.einsum('ijk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
    grad_W = np.einsum('ijk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
    
    return grad_U.reshape(dim_u, rank), grad_V.reshape(dim_v, rank), grad_W.reshape(dim_w, rank)


def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using vectorized L-BFGS-B."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_vars = dim_u * rank + dim_v * rank + dim_w * rank

    rng = random.Random(42 + rank * 13)

    max_restarts = 6
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break

        # Random initialization
        scale = rng.uniform(0.5, 2.0)
        U = np.random.randn(dim_u, rank) * scale
        V = np.random.randn(dim_v, rank) * scale
        W = np.random.randn(dim_w, rank) * scale

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def objective(x):
            U_flat = x[:dim_u * rank].reshape(dim_u, rank)
            V_flat = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_flat = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            T = _compute_tensor(U_flat, V_flat, W_flat, n, m, p)
            return 0.5 * np.sum((T - 1.0) ** 2)

        def gradient(x):
            U_flat = x[:dim_u * rank].reshape(dim_u, rank)
            V_flat = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_flat = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            gU, gV, gW = _compute_gradients(U_flat, V_flat, W_flat, n, m, p)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxls': 50, 'maxfun': 500})
        except Exception:
            continue

        if time.time() > deadline - 10:
            break

        if result.fun < 1e-10:
            U_opt = result.x[:dim_u * rank].reshape(dim_u, rank)
            V_opt = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_opt = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)

            # Newton polish
            U_opt, V_opt, W_opt = _newton_polish(U_opt, V_opt, W_opt, n, m, p)

            # Try to rationalize
            rat_result = _rationalize_optimized(U_opt, V_opt, W_opt, n, m, p)
            if rat_result is not None:
                return rat_result

    return None


def _newton_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Apply Newton-Raphson iterations to polish the decomposition."""
    rank = U.shape[1]
    
    for _ in range(max_iter):
        T = _compute_tensor(U, V, W, n, m, p)
        residual = T - 1.0
        err = np.max(np.abs(residual))
        if err < tol:
            break
        
        # Compute Jacobian-like structure for Newton step
        # This is complex; use a simple damped Newton approach
        # For now, use gradient descent with adaptive step
        gU, gV, gW = _compute_gradients(U, V, W, n, m, p)
        
        # Adaptive step size
        step = 0.1
        for _ in range(10):
            U_new = U - step * gU
            V_new = V - step * gV
            W_new = W - step * gW
            T_new = _compute_tensor(U_new, V_new, W_new, n, m, p)
            err_new = np.max(np.abs(T_new - 1.0))
            if err_new < err:
                U, V, W = U_new, V_new, W_new
                break
            step *= 0.5
    
    return U, V, W


def _rationalize_optimized(U, V, W, n, m, p, max_denom=16):
    """Try to find rational/half-integer approximation to the optimized decomposition."""
    rank = U.shape[1]

    for denom in [1, 2, 4, 8, 16]:
        if denom > max_denom:
            break
        scale = 2.0 / denom

        U_int = np.round(U * scale).astype(int)
        V_int = np.round(V * scale).astype(int)
        W_int = np.round(W * scale).astype(int)

        # Verify using vectorized computation
        U_check = U_int.astype(float) / scale
        V_check = V_int.astype(float) / scale
        W_check = W_int.astype(float) / scale
        T = _compute_tensor(U_check, V_check, W_check, n, m, p)
        max_err = np.max(np.abs(T - 1.0))

        if max_err < 1e-9:
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]

            for i in range(n * m):
                for r in range(rank):
                    U_list[i][r] = U_int[i, r] / scale
            for i in range(m * p):
                for r in range(rank):
                    V_list[i][r] = V_int[i, r] / scale
            for i in range(p * n):
                for r in range(rank):
                    W_list[i][r] = W_int[i, r] / scale

            return {"U": U_list, "V": V_list, "W": W_list}

    return None
# EVOLVE-BLOCK-END
