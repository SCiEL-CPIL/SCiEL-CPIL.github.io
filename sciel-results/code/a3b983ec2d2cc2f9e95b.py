import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=1e-9)

    def error_and_grad(params, rank):
        # params is a flat array of size (n*m + m*p + p*n) * rank
        # We need to split it into U, V, W
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Compute Rec
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
            
        err = Rec - 1.0
        loss = np.sum(err**2)
        
        # Compute Gradient
        # dLoss/dU[i*m+j, r] = 2 * sum_{k} err[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        # dLoss/dV[j*p+k, r] = 2 * sum_{i} err[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
        # dLoss/dW[k*n+i, r] = 2 * sum_{j} err[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            # grad_U[i, j] = 2 * sum_k err[i, j, k] * v[j, k] * w[k, i]
            # This is 2 * einsum('ijk,jk,ki->ij', err, v, w)
            grad_U[:, r] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w, optimize=True).flatten()
            
            # grad_V[j, k] = 2 * sum_i err[i, j, k] * u[i, j] * w[k, i]
            # This is 2 * einsum('ijk,ij,ki->jk', err, u, w, optimize=True)
            grad_V[:, r] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w, optimize=True).flatten()
            
            # grad_W[k, i] = 2 * sum_j err[i, j, k] * u[i, j] * v[j, k]
            # This is 2 * einsum('ijk,ij,jk->ki', err, u, v, optimize=True)
            grad_W[:, r] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v, optimize=True).flatten()
            
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad_flat

    def try_rank(rank, num_restarts=10):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return False
                
            # Initialize randomly
            U_init = np.random.randn(n * m, rank)
            V_init = np.random.randn(m * p, rank)
            W_init = np.random.randn(p * n, rank)
            
            # Scale initialization to avoid overflow/underflow during optimization
            # The target is 1. If we initialize with large values, gradient is huge.
            # If we initialize with small values, gradient is small.
            # Let's try scaling by 1/sqrt(rank) or similar.
            scale = 1.0 / np.sqrt(rank)
            U_init *= scale
            V_init *= scale
            W_init *= scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(error_and_grad, params_init, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception as e:
                continue
                
            if res.success or res.fun < 1e-6:
                # Extract optimized U, V, W
                size_U = n * m * rank
                size_V = m * p * rank
                U_opt = res.x[:size_U].reshape(n * m, rank)
                V_opt = res.x[size_U:size_U + size_V].reshape(m * p, rank)
                W_opt = res.x[size_U + size_V:].reshape(p * n, rank)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                if is_valid(U_snap, V_snap, W_snap, rank):
                    best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                    best_rank = rank
                    return True
                    
                # Also try rounding to integer
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if is_valid(U_int, V_int, W_int, rank):
                    best_solution = {"U": U_int, "V": V_int, "W": W_int}
                    best_rank = rank
                    return True

        return False

    # Search for lower ranks
    # Known bounds suggest rank around 23-26 for 3x3x5
    # We start from 24 to be safe, going up to best_rank - 1
    for rank in range(24, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank, num_restarts=15):
            # Found a solution, but we want minimal.
            # Since we iterate upwards, the first one found is the minimal in this range.
            # However, if 24 fails, 25 might succeed.
            # If 25 succeeds, we return 25.
            # If we find 25, we should stop searching for 26..44?
            # Yes, because we want the minimum.
            # But what if 24 was skipped? We started at 24.
            # So if we find a solution at 25, it's the best found so far.
            # We can break and return.
            # But wait, did we check 24? Yes.
            # So if try_rank(24) fails, we try 25. If 25 succeeds, we return 25.
            # This is correct.
            break

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END
