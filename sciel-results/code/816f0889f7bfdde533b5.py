"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Precompute target tensor
    # Shape: (n*m, m*p, p*n) = (10, 30, 12)
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        # Convert to list of lists for output interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    best_rank = rank_schoolbook

    def verify_decomposition_np(U, V, W):
        """Verify using NumPy. U, V, W are numpy arrays."""
        # Compute T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # einsum is efficient for this
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, target, atol=1e-10)

    def flatten_params(U, V, W):
        """Flatten the three matrices into a single vector."""
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(x):
        """Unflatten a single vector back into three matrices."""
        u_size = n * m
        v_size = m * p
        w_size = p * n
        U = x[:u_size].reshape(n*m, -1)
        V = x[u_size:u_size+v_size].reshape(m*p, -1)
        W = x[u_size+v_size:].reshape(p*n, -1)
        return U, V, W

    def compute_loss_and_grad(x):
        """Compute loss and gradient for L-BFGS."""
        U, V, W = unflatten_params(x)
        # Loss: sum of squared differences
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_decomp - target
        loss = np.sum(diff * diff)
        
        # Gradient computation using chain rule
        # dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # dL/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
        # dL/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank_lbfgs(r, max_iter=500, seed=42, init_scale=1.0):
        """Try to find a rank-r decomposition using L-BFGS."""
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = rng.gauss(0, init_scale, (n*m, r))
        V = rng.gauss(0, init_scale, (m*p, r))
        W = rng.gauss(0, init_scale, (p*n, r))
        
        x0 = flatten_params(U, V, W)
        
        # Use L-BFGS with gradient
        result = minimize(
            compute_loss_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        
        if result.success or result.fun < 1e-10:
            U_opt, V_opt, W_opt = unflatten_params(result.x)
            if verify_decomposition_np(U_opt, V_opt, W_opt):
                return U_opt, V_opt, W_opt
        
        return None

    def try_rank_with_restart(r, n_restarts=5, max_iter=300, seed_base=42):
        """Try multiple restarts for rank r."""
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            seed = seed_base + restart * 1000
            init_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
            init_scale = init_scales[restart % len(init_scales)]
            
            result = try_rank_lbfgs(r, max_iter=max_iter, seed=seed, init_scale=init_scale)
            if result is not None:
                return result
        
        return None

    # Try ranks from small to larger
    # Start with ranks that might be achievable
    ranks_to_try = list(range(10, 25)) + list(range(30, 45)) + list(range(50, 60))
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = try_rank_with_restart(r, n_restarts=3, max_iter=200, seed_base=r*100)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            print(f"Found rank {r} decomposition!")
            return best_result  # Return immediately if found
    
    return best_result
