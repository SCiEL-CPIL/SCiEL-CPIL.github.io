"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s limit

    # Build the target tensor
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using einsum."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Using einsum: 'ar,br,cr->abc'
        try:
            computed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        except Exception:
            return False
        return np.allclose(computed, target, atol=1e-9)

    def verify_snapped(U, V, W):
        """Verify after snapping to half-integers."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Snap to nearest half-integer
        U_s = np.round(U_arr * 2) / 2
        V_s = np.round(V_arr * 2) / 2
        W_s = np.round(W_arr * 2) / 2
        
        # Check if snap is close to original
        if not (np.allclose(U_s, U_arr, atol=1e-7) and 
                np.allclose(V_s, V_arr, atol=1e-7) and 
                np.allclose(W_s, W_arr, atol=1e-7)):
            return False
        
        # Verify the snapped version
        try:
            computed = np.einsum('ar,br,cr->abc', U_s, V_s, W_s, optimize=True)
        except Exception:
            return False
        return np.allclose(computed, target, atol=1e-9)

    def optimize_decomposition(rank, init_seed=0):
        """Use gradient-based optimization to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
            
        random.seed(init_seed)
        np.random.seed(init_seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Flatten parameters
        def flatten(params):
            U = params[:n * m * rank].reshape(n * m, rank)
            V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            return U, V, W
        
        # Objective: minimize ||T_computed - T_target||^2
        def objective(params):
            if time.time() > deadline - 5:
                return float('inf')
            U, V, W = flatten(params)
            try:
                computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            except Exception:
                return float('inf')
            diff = computed - target
            return float(np.sum(diff * diff))
        
        # Try multiple random restarts
        best_obj = float('inf')
        best_params = None
        
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            scale = 0.3 + 0.1 * restart
            U_init = np.random.randn(n * m, rank) * scale
            V_init = np.random.randn(m * p, rank) * scale
            W_init = np.random.randn(p * n, rank) * scale
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            # Optimize
            try:
                result = minimize(
                    objective,
                    params_init,
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-6}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
                    
                    # Check if we've converged
                    if best_obj < 1e-8:
                        break
                        
            except Exception:
                continue
        
        if best_params is None:
            return None
            
        U, V, W = flatten(best_params)
        
        # Try to snap to half-integers
        if verify_snapped(U.tolist(), V.tolist(), W.tolist()):
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
        # Check without snapping
        if verify(U.tolist(), V.tolist(), W.tolist(), rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        return None

    def try_random_search(rank, max_iters=500, seed=0):
        """Random search with structured values."""
        if seed is not None:
            random.seed(seed)
            
        vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
                
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try optimization for various ranks
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # Schoolbook: 50
    
    for rank in range(25, 51):
        if time.time() > deadline - 10:
            break
        
        # Try gradient-based optimization
        result = optimize_decomposition(rank, init_seed=rank)
        if result is not None:
            return result
        
        # Try random search
        result = try_random_search(rank, max_iters=500, seed=rank)
        if result is not None:
            return result
    
    # If no improvement found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
