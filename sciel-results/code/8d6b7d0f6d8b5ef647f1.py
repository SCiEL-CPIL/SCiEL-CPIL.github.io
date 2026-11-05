"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def check_valid(U, V, W, tol=1e-10):
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T_target = np.ones((n, m, p))
        return np.allclose(T_rec, T_target, atol=tol)
    
    # Try to find a lower rank decomposition using optimization
    def try_optimize(rank, time_budget=30):
        """Try to find a valid decomposition of given rank using optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Objective: minimize ||T_rec - T_target||^2
        def objective(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            return np.sum(err ** 2)
        
        # Gradient
        def gradient(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            
            # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
            dU = np.einsum('jkc,kic->ijc', V_3d, W_3d)
            dV = np.einsum('ijc,kic->jkc', U_3d, W_3d)
            dW = np.einsum('ijc,jkc->kic', U_3d, V_3d)
            
            # 2 * err * dT/dparam
            dU = 2 * np.einsum('ijk,jkc->ijc', err, W_3d)
            dV = 2 * np.einsum('ijk,kic->jkc', err, U_3d)
            dW = 2 * np.einsum('ijk,jkc->kic', err, V_3d)
            
            return [dU.flatten(), dV.flatten(), dW.flatten()]
        
        # Flatten parameters
        def pack(U, V, W):
            return [U.flatten(), V.flatten(), W.flatten()]
        
        def unpack(params):
            U_flat, V_flat, W_flat = params
            return U_flat.reshape(n, m, rank), V_flat.reshape(m, p, rank), W_flat.reshape(p, n, rank)
        
        # Use L-BFGS-B optimization
        try:
            result = opt.minimize(
                objective,
                pack(U, V, W),
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < 1e-8:
                U_opt, V_opt, W_opt = unpack(result.x)
                if check_valid(U_opt, V_opt, W_opt, tol=1e-8):
                    return U_opt.reshape(n*m, rank), V_opt.reshape(m*p, rank), W_opt.reshape(p*n, rank), rank
        except Exception:
            pass
        
        return None
    
    # Try various ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    min_possible = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = n * m * p
    
    # Try to find decompositions at various ranks
    for rank in range(min_possible, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Skip ranks that are unlikely to work (heuristic)
        if rank < 30 and rank % 3 != 0:
            continue
            
        result = try_optimize(rank, time_budget=20)
        if result is not None:
            U, V, W, r = result
            best_result = (U, V, W, r)
            best_rank = r
            break  # Found a valid decomposition, stop
    
    # If no better found, use schoolbook
    if best_result is None:
        U, V, W, rank = schoolbook()
    else:
        U, V, W, rank = best_result
    
    # Final verification
    if not check_valid(U, V, W, tol=1e-8):
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
