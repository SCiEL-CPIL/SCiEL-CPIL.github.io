"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def is_exact(U, V, W, n, m, p, tol=1e-6):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(arr, tol=1e-4):
        """Round array values to nearest half-integer (0, 0.5, 1, 1.5, ...) if close enough."""
        arr = np.asarray(arr, dtype=float)
        rounded = np.round(arr * 2) / 2
        if np.allclose(arr, rounded, atol=tol):
            return rounded
        return None
    
    def try_gradient_descent(rank, max_iters=2000, lr=0.01, seed=None, time_limit=30):
        """Use gradient descent to find a rank-r decomposition."""
        if seed is not None:
            np.random.seed(seed)
        
        start_time = time.time()
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Precompute indices for efficiency
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k) in indices:
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                
                # Current value
                s = np.sum(U[ui, :] * V[vj, :] * W[wk, :])
                error = s - 1.0
                
                # Gradients
                dU[ui, :] += error * V[vj, :] * W[wk, :]
                dV[vj, :] += error * U[ui, :] * W[wk, :]
                dW[wk, :] += error * U[ui, :] * V[vj, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if iteration % 100 == 0:
                if is_exact(U, V, W, n, m, p, tol=1e-4):
                    # Try to round to half-integers
                    U_rounded = round_to_half_int(U, tol=1e-3)
                    V_rounded = round_to_half_int(V, tol=1e-3)
                    W_rounded = round_to_half_int(W, tol=1e-3)
                    
                    if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                        if is_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
            
            # Adaptive learning rate
            if iteration % 500 == 0 and iteration > 0:
                lr *= 0.9
        
        # Final check
        if is_exact(U, V, W, n, m, p, tol=1e-4):
            U_rounded = round_to_half_int(U, tol=1e-3)
            V_rounded = round_to_half_int(V, tol=1e-3)
            W_rounded = round_to_half_int(W, tol=1e-3)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if is_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        
        return None
    
    def try_structured_construction(rank):
        """Try a structured approach based on Kronecker products."""
        # For 2x5x6, we can try to exploit the structure
        # One idea: decompose into blocks
        
        # Try to find if there's a pattern with rank < 60
        # This is heuristic and may not work
        
        return None
    
    # Try gradient descent for various ranks
    # Start from the theoretical lower bound (30) up to schoolbook (60)
    # But be smart about which ranks to try
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks that are multiples of small numbers or have special structure
    ranks_to_try = [30, 32, 34, 36, 38, 40, 42, 44, 45, 48, 50, 54, 56, 60]
    
    for rank in sorted(ranks_to_try):
        if rank >= best_rank:
            continue
        
        result = try_gradient_descent(rank, max_iters=1000, lr=0.02, seed=42, time_limit=20)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
