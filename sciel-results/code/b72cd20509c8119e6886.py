"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s margin for 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    best_error = 0.0
    
    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        return error

    def solve_W(U, V):
        """For fixed U and V, compute optimal W using least squares."""
        W = np.zeros((p * n, U.shape[1]))
        for i in range(n):
            for k in range(p):
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, U.shape[1]))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve A @ w = ones(m) using least squares
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
                except:
                    W[k * n + i, :] = np.zeros(U.shape[1])
        return W

    def try_rank(r, iterations=2000, seed=None):
        """Try to find a rank-r decomposition using alternating optimization."""
        if time.time() > deadline:
            return None, None, None, float('inf')
            
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize U, V randomly with small values
        scale = 1.0 / np.sqrt(r)
        U = scale * np.random.randn(n * m, r)
        V = scale * np.random.randn(m * p, r)
        
        best_local_error = float('inf')
        best_local_U, best_local_V, best_local_W = U.copy(), V.copy(), None
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            # Compute W optimally for fixed U, V
            W = solve_W(U, V)
            
            # Compute error
            error = compute_error(U, V, W)
            
            if error < best_local_error:
                best_local_error = error
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
                
            if best_local_error < 1e-12:
                break
                
            # Gradient-free update: perturb U and V
            # Use adaptive step size based on current error
            step = 0.05 * max(1.0, np.sqrt(best_local_error))
            U += step * np.random.randn(n * m, r)
            V += step * np.random.randn(m * p, r)
            
        return best_local_U, best_local_V, best_local_W, best_local_error

    # Try ranks from rank0-1 down to theoretical lower bound
    # Lower bound: n*m + m*p + n*p - m = 6 + 30 + 20 - 3 = 53
    lower_bound = n * m + m * p + n * p - m
    lower_bound = max(lower_bound, 50)
    
    # Try a few promising ranks with multiple random restarts
    ranks_to_try = list(range(rank0 - 1, max(lower_bound - 1, rank0 - 15), -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for each rank
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            seed = random.randint(0, 1000000)
            U_r, V_r, W_r, err_r = try_rank(r, iterations=1500, seed=seed)
            
            if W_r is not None and err_r < best_error:
                best_error = err_r
                best_U = U_r
                best_V = V_r
                best_W = W_r
                best_rank = r
                
                if best_error < 1e-12:
                    break
        
        if best_error < 1e-12:
            break

    # Final polish: if we have a good solution, try to refine it
    if best_error < 1e-6 and best_rank < rank0:
        # Try to reduce error further with more iterations
        for _ in range(500):
            if time.time() > deadline:
                break
            
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
            
            # Small perturbations
            step = 0.01
            U += step * np.random.randn(n * m, best_rank)
            V += step * np.random.randn(m * p, best_rank)
            W_new = solve_W(U, V)
            
            err = compute_error(U, V, W_new)
            if err < best_error:
                best_error = err
                best_U = U
                best_V = V
                best_W = W_new

    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
