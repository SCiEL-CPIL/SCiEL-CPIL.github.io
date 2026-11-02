"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

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
                    s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_integers(arr):
        """Round array entries to nearest half-integer (multiple of 0.5)."""
        return np.round(arr * 2) / 2
    
    def check_half_integer(arr):
        """Check if all entries are half-integers."""
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=1e-6)
    
    def gradient_descent_decomposition(rank, n, m, p, max_iters=5000, lr=0.01, seed=42):
        """
        Find a rank-r decomposition using gradient descent on the QCQP formulation.
        Minimize ||T - T_decomp||^2 where T_decomp is the tensor from factors U,V,W.
        """
        np.random.seed(seed)
        
        # Initialize factors randomly
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Compute current tensor and loss
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target is 1 for all (i,j,k) in matrix multiplication tensor
                        current = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(rank))
                        error = current - 1.0
                        loss += error * error
                        
                        # Gradients
                        for t in range(rank):
                            u_val = U[i*m+j, t]
                            v_val = V[j*p+k, t]
                            w_val = W[k*n+i, t]
                            
                            # d/dU[i*m+j,t] = 2*error * V[j*p+k,t] * W[k*n+i,t]
                            grad_U[i*m+j, t] += 2 * error * v_val * w_val
                            # d/dV[j*p+k,t] = 2*error * U[i*m+j,t] * W[k*n+i,t]
                            grad_V[j*p+k, t] += 2 * error * u_val * w_val
                            # d/dW[k*n+i,t] = 2*error * U[i*m+j,t] * V[j*p+k,t]
                            grad_W[k*n+i, t] += 2 * error * u_val * v_val
            
            # Update with learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-8:
                break
        
        return best_U, best_V, best_W, best_loss
    
    def try_rank_with_rounding(rank, n, m, p, num_trials=5, max_iters=3000):
        """Try to find a valid rank-r decomposition with half-integer entries."""
        for trial in range(num_trials):
            seed = 42 + trial * 1000
            U, V, W, loss = gradient_descent_decomposition(rank, n, m, p, max_iters=max_iters, seed=seed)
            
            # Round to half-integers
            U_rounded = round_to_half_integers(U)
            V_rounded = round_to_half_integers(V)
            W_rounded = round_to_half_integers(W)
            
            # Check if rounded solution is exact
            if is_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-4):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Also try without rounding if the continuous solution is already close to half-integers
            if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                if is_exact(U, V, W, n, m, p, tol=1e-6):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    lower_bound = max(n*m, m*p, p*n)
    schoolbook_rank = n * m * p
    
    # Try a range of ranks, starting from lower bound
    # We'll try a few key ranks
    ranks_to_try = []
    for r in range(lower_bound, min(lower_bound + 15, schoolbook_rank + 1)):
        ranks_to_try.append(r)
    # Also try some intermediate ranks
    for r in [35, 40, 45, 50, 55]:
        if lower_bound <= r < schoolbook_rank and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        result = try_rank_with_rounding(rank, n, m, p, num_trials=3, max_iters=2000)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
