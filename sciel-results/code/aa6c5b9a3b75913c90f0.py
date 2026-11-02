import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    def validate(U, V, W, tol=1e-9):
        """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]
                    v_val = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard rank-45 schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        return U, V, W
    
    def rationalize_matrix(mat, max_denom=1000):
        """Try to convert floating point matrix to rational (integer or half-integer) entries"""
        rows, cols = mat.shape
        result = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                val = mat[i, j]
                if abs(val) < 1e-10:
                    result[i, j] = 0
                else:
                    # Try to find a fraction with small denominator
                    found = False
                    for denom in range(1, max_denom + 1):
                        num = round(val * denom)
                        if abs(val - num / denom) < 1e-8:
                            result[i, j] = num / denom
                            found = True
                            break
                    if not found:
                        # Fallback: use the value as-is
                        result[i, j] = val
        return result
    
    def optimize_rank(R, max_iter=2000, lr=0.01, seed=42):
        """Optimize factors for given rank R"""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(N, R) * 0.5
        V = rng.randn(M, R) * 0.5
        W = rng.randn(P, R) * 0.5
        
        # Precompute index arrays for efficiency
        idx_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_list.append((i*m + j, j*p + k, k*n + i))
        
        idx_u = np.array([x[0] for x in idx_list])
        idx_v = np.array([x[1] for x in idx_list])
        idx_w = np.array([x[2] for x in idx_list])
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute current values
            s_vals = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :], axis=1)
            errors = 1.0 - s_vals
            loss = np.sum(errors**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for t in range(len(idx_list)):
                ui, vi, wi = idx_u[t], idx_v[t], idx_w[t]
                err = errors[t]
                
                # dLoss/dU[ui, r] = -2 * err * V[vi, r] * W[wi, r]
                grad_U[ui, :] += -2.0 * err * V[vi, :] * W[wi, :]
                # dLoss/dV[vi, r] = -2 * err * U[ui, r] * W[wi, r]
                grad_V[vi, :] += -2.0 * err * U[ui, :] * W[wi, :]
                # dLoss/dW[wi, r] = -2 * err * U[ui, r] * V[vi, r]
                grad_W[wi, :] += -2.0 * err * U[ui, :] * V[vi, :]
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr = max(0.001, min(0.1, lr * 0.95))
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        return best_U, best_V, best_W, best_loss
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try ranks from 15 up to 44
    
    ranks_to_try = list(range(15, 45))
    
    for R in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            seed = 42 + R * 100 + attempt
            U_opt, V_opt, W_opt, loss = optimize_rank(
                R, max_iter=3000, lr=0.02, seed=seed
            )
            
            # Check if solution is good
            if loss < 1e-8:
                # Try to rationalize
                U_rat = rationalize_matrix(U_opt)
                V_rat = rationalize_matrix(V_opt)
                W_rat = rationalize_matrix(W_opt)
                
                # Validate the rationalized version
                if validate(U_rat, V_rat, W_rat, tol=1e-6):
                    best_U, best_V, best_W = U_rat, V_rat, W_rat
                    best_rank = R
                    found = True
                    break
                elif validate(U_opt, V_opt, W_opt, tol=1e-6):
                    # Use the floating point version if rationalization fails
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # Final validation
    if not validate(best_U, best_V, best_W, tol=1e-6):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
