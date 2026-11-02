"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Build target tensor: T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j', i=i', k=k'
    # For C[i,k] = sum_j A[i,j] * B[j,k]:
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        r = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_recon += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
        return np.allclose(T_recon, T_target, atol=tol, rtol=0), T_recon
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, current_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), current_rank
    
    # Strategy: Use structured rank reduction
    # For 3x3x6, we can think of it as two 3x3x3 multiplications with sharing
    # Or use the fact that rank(3x3x3) <= 23 (best known), so 3x3x6 might be ~40-45
    
    # Approach 1: Try to find a rank-r decomposition using iterative methods
    # with better initialization based on structure
    
    def try_structured_rank_reduction(target_rank, num_trials=3, max_iter=300):
        """Try to find a rank-target_rank decomposition using structured init."""
        if target_rank >= current_rank:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with partial schoolbook + random perturbation
            np.random.seed(trial * 7919 + 12345)
            
            # Use a smarter initialization: start from schoolbook and merge
            if target_rank >= n*m*p//2:
                # Start from schoolbook and randomly merge pairs
                U_t = np.zeros((n*m, target_rank))
                V_t = np.zeros((m*p, target_rank))
                W_t = np.zeros((p*n, target_rank))
                
                # Assign first target_rank terms from schoolbook
                for t in range(target_rank):
                    idx = t
                    i, j, k = idx // (m*p), (idx // p) % m, idx % p
                    U_t[i*m+j, t] = 1.0
                    V_t[j*p+k, t] = 1.0
                    W_t[k*n+i, t] = 1.0
                
                # For remaining, use random
                for t in range(target_rank):
                    if U_t[:, t].sum() == 0:
                        U_t[:, t] = np.random.randn(n*m) * 0.1
                        V_t[:, t] = np.random.randn(m*p) * 0.1
                        W_t[:, t] = np.random.randn(p*n) * 0.1
            else:
                # Pure random with scaling
                U_t = np.random.randn(n*m, target_rank) * 0.5
                V_t = np.random.randn(m*p, target_rank) * 0.5
                W_t = np.random.randn(p*n, target_rank) * 0.5
            
            # Iterative refinement with adaptive step size
            lr = 0.05
            best_loss = float('inf')
            best_state = (U_t.copy(), V_t.copy(), W_t.copy())
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute reconstruction
                T_recon = np.zeros((n*m, m*p, p*n))
                for t in range(target_rank):
                    T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
                
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U_t.copy(), V_t.copy(), W_t.copy())
                
                if loss < 1e-14:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U_t)
                dV = np.zeros_like(V_t)
                dW = np.zeros_like(W_t)
                for t in range(target_rank):
                    dU[:,t] = np.einsum('abc,bc->a', error, V_t[:,t] * W_t[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, U_t[:,t] * W_t[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, U_t[:,t] * V_t[:,t])
                
                # Adaptive learning rate with backtracking
                for _ in range(10):
                    U_new = U_t - lr * dU
                    V_new = V_t - lr * dV
                    W_new = W_t - lr * dW
                    
                    T_recon_new = np.zeros((n*m, m*p, p*n))
                    for t in range(target_rank):
                        T_recon_new += np.outer(np.outer(U_new[:,t], V_new[:,t]), W_new[:,t])
                    loss_new = np.sum((T_target - T_recon_new)**2)
                    
                    if loss_new < loss:
                        U_t, V_t, W_t = U_new, V_new, W_new
                        loss = loss_new
                        break
                    lr *= 0.5
                
                # Decay learning rate
                if it % 20 == 0 and it > 0:
                    lr *= 0.9
                
                # Check if we've converged
                if loss < 1e-12:
                    break
            
            # Final check with best state found
            U_t, V_t, W_t = best_state
            T_recon = np.zeros((n*m, m*p, p*n))
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
            
            if np.allclose(T_recon, T_target, atol=1e-10, rtol=0):
                return U_t, V_t, W_t, target_rank
        
        return None
    
    # Try progressively lower ranks
    # For 3x3x6, best known is likely around 40-45
    # Start from 45 and go down
    for target_rank in range(45, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        result = try_structured_rank_reduction(target_rank, num_trials=4, max_iter=200)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    U_final, V_final, W_final, r_final = best_U, best_V, best_W, best_rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        """Convert numpy array to list of lists, rounding to nearest half-integer if close."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    new_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    new_row.append(nearest_half)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END

