import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 100)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Construct target tensor for verification
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_mat, V_mat, W_mat, T_target, tol=1e-6):
        """Check if U, V, W exactly decompose T_target"""
        R = U_mat.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = np.max(np.abs(T_approx - T_target))
        return diff < tol, diff

    # Verify initial schoolbook
    is_valid, err = verify(U, V, W, T)
    if not is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank_sb
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds limit
    
    # Use algebraic geometry approach: compute border rank via catalecticant matrices
    # The border rank gives a lower bound on tensor rank
    # For matrix multiplication tensor M_{n,m,p}, we can analyze the flattening
    
    # Compute the matrix flattening (matricization) of T
    # Flatten along the first mode: shape (n*m, m*p * p*n)
    T_flat1 = T.reshape(n * m, m * p * p * n)
    rank_flat1 = np.linalg.matrix_rank(T_flat1)
    
    # Flatten along second mode: shape (m*p, n*m * p*n)
    T_flat2 = T.reshape(m * p, n * m * p * n)
    rank_flat2 = np.linalg.matrix_rank(T_flat2)
    
    # Flatten along third mode: shape (p*n, n*m * m*p)
    T_flat3 = T.reshape(p * n, n * m * m * p)
    rank_flat3 = np.linalg.matrix_rank(T_flat3)
    
    # The tensor rank is at least the maximum of these flattening ranks
    # Also, for a rank-r decomposition, each flattening has rank at most r
    # So tensor_rank >= max(rank_flat1, rank_flat2, rank_flat3)
    
    lower_bound = max(rank_flat1, rank_flat2, rank_flat3)
    
    # For 4x5x5 matrix multiplication, let's see what the flattening ranks are
    # If lower_bound < 100, we might be able to find a better decomposition
    
    # Try to find a decomposition with rank close to the lower bound
    # Use iterative refinement with random initialization
    
    def try_rank(target_rank, max_iters=50):
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return False
        
        # Multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                return False
            
            np.random.seed(1000 + target_rank * 10 + trial)
            U_curr = np.random.randn(n * m, target_rank) * 0.5
            V_curr = np.random.randn(m * p, target_rank) * 0.5
            W_curr = np.random.randn(p * n, target_rank) * 0.5
            
            # Use alternating least squares style updates
            lr = 0.1
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Compute residual
                T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                diff = T_approx - T
                
                # Check if we've converged
                if np.max(np.abs(diff)) < 1e-8:
                    # Round to half-integers and verify
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    is_valid, _ = verify(U_rounded, V_rounded, W_rounded, T)
                    if is_valid:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        return True
                
                # Gradient descent step
                grad_U = np.einsum('abc,br,cr->ar', diff, V_curr, W_curr)
                grad_V = np.einsum('abc,ar,cr->br', diff, U_curr, W_curr)
                grad_W = np.einsum('abc,ar,br->cr', diff, U_curr, V_curr)
                
                # Adaptive learning rate based on gradient magnitude
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    lr_eff = lr * (1.0 / (1.0 + grad_norm * 0.1))
                    U_curr -= lr_eff * grad_U
                    V_curr -= lr_eff * grad_V
                    W_curr -= lr_eff * grad_W
            
            # Final check
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            is_valid, _ = verify(U_rounded, V_rounded, W_rounded, T)
            if is_valid:
                best_rank = target_rank
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                return True
        
        return False
    
    # Try ranks from lower_bound up to 99
    # Start from a reasonable estimate
    search_start = min(99, lower_bound + 5)
    search_end = max(lower_bound, 85)
    
    for r in range(search_start, search_end - 1, -1):
        if time.time() > deadline:
            break
        if try_rank(r):
            break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
