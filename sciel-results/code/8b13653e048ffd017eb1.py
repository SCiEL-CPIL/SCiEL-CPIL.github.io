import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    deadline = 110  # safety margin before 120s limit
    
    # Build the target tensor T[i, j, k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # T[i,j,k] = delta_{j,j} for the multiplication tensor
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to (i,k) entry of C. So T[i,j,k] = 1 for all valid indices.
    # The tensor is: T[i,j,k] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # But that's not right. The multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k in their ranges.
    
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Target: T[i,j,k] = 1 for all valid i,j,k
    # Flatten: T_flat[idx] where idx = (i*m + j)*p + k for U, etc.
    
    # Let's verify the current rank-42 solution first
    rank_42 = 42
    U42 = np.zeros((n*m, rank_42))
    V42 = np.zeros((m*p, rank_42))
    W42 = np.zeros((p*n, rank_42))
    
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    col = 0
    for j_block in range(2):
        for k_block in range(3):
            for prod_idx in range(7):
                r = col + prod_idx
                for local_i, sign in strassen_U[prod_idx].items():
                    gi = local_i // 2
                    gj = local_i % 2
                    U42[gi * m + (j_block * 2 + gj), r] = sign
                for local_jk, sign in strassen_V[prod_idx].items():
                    gj = local_jk // 2
                    gk = local_jk % 2
                    V42[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                for local_ik, sign in strassen_W[prod_idx].items():
                    gi = local_ik // 2
                    gk = local_ik % 2
                    W42[(k_block * 2 + gk) * n + gi, r] = sign
                col += 1
            col = col  # already incremented per prod_idx
    
    # Verify rank-42 solution
    def verify_decomp(U, V, W, n, m, p):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build tensor from decomposition
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T_reconstructed[i, j, k] = val
        
        # Target tensor: T[i,j,k] = 1 for all i,j,k
        T_target = np.ones((n, m, p))
        max_err = np.max(np.abs(T_reconstructed - T_target))
        return max_err < 1e-10
    
    # Verify our rank-42 solution
    err_42 = verify_decomp(U42, V42, W42, n, m, p)
    
    # Try to find lower rank solutions using border rank approach
    # For rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Use iterative optimization: start with random init, minimize residual
    def try_rank(R, max_iters=500, lr=0.01, seed=42):
        """Try to find a rank-R decomposition using gradient descent on the residual."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute residual: T_reconstructed[i,j,k] - 1
            # T_recon[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # Compute gradients using the chain rule
            # d/dU[i*m+j,r] of sum_{i,j,k} (T_recon[i,j,k] - 1)^2
            # = 2 * sum_{i,j,k} (T_recon[i,j,k] - 1) * V[j*p+k,r] * W[k*n+i,r] * [i,j,k matches]
            # For a specific U[a,b,r] where a = i*m+j:
            # grad_U[a, r] = 2 * sum_k (T_recon[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            
            # Compute T_reconstructed
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            residual = T_recon - 1.0
            err = np.max(np.abs(residual))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residual[i, j, k]
                        if abs(res) < 1e-12:
                            continue
                        # d/dU[i*m+j, r]
                        grad_U[i*m+j, :] += 2 * res * V[j*p+k, :] * W[k*n+i, :]
                        # d/dV[j*p+k, r]
                        grad_V[j*p+k, :] += 2 * res * U[i*m+j, :] * W[k*n+i, :]
                        # d/dW[k*n+i, r]
                        grad_W[k*n+i, :] += 2 * res * U[i*m+j, :] * V[j*p+k, :]
            
            # Gradient descent with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.5
        
        return best_err, best_U, best_V, best_W
    
    # Try ranks from 41 down to 30
    best_rank = 42
    best_solution = (U42, V42, W42)
    
    for R in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            err, U_R, V_R, W_R = try_rank(R, max_iters=300, lr=0.02, seed=seed)
            
            if err < 1e-8:
                # Found a valid solution! Try to rationalize
                # Round to nearest half-integers
                U_rat = np.round(U_R * 2) / 2
                V_rat = np.round(V_R * 2) / 2
                W_rat = np.round(W_R * 2) / 2
                
                # Verify the rationalized solution
                err_rat = verify_decomp(U_rat, V_rat, W_rat, n, m, p)
                if err_rat < 1e-10:
                    best_rank = R
                    best_solution = (U_rat, V_rat, W_rat)
                    found = True
                    break
        
        if found:
            continue  # Try even lower ranks
        else:
            break  # If we can't find R, likely can't find R-1 either
    
    # If no lower rank found, return the rank-42 solution
    if best_rank == 42:
        U, V, W = U42, V42, W42
    else:
        U, V, W = best_solution
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
