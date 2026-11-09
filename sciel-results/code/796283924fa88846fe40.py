<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # T is a 3-tensor of shape (n, m, p)
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Initialize with schoolbook decomposition (rank = n*m*p = 50)
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank using randomized optimization
    # Start with various initial ranks and try to find exact decompositions
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try different target ranks
    target_ranks = [40, 45, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize factors with random values
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent to minimize reconstruction error
            lr = 0.01
            tol = 1e-10
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Compute reconstruction: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # This is expensive, so we'll use a smarter approach
                
                # Compute the reconstructed tensor
                recon = np.zeros((n, m, p))
                for r in range(target_rank):
                    # U[:, r] is vector of length n*m, reshape to (n, m)
                    # V[:, r] is vector of length m*p, reshape to (m, p)
                    # W[:, r] is vector of length p*n, reshape to (p, n)
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    
                    # For each (i,j,k): Ur[i,j] * Vr[j,k] * Wr[k,i]
                    # This is a rank-1 tensor
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
                
                # Target tensor
                target = np.ones((n, m, p))
                error = np.sum((recon - target) ** 2)
                
                if error < tol:
                    break
                
                # Compute gradients (simplified - use finite differences for correctness)
                # For efficiency, we'll use a simpler update rule
                
                # Update U
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_U = np.zeros((n, m))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad_U[i, j] += 2 * (recon[i, j, k] - 1) * Vr[j, k] * Wr[k, i]
                    U_r[:, r] -= lr * grad_U.flatten()
                
                # Update V
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_V = np.zeros((m, p))
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                grad_V[j, k] += 2 * (recon[i, j, k] - 1) * Ur[i, j] * Wr[k, i]
                    V_r[:, r] -= lr * grad_V.flatten()
                
                # Update W
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_W = np.zeros((p, n))
                    for k in range(p):
                        for i in range(n):
                            for j in range(m):
                                grad_W[k, i] += 2 * (recon[i, j, k] - 1) * Ur[i, j] * Vr[j, k]
                    W_r[:, r] -= lr * grad_W.flatten()
            
            # Check final error
            recon = np.zeros((n, m, p))
            for r in range(target_rank):
                Ur = U_r[:, r].reshape(n, m)
                Vr = V_r[:, r].reshape(m, p)
                Wr = W_r[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
            
            target = np.ones((n, m, p))
            final_error = np.sum((recon - target) ** 2)
            
            if final_error < tol and target_rank < best_rank:
                best_rank = target_rank
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
=======
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Try to find lower rank decompositions using vectorized optimization
    # The key insight: we can vectorize the reconstruction and gradient computation
    
    def compute_recon(U_r, V_r, W_r, n, m, p, target_rank):
        """Compute reconstruction tensor using vectorized operations"""
        recon = np.zeros((n, m, p))
        for r in range(target_rank):
            Ur = U_r[:, r].reshape(n, m)
            Vr = V_r[:, r].reshape(m, p)
            Wr = W_r[:, r].reshape(p, n)
            # Rank-1 tensor: Ur[i,j] * Vr[j,k] * Wr[k,i]
            # Use einsum for efficiency
            recon += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
        return recon
    
    def compute_gradients(U_r, V_r, W_r, recon, target, n, m, p, target_rank):
        """Compute gradients for U, V, W"""
        diff = recon - target
        grad_U = np.zeros_like(U_r)
        grad_V = np.zeros_like(V_r)
        grad_W = np.zeros_like(W_r)
        
        for r in range(target_rank):
            Ur = U_r[:, r].reshape(n, m)
            Vr = V_r[:, r].reshape(m, p)
            Wr = W_r[:, r].reshape(p, n)
            
            # grad_U[i,j] = sum_k 2 * diff[i,j,k] * Vr[j,k] * Wr[k,i]
            # grad_V[j,k] = sum_i 2 * diff[i,j,k] * Ur[i,j] * Wr[k,i]
            # grad_W[k,i] = sum_j 2 * diff[i,j,k] * Ur[i,j] * Vr[j,k]
            
            grad_U[:, r] = np.einsum('ijk,jk,ki->ij', diff, Vr, Wr).flatten() * 2
            grad_V[:, r] = np.einsum('ijk,ij,ki->jk', diff, Ur, Wr).flatten() * 2
            grad_W[:, r] = np.einsum('ijk,ij,jk->ki', diff, Ur, Vr).flatten() * 2
        
        return grad_U, grad_V, grad_W
    
    target_tensor = np.ones((n, m, p))
    
    # Try different target ranks with vectorized optimization
    target_ranks = [40, 45, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts with different initializations
        for restart in range(15):
            if time.time() > deadline - 3:
                break
            
            # Different initialization strategies
            if restart < 5:
                # Small random initialization
                U_r = np.random.randn(n * m, target_rank) * 0.1
                V_r = np.random.randn(m * p, target_rank) * 0.1
                W_r = np.random.randn(p * n, target_rank) * 0.1
            elif restart < 10:
                # Moderate random initialization
                U_r = np.random.randn(n * m, target_rank) * 0.5
                V_r = np.random.randn(m * p, target_rank) * 0.5
                W_r = np.random.randn(p * n, target_rank) * 0.5
            else:
                # Sparse-like initialization
                U_r = np.random.choice([-1, 0, 1], size=(n * m, target_rank)) * 0.3
                V_r = np.random.choice([-1, 0, 1], size=(m * p, target_rank)) * 0.3
                W_r = np.random.choice([-1, 0, 1], size=(p * n, target_rank)) * 0.3
            
            # Adaptive learning rate gradient descent
            lr = 0.01
            tol = 1e-10
            max_iter = 200
            best_error = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                # Compute reconstruction
                recon = compute_recon(U_r, V_r, W_r, n, m, p, target_rank)
                
                # Compute error
                error = np.sum((recon - target_tensor) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                
                if error < tol:
                    break
                
                # Compute gradients
                grad_U, grad_V, grad_W = compute_gradients(U_r, V_r, W_r, recon, target_tensor, n, m, p, target_rank)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    effective_lr = lr * (1 + error) / (1 + grad_norm)
                else:
                    effective_lr = lr
                
                # Update
                U_r -= effective_lr * grad_U
                V_r -= effective_lr * grad_V
                W_r -= effective_lr * grad_W
            
            # Check final error
            recon = compute_recon(U_r, V_r, W_r, n, m, p, target_rank)
            final_error = np.sum((recon - target_tensor) ** 2)
            
            if final_error < tol and target_rank < best_rank:
                best_rank = target_rank
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                break  # Found a valid decomposition at this rank
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
