import numpy as np
import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target Tensor indices
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
    
    # Precompute target tensor for verification
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for u, v, w in target_indices:
        TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        # U: (size_u, R), V: (size_v, R), W: (size_w, R)
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def verify_solution(U, V, W, tol=1e-9):
        T = tensor_contraction(U, V, W)
        return np.allclose(T, TARGET_TENSOR, atol=tol)

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    # Schoolbook Solution (Rank 48)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook()
    
    best_solution = None
    best_rank = n * m * p

    # Strategy: Try to reduce rank by finding if any combination of the schoolbook
    # terms can be eliminated. For matrix multiplication, the tensor has a specific
    # structure that might allow for rank reduction.
    
    # Let's try a more systematic approach:
    # 1. Check if the 48 rank-1 tensors are linearly independent
    # 2. If they are, then 48 is the minimal rank
    # 3. If not, try to find a lower rank decomposition
    
    # Flatten each rank-1 tensor to a vector
    def flatten_tensor(u, v, w):
        # u: (size_u,), v: (size_v,), w: (size_w,)
        return np.outer(np.outer(u, v), w).flatten()
    
    # Build the matrix of all 48 rank-1 tensors
    M = np.zeros((size_u * size_v * size_w, n * m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                u = np.zeros(size_u)
                v = np.zeros(size_v)
                w = np.zeros(size_w)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                M[:, col] = flatten_tensor(u, v, w)
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < n * m * p:
        # There are linear dependencies, so we might be able to reduce rank
        # Find the null space
        U_null, s, Vt = np.linalg.svd(M, full_matrices=True)
        null_space = Vt[rank_M:, :].T  # (n*m*p, n*m*p-rank_M)
        
        # For each null vector, we can potentially eliminate one term
        # This is complex, so let's try a simpler approach:
        # Try to find a rank-47 solution by removing one term and solving for the rest
        
        pass
    
    # Let's try a direct approach: for each rank from 47 down to 40,
    # try to find a solution using a more structured initialization
    
    def try_rank_with_structure(R, seed=0):
        if time.time() > deadline - 10:
            return None
            
        np.random.seed(seed)
        
        # Initialize with schoolbook terms, but combine some
        U = np.zeros((size_u, R))
        V = np.zeros((size_v, R))
        W = np.zeros((size_w, R))
        
        # Use the first R-1 schoolbook terms
        for idx in range(min(R - 1, n * m * p)):
            i = idx // (m * p)
            j = (idx // p) % m
            k = idx % p
            U[i * m + j, idx] = 1
            V[j * p + k, idx] = 1
            W[k * n + i, idx] = 1
        
        # Last term is random
        U[:, R-1] = np.random.randn(size_u) * 0.1
        V[:, R-1] = np.random.randn(size_v) * 0.1
        W[:, R-1] = np.random.randn(size_w) * 0.1
        
        # Optimize using gradient descent with adaptive learning rate
        def compute_residual(U, V, W):
            T_comp = tensor_contraction(U, V, W)
            return T_comp - TARGET_TENSOR
        
        def compute_grad(U, V, W):
            T_comp = tensor_contraction(U, V, W)
            residual = T_comp - TARGET_TENSOR
            
            grad_U = np.einsum('uvw,vr,wr->ur', residual, V, W)
            grad_V = np.einsum('uvw,ur,wr->vr', residual, U, W)
            grad_W = np.einsum('uvw,ur,vr->wr', residual, U, V)
            
            return grad_U, grad_V, grad_W
        
        # Adam optimizer
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_U, v_U = np.zeros_like(U), np.zeros_like(U)
        m_V, v_V = np.zeros_like(V), np.zeros_like(V)
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        t = 0
        lr = 0.01
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(300):
            if time.time() > deadline - 5:
                break
                
            residual = compute_residual(U, V, W)
            res_norm = np.linalg.norm(residual)
            
            if res_norm < best_residual:
                best_residual = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < 1e-10:
                break
            
            grad_U, grad_V, grad_W = compute_grad(U, V, W)
            t += 1
            
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            m_U_hat = m_U / (1 - beta1**t)
            m_V_hat = m_V / (1 - beta1**t)
            m_W_hat = m_W / (1 - beta1**t)
            
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            v_U_hat = v_U / (1 - beta2**t)
            v_V_hat = v_V / (1 - beta2**t)
            v_W_hat = v_W / (1 - beta2**t)
            
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        # Final polishing pass with higher precision
        if best_residual < 1e-4:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                residual = compute_residual(U, V, W)
                if np.linalg.norm(residual) < 1e-12:
                    break
                grad_U, grad_V, grad_W = compute_grad(U, V, W)
                U -= 0.001 * grad_U
                V -= 0.001 * grad_V
                W -= 0.001 * grad_W
        
        # Check if we found a good solution
        if best_residual < 1e-6:
            U_s, V_s, W_s = snap_to_half_integers(best_U, best_V, best_W)
            if verify_solution(U_s, V_s, W_s, tol=1e-9):
                return U_s, V_s, W_s
            elif verify_solution(best_U, best_V, best_W, tol=1e-9):
                # Solution exists but can't be snapped to half-integers
                # This is invalid for the final output
                return None
        
        return None
    
    # Try ranks from 47 down to 40 with more restarts for lower ranks
    for R in range(47, 39, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        # More restarts for lower ranks where solution is harder to find
        n_restarts = 3 if R > 44 else 8
        for seed in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            res = try_rank_with_structure(R, seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            break

    if best_solution:
        return best_solution
    else:
        # Return schoolbook solution
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
