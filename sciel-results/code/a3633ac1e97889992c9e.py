"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the index
    triple (i,j,k) corresponds to the matrix product, else 0.
    
    Actually, for matrix multiplication, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p),
    and T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    
    The standard tensor for matrix multiplication has T[i,j,k] = delta_{i,j,k} in the sense that
    C[i,k] = sum_j A[i,j] B[j,k].
    
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the convention. The output factor W[k*n+i, r].
    
    For matrix multiplication of A (n x m) and B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 for all valid (i,j,k) because the product involves all such triples.
    
    Actually no. The matrix multiplication tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    This can be written as a tensor contraction. The "multiplication tensor" T has entries
    T[i,j,k] = 1 for all i, j, k, and the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not right either. Let me think more carefully.
    
    The standard schoolbook decomposition has rank n*m*p with:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    
    This gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for each (i,j,k)
    
    But for matrix multiplication, we need:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor that encodes this is T[i,j,k] = 1 for all (i,j,k), because each term A[i,j]*B[j,k]
    contributes to C[i,k].
    
    So the decomposition T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] with all T entries = 1
    is what we need.
    
    The question is: can we do this with rank < n*m*p?
    
    For small matrices, the minimum rank of matrix multiplication is known in some cases.
    For 3x5 by 5x7, I don't know the exact minimum rank.
    
    Let me try to find a lower rank decomposition using optimization.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank via optimization
    # The constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a non-convex optimization. Let's try a simple approach:
    # Start with a random low-rank initialization and use gradient descent.
    
    start_time = time.time()
    deadline = 145.0  # Leave 5 seconds margin
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various target ranks, starting from lower bounds
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        max_trials = 5 if target_rank < 50 else 3
        for trial in range(max_trials):
            if time.time() > deadline - 25:
                break
            
            # Random initialization with structured components
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Scale to reasonable magnitude
            scale = 1.0 / np.cbrt(target_rank)
            U_r *= scale
            V_r *= scale
            W_r *= scale
            
            # Adam optimizer state
            m_U = np.zeros_like(U_r)
            v_U = np.zeros_like(U_r)
            m_V = np.zeros_like(V_r)
            v_V = np.zeros_like(V_r)
            m_W = np.zeros_like(W_r)
            v_W = np.zeros_like(W_r)
            t = 0
            lr = 0.1
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            converged = False
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                # Compute residuals and gradients efficiently
                residuals = np.zeros((n, m, p))
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            
                            val = np.dot(U_r[u_idx, :], V_r[v_idx, :])
                            val = np.dot(val, W_r[w_idx, :])
                            residuals[i, j, k] = val - 1.0
                            
                            r = residuals[i, j, k]
                            dU[u_idx, :] += r * V_r[v_idx, :] * W_r[w_idx, :]
                            dV[v_idx, :] += r * U_r[u_idx, :] * W_r[w_idx, :]
                            dW[w_idx, :] += r * U_r[u_idx, :] * V_r[v_idx, :]
                
                # Check convergence
                max_res = np.max(np.abs(residuals))
                if max_res < 1e-8:
                    converged = True
                    break
                
                # Adam update
                t += 1
                lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
                
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * dU**2
                U_r -= lr_t * m_U / (np.sqrt(v_U) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * dV**2
                V_r -= lr_t * m_V / (np.sqrt(v_V) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * dW**2
                W_r -= lr_t * m_W / (np.sqrt(v_W) + eps)
                
                # Periodic learning rate decay
                if step % 500 == 0 and step > 0:
                    lr *= 0.5
            
            if converged and target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
    
    # Convert to lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
