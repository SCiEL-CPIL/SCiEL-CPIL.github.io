"""Matrix multiplication 3x4 by 4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _validate(U, V, W):
    """Check that U, V, W form a valid decomposition of the matmul tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The matmul tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k] -> C[i,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few random instances
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            # Extract submatrices
            Ar = U_arr[:, r].reshape(n, m)
            Br = V_arr[:, r].reshape(m, p)
            Cr = W_arr[:, r].reshape(p, n)
            # The decomposition gives: sum_r Ar * Br * Cr (with appropriate indexing)
            # Actually: C[i,k] = sum_r sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
            # Let's compute properly
            pass
        
        # Better: compute the decomposed product
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            Ar = U_arr[:, r].reshape(n, m)
            Br = V_arr[:, r].reshape(m, p)
            Cr = W_arr[:, r].reshape(p, n)
            # C[i,k] += sum_j Ar[i,j] * A[i,j] * Br[j,k] * B[j,k] * Cr[k,i]
            # This is element-wise: (Ar*A) * (Br*B) contracted with Cr
            temp = (Ar * A) @ (Br * B)  # n x p
            C_decomposed += temp * Cr.T  # Cr is p x n, so Cr.T is n x p
            # Wait, Cr[k*n+i] so Cr is indexed [k*n+i], shape (p*n,). Reshape to (p, n).
            # So Cr[k,i] is the element. We need C[i,k] += sum_j ... * Cr[k,i]
            # temp[i,k] = sum_j Ar[i,j]*A[i,j]*Br[j,k]*B[j,k]
            # Then C[i,k] += temp[i,k] * Cr[k,i]
            # Cr.T[i,k] = Cr[k,i]. So C_decomposed += temp * Cr.T
            # That's what we did. Good.
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook():
    """Standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _compute_tensor_residual(T, U, V, W):
    """Compute the residual tensor R = T - sum_r U[:,r] V[:,r] W[:,r]"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    R = np.zeros_like(T)
    for r in range(rank):
        R -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return R

def _gradient_descent_step(T, U, V, W, lr):
    """One gradient descent step to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||^2"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Compute residual
    R = _compute_tensor_residual(T, U, V, W)
    
    # Compute gradients
    # d/dU[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
    # d/dV[b,r] = sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
    # d/dW[c,r] = sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
    
    # Efficient computation using einsum
    # For U gradient: sum over b,c of R[a,b,c] * V[b,r] * W[c,r]
    # = einsum('abc,br,cr->ar', R, V, W)
    grad_U = np.einsum('abc,br,cr->ar', R, V, W)
    grad_V = np.einsum('abc,ar,cr->br', R, U, W)
    grad_W = np.einsum('abc,ar,br->cr', R, U, V)
    
    # Update
    U_new = U - lr * grad_U
    V_new = V - lr * grad_V
    W_new = W - lr * grad_W
    
    return U_new, V_new, W_new

def _try_find_decomposition(T, target_rank, max_iters, deadline, seed=0):
    """Try to find a rank-target_rank decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Adaptive learning rate
    lr = 0.01
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current error
        R = _compute_tensor_residual(T, U, V, W)
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check convergence
        if err < 1e-8:
            # Verify with the actual validation
            U_list = [[best_U[i, r] for r in range(target_rank)] for i in range(n * m)]
            V_list = [[best_V[i, r] for r in range(target_rank)] for i in range(m * p)]
            W_list = [[best_W[i, r] for r in range(target_rank)] for i in range(p * n)]
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list, True
        
        # Gradient step
        U, V, W = _gradient_descent_step(T, U, V, W, lr)
        
        # Adaptive learning rate: decrease if error increases
        R_new = _compute_tensor_residual(T, U, V, W)
        err_new = np.linalg.norm(R_new)
        if err_new > err * 1.5:
            lr *= 0.5
            # Revert
            U, V, W = best_U, best_V, best_W
        elif err_new < err * 0.9:
            lr *= 1.1  # Increase slightly if making good progress
            lr = min(lr, 0.1)  # Cap learning rate
    
    # Final check
    R = _compute_tensor_residual(T, best_U, best_V, best_W)
    err = np.linalg.norm(R)
    if err < 1e-6:
        U_list = [[best_U[i, r] for r in range(target_rank)] for i in range(n * m)]
        V_list = [[best_V[i, r] for r in range(target_rank)] for i in range(m * p)]
        W_list = [[best_W[i, r] for r in range(target_rank)] for i in range(p * n)]
        if _validate(U_list, V_list, W_list):
            return U_list, V_list, W_list, True
    
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Start with schoolbook as best known
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But realistically, for matrix multiplication, the rank is much higher
    
    # Try ranks from 95 down to some lower bound
    # Use multiple seeds and adaptive learning rates
    
    max_iters_per_attempt = 500
    attempts_per_rank = 3
    
    for target_rank in range(schoolbook_rank - 1, 79, -1):  # Try ranks 95 down to 80
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(attempts_per_rank):
            if time.time() > deadline - 5:
                break
            
            seed = target_rank * 100 + attempt
            U_list, V_list, W_list, success = _try_find_decomposition(
                T, target_rank, max_iters_per_attempt, deadline, seed=seed
            )
            
            if success:
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    # Return the best result found
    return best_result
