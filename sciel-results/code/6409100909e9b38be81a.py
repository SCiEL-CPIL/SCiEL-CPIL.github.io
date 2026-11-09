"""Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS
N = n * m * p  # 40

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop searching

def _time_left():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _verify(U, V, W, tol=1e-6):
    """Verify that U @ diag(W) @ V^T reconstructs the multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_c U[i,j,c] * W[k,i,c] * V[j,k,c]
    # We need to check that this equals the identity mapping for matrix mult.
    # The multiplication tensor: T[i,j,k] = 1 if (i,j,k) is a valid entry, else 0
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which maps A[i,j]*B[j,k] to C[i,k]
    # So the decomposition is: C[i,k] = sum_c U[i*m+j... wait, let me re-read.
    
    # From the schoolbook: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So U is indexed by (i,j) flattened as i*m+j, shape (n*m, rank)
    # V is indexed by (j,k) flattened as j*p+k, shape (m*p, rank)  
    # W is indexed by (k,i) flattened as k*n+i, shape (p*n, rank)
    
    # The reconstruction: for each (i,j,k), the product should give 1 if it's part of the sum
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the full tensor and check
    # T should be all 1s for the 2*4*5 = 40 entries
    # Let's verify by computing the reconstructed tensor
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each c, the contribution is outer product structure
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Efficient check: for random test vectors
    # Actually let's just build it directly for small sizes
    # n=2, m=4, p=5, so 40 entries total - small enough
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i,j,k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
    
    # The target tensor is all 1s
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=tol)

def _verify_fast(U, V, W, tol=1e-6):
    """Faster verification using numpy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Reshape for broadcasting
    # U: (n*m, rank) -> (n, m, rank)
    U_r = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_r = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # W_r[k,i,c] needs to be (n, p, rank) indexed as [i, k, c]
    W_t = W_r.transpose(1, 0, 2)  # (n, p, rank)
    
    # Now: U_r[i,j,c] * V_r[j,k,c] * W_t[i,k,c]
    # Sum over c
    # U_r: (n, m, rank), V_r: (m, p, rank), W_t: (n, p, rank)
    # Product: U_r[:, :, :] * V_r[:, :, :] would need broadcasting...
    # Let's do it step by step
    # temp[i,j,k,c] = U_r[i,j,c] * V_r[j,k,c]
    # Then T[i,j,k] = sum_c temp[i,j,k,c] * W_t[i,k,c]
    
    # U_r[:, :, None, :] * V_r[None, :, :, :] -> (n, m, p, rank)
    temp = U_r[:, :, None, :] * V_r[None, :, :, :]  # (n, m, p, rank)
    T_recon = np.sum(temp * W_t[:, None, :, :], axis=3)  # (n, m, p)
    
    return np.allclose(T_recon, np.ones((n, m, p)), atol=tol)

def _schoolbook():
    """Return the schoolbook decomposition of rank 40."""
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

def _try_rank(r, max_iter=5000, seed=None):
    """Try to find a decomposition of rank r using random initialization + gradient descent."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Random initialization with small values
    for attempt in range(3):
        if _time_left() < 5:
            break
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Gradient descent on the loss ||T_recon - T_target||^2
        lr = 0.01
        for it in range(max_iter):
            if _time_left() < 10:
                break
            
            # Compute T_recon
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            W_t = W_r.transpose(1, 0, 2)  # (n, p, rank)
            
            temp = U_r[:, :, None, :] * V_r[None, :, :, :]  # (n, m, p, r)
            T_recon = np.sum(temp * W_t[:, None, :, :], axis=3)  # (n, m, p)
            error = T_recon - 1.0  # (n, m, p)
            
            # Check convergence
            if np.max(np.abs(error)) < 1e-8:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Compute gradients
            # dL/dU_r[i,j,c] = sum_k error[i,j,k] * V_r[j,k,c] * W_t[i,k,c]
            # dL/dV_r[j,k,c] = sum_i error[i,j,k] * U_r[i,j,c] * W_t[i,k,c]
            # dL/dW_t[i,k,c] = sum_j error[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
            
            # grad_U: (n, m, r)
            # grad_U[i,j,c] = sum_k error[i,j,k] * V_r[j,k,c] * W_t[i,k,c]
            # = einsum('ijk,jkc,ikc->ijc', error, V_r, W_t)
            grad_U = np.einsum('ijk,jkc,ikc->ijc', error, V_r, W_t)
            # grad_V: (m, p, r)
            # grad_V[j,k,c] = sum_i error[i,j,k] * U_r[i,j,c] * W_t[i,k,c]
            grad_V = np.einsum('ijk,ijc,ikc->jkc', error, U_r, W_t)
            # grad_W_t: (n, p, r)
            # grad_W_t[i,k,c] = sum_j error[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
            grad_W_t = np.einsum('ijk,ijc,jkc->ikc', error, U_r, V_r)
            
            # Update
            U_r -= lr * grad_U
            V_r -= lr * grad_V
            W_t -= lr * grad_W_t
            
            # Convert back
            U = U_r.reshape(n * m, r)
            V = V_r.reshape(m * p, r)
            W = W_t.transpose(1, 0, 2).reshape(p * n, r)
            
            # Adaptive learning rate
            if it % 100 == 99:
                loss = np.sum(error ** 2)
                if loss < 1e-4:
                    lr = max(lr * 0.5, 0.0001)
        
        # Final check
        if _verify_fast(U, V, W, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_with_snip(r, max_iter=3000):
    """Try to find decomposition of rank r using SVD-based refinement."""
    # Start from schoolbook and try to merge components
    # This is more complex; for now, use random init + GD
    return _try_rank(r, max_iter=max_iter)

def construct():
    """Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    # The schoolbook rank is 40. We want to find if a lower rank exists.
    # Known bounds: for m x n matrix mult, rank is between mn and m^2 n (or similar).
    # For 2x4 * 4x5: the trivial rank is 40.
    # Let's try to find a lower rank.
    
    # First, return schoolbook as fallback
    fallback = _schoolbook()
    
    # Try ranks from 39 down to some minimum
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But in practice, for these small sizes, let's try a few ranks
    
    best_result = fallback
    best_rank = n * m * p
    
    # Try to find lower rank
    for r in range(best_rank - 1, 19, -1):
        if _time_left() < 15:
            break
        
        result = _try_rank(r, max_iter=2000)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, tol=1e-6):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                # Don't break; try even lower
                break  # Found one, try lower
    
    # If we found a lower rank, try to go even lower
    if best_rank < n * m * p:
        for r in range(best_rank - 1, 19, -1):
            if _time_left() < 15:
                break
            result = _try_rank(r, max_iter=2000, seed=r)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, tol=1e-6):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    break
    
    return best_result
