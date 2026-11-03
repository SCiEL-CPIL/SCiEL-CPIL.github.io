"""Rewritten program for matrix multiplication 3,5,5 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D array T[i,j,k] with 1 if the (i,k) entry of the product depends on A[i,j]*B[j,k].
    For C = A B, C[i,k] = sum_j A[i,j] B[j,k].
    So the tensor M[i,j,k] = 1 for all i,j,k in ranges."""
    return np.ones((n, m, p), dtype=float)

def _tensor_to_matrices(n, m, p, rank, U, V, W):
    """Given U (n*m, rank), V (m*p, rank), W (p*n, rank), compute the implied tensor.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] ... wait no.
    
    The standard tensor decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?
    
    Actually for matrix multiplication, the convention is:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T[i,j,k] has T[i,j,k] = 1.
    
    A rank-r decomposition means T[i,j,k] = sum_{c=1}^{r} u_c[i,j] * v_c[j,k] * w_c[k,i]
    where u_c is an n x m matrix, v_c is an m x p matrix, w_c is a p x n matrix.
    
    In the interface:
    U has shape (n*m, rank): U[i*m + j, c] = u_c[i,j]
    V has shape (m*p, rank): V[j*p + k, c] = v_c[j,k]  
    W has shape (p*n, rank): W[k*n + i, c] = w_c[k,i]
    
    So T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
    # Use einsum
    return np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)

def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if the decomposition is exact."""
    T_expected = np.ones((n, m, p), dtype=float)
    T_computed = _tensor_to_matrices(n, m, p, U.shape[1], U, V, W)
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < tol, diff

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using random initialization + gradient descent
    # The tensor is small (3x5x5 = 75 entries), so we can try various ranks
    
    # Known lower bounds: for 3x5x5, schoolbook is 75. 
    # Let's try ranks from 50 down to see if we can find valid decompositions.
    
    # Strategy: For each candidate rank, try multiple random initializations and 
    # use gradient descent to minimize ||T_computed - T_expected||_F^2
    
    def try_rank(r, n_init=10, max_iter=2000, lr=0.01):
        """Try to find a rank-r decomposition."""
        nonlocal best_rank, best_U, best_V, best_W
        nonlocal deadline
        
        for init_idx in range(n_init):
            if time.time() > deadline:
                return
            
            # Random initialization
            U_r = np.random.randn(n * m, r) * 0.1
            V_r = np.random.randn(m * p, r) * 0.1
            W_r = np.random.randn(p * n, r) * 0.1
            
            # Also try structured initializations
            if init_idx == 0:
                # Initialize with some structure
                U_r = np.zeros((n * m, r))
                V_r = np.zeros((m * p, r))
                W_r = np.zeros((p * n, r))
                # Try to reuse some schoolbook terms
                for c in range(min(r, n*m*p)):
                    i = c // (m*p)
                    j = (c // p) % m
                    k = c % p
                    if i*m+j < n*m and j*p+k < m*p and k*n+i < p*n:
                        U_r[i*m+j, c] = 1.0
                        V_r[j*p+k, c] = 1.0
                        W_r[k*n+i, c] = 1.0
            
            # Gradient descent
            for it in range(max_iter):
                if time.time() > deadline:
                    return
                
                U_mat = U_r.reshape(n, m, r)
                V_mat = V_r.reshape(m, p, r)
                W_mat = W_r.reshape(p, n, r)
                
                # Compute gradient of ||T_comp - T_exp||_F^2
                # T_comp[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
                # dL/dU_mat[i,j,c] = (T_comp[i,j,k] - 1) * V_mat[j,k,c] * W_mat[k,i,c]
                # Sum over k
                
                T_comp = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
                error = T_comp - 1.0  # T_expected is all ones
                
                # Gradients
                dU = np.einsum('ijk,jkc,kic->ijc', error, V_mat, W_mat)
                dV = np.einsum('ijk,ijc,kic->jkc', error, U_mat, W_mat)
                dW = np.einsum('ijk,ijc,jkc->kic', error, U_mat, V_mat)
                
                # Update
                U_r -= lr * dU.reshape(n*m, r)
                V_r -= lr * dV.reshape(m*p, r)
                W_r -= lr * dW.reshape(p*n, r)
                
                # Check convergence
                if it % 50 == 0:
                    diff = np.max(np.abs(T_comp - 1.0))
                    if diff < 1e-9:
                        if r < best_rank:
                            best_rank = r
                            best_U = U_r.copy()
                            best_V = V_r.copy()
                            best_W = W_r.copy()
                        return True
            else:
                # Final check
                U_mat = U_r.reshape(n, m, r)
                V_mat = V_r.reshape(m, p, r)
                W_mat = W_r.reshape(p, n, r)
                T_comp = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
                diff = np.max(np.abs(T_comp - 1.0))
                if diff < 1e-9:
                    if r < best_rank:
                        best_rank = r
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                    return True
        
        return False
    
    # Try decreasing ranks
    for r in range(70, 30, -1):
        if time.time() > deadline:
            break
        if try_rank(r, n_init=5, max_iter=1000, lr=0.005):
            # Found a valid decomposition at this rank, try to go lower
            continue
        # If we fail at this rank, we might still try lower but it's less likely
        # Actually, if we can't find rank r, we might still find rank r-1, so keep trying
        # But to save time, let's try a few more initializations for the current best
    
    # Convert to list format for output
    def to_list(mat):
        return [[mat[i][j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
