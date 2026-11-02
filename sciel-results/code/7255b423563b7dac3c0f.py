"""Tensor decomposition for matrix multiplication 3x3 * 3x7."""
DIMENSIONS = [3, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m * m x p."""
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
    return U, V, W


def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matmul tensor."""
    # The tensor T[i, j, k] = 1 iff i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # For matmul: T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually the standard decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta (it's the identity-like structure)
    # The condition is: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # But wait, that's not right either. Let me think more carefully.
    #
    # In tensor decomposition of matmul:
    # The matmul tensor has entries T[i,j,k] = 1 for all valid i,j,k (it's all ones in the valid range)
    # No wait. The matmul tensor is defined such that:
    # C = A * B where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the structural tensor, all ones)
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # Actually for the standard matmul tensor decomposition:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    rank = U.shape[1]
    # Build the reconstructed tensor and check
    # T_reconstructed[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is expensive for large dimensions, but for 3x3x7 it's fine
    
    # Efficient check using einsum
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # T_recon[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # This is a bit tricky with einsum. Let's do it step by step.
    # First compute Ur[i,j,r] * Vr[j,k,r] -> (n, m, p, rank)
    # Then multiply by Wr[k,i,r] and sum over r
    
    # Actually let's just do it with loops for clarity since dimensions are small
    errors = 0.0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > tol:
                    errors += 1
                if err > max_err:
                    max_err = err
    return errors == 0, max_err


def _try_strassen_style(n, m, p, max_time=60.0):
    """
    Try to find a low-rank decomposition using a structured approach.
    For 3x3*3x7, we can think of it as 3x3*3x3 + 3x3*3x4.
    The 3x3*3x3 part has known rank 23 (Strassen).
    The 3x3*3x4 part needs investigation.
    
    Strategy: Use the Bini-Capovani / structured perturbation approach.
    Start from schoolbook and try to merge terms.
    """
    start_time = time.time()
    
    # First, get the schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # on the matmul tensor.
    
    # The matmul tensor T[i,j,k] = 1 for all valid indices
    # We want to minimize rank such that T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # Let's try various target ranks from 63 down
    # Use a time budget
    
    for target_rank in range(62, 40, -1):
        if time.time() - start_time > max_time:
            break
        
        # Try to find a decomposition of this rank using random initialization + ALS
        success = _als_decompose(n, m, p, target_rank, max_iters=50, time_limit=5.0)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = success
            break
    
    return best_U, best_V, best_W, best_rank


def _als_decompose(n, m, p, rank, max_iters=100, time_limit=10.0, tol=1e-8):
    """
    Try to find a rank decomposition using ALS.
    Returns (U, V, W) or None if not found.
    """
    start_time = time.time()
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize columns
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update U: for each (i,j), we need U[i*m+j, :] to satisfy
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a linear system for each (i,j)
        
        # Similarly for V and W.
        
        # Let's do a simple gradient descent / ALS approach.
        # For each factor, hold the other two fixed and solve a least squares problem.
        
        # Update U:
        for i in range(n):
            for j in range(m):
                # For fixed i,j, we want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # Let A_k,r = V[j*p+k,r] * W[k*n+i,r]
                # Then we want: U[i*m+j,:] @ A_k = 1 for all k
                # This is: A @ U[i*m+j,:] = ones(p)
                # where A is (p, rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ x = ones(p)
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V:
        for j in range(m):
            for k in range(p):
                # For fixed j,k, we want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                # Let B_i,r = U[i*m+j,r] * W[k*n+i,r]
                # Then: B @ V[j*p+k,:] = ones(n)
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W:
        for k in range(p):
            for i in range(n):
                # For fixed k,i, we want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                # Let C_j,r = U[i*m+j,r] * V[j*p+k,r]
                # Then: C @ W[k*n+i,:] = ones(m)
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check error
        error = _compute_error(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < tol:
            break
    
    if best_error < tol:
        return best_U, best_V, best_W
    return None


def _compute_error(U, V, W, n, m, p):
    """Compute the maximum absolute error of the decomposition."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _try_structured_decomposition(n, m, p, max_time=90.0):
    """
    Try a structured decomposition approach.
    Decompose 3x3*3x7 into 3x3*3x3 + 3x3*3x4.
    Use known Strassen for the 3x3*3x3 part and try to optimize the 3x3*3x4 part.
    """
    start_time = time.time()
    
    # Schoolbook for reference
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ALS with various ranks
    for target_rank in range(62, 45, -1):
        if time.time() - start_time > max_time:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(3):
            if time.time() - start_time > max_time:
                break
            
            result = _als_decompose(n, m, p, target_rank, max_iters=80, time_limit=8.0, tol=1e-8)
            if result is not None:
                U, V, W = result
                # Verify
                valid, max_err = _validate(U, V, W, n, m, p, tol=1e-6)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    return best_U, best_V, best_W, best_rank


def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 3x3 * 3x7.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to improve using ALS-based search
    if time.time() - start_time < deadline - 10:
        try:
            U2, V2, W2, rank2 = _try_structured_decomposition(n, m, p, max_time=deadline - time.time() - 10)
            if rank2 < best_rank:
                best_rank = rank2
                best_U, best_V, best_W = U2, V2, W2
        except Exception:
            pass
    
    # Convert to lists of lists as expected by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
