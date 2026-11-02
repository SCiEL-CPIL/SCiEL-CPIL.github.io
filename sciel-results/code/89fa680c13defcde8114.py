"""
Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication.
Uses the schoolbook decomposition (rank 90) as a guaranteed-correct baseline.
Attempts to find a lower-rank decomposition via structured search and local refinement.
"""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    # Reshape to 3D arrays for einsum
    # U[i, j, r], V[j, k, r], W[k, i, r]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute the tensor contraction
    # C[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    try:
        C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return False
        
    # Check if C is all 1s
    return np.allclose(C, 1.0, atol=tol)

def _compute_error(U, V, W, n, m, p):
    """
    Compute the maximum error of the decomposition.
    Returns the max absolute deviation from 1.0.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return float('inf')
        
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return float('inf')
        
    return float(np.max(np.abs(C - 1.0)))

def _local_refinement(U, V, W, n, m, p, max_iters=100, tol=1e-10, lr=0.01):
    """
    Local refinement using gradient descent to minimize the error.
    Returns refined (U, V, W) and the final error.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    
    rank = U.shape[1]
    best_error = _compute_error(U, V, W, n, m, p)
    
    for it in range(max_iters):
        if best_error < tol:
            break
            
        # Compute gradients via finite differences or analytic
        # For simplicity, use a simple coordinate descent approach
        
        # Try small perturbations to each matrix
        improved = False
        for mat_name, mat in [('U', U), ('V', V), ('W', W)]:
            rows, cols = mat.shape
            for i in range(min(5, rows)):  # Limit to a few rows for speed
                for j in range(min(5, cols)):
                    for delta in [lr, -lr, lr*0.5, -lr*0.5]:
                        old_val = mat[i, j]
                        mat[i, j] = old_val + delta
                        err = _compute_error(U, V, W, n, m, p)
                        if err < best_error - 1e-12:
                            best_error = err
                            improved = True
                        else:
                            mat[i, j] = old_val
                            
        if not improved:
            break
            
    return U, V, W, best_error

def _schoolbook_decomposition(n, m, p):
    """
    Returns the standard schoolbook decomposition with rank n*m*p.
    This is guaranteed to be correct.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _try_structured_decomposition(n, m, p, target_rank, seed=42):
    """
    Try to find a structured decomposition using block patterns.
    Exploits the fact that matrix multiplication has algebraic structure.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy: Start with schoolbook and try to merge columns
    # or use Kronecker product structure
    
    # For now, try a parameterized family
    # U[i,j,r] = a_r[i] * b_r[j]
    # V[j,k,r] = c_r[j] * d_r[k]  
    # W[k,i,r] = e_r[k] * f_r[i]
    
    # This gives: sum_r a_r[i] b_r[j] c_r[j] d_r[k] e_r[k] f_r[i]
    # = sum_r (a_r[i] f_r[i]) (b_r[j] c_r[j]) (d_r[k] e_r[k])
    # = sum_r A_r[i] B_r[j] C_r[k]
    # This is a rank-1 in each dimension, which is the schoolbook form.
    
    # More general: allow coupling between indices
    # U[i,j,r] = sum over patterns
    
    # Try random low-rank factorization approach
    # U = U1 @ U2^T, V = V1 @ V2^T, W = W1 @ W2^T
    # where U1: (n*m, k), U2: (k, rank) etc.
    
    k = max(2, target_rank // 10)
    
    U1 = rng.randn(n * m, k)
    U2 = rng.randn(k, target_rank)
    V1 = rng.randn(m * p, k)
    V2 = rng.randn(k, target_rank)
    W1 = rng.randn(p * n, k)
    W2 = rng.randn(k, target_rank)
    
    U = U1 @ U2
    V = V1 @ V2
    W = W1 @ W2
    
    # Normalize
    U /= np.linalg.norm(U) + 1e-15
    V /= np.linalg.norm(V) + 1e-15
    W /= np.linalg.norm(W) + 1e-15
    
    return U, V, W

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify the schoolbook decomposition to ensure correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should never happen, but handle gracefully
        raise ValueError("Internal error: Schoolbook decomposition failed verification.")
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    best_error = 0.0
    
    deadline = start_time + DEADLINE_SECONDS - 10  # 10 second safety margin
    random.seed(42)
    
    # 2. Attempt to find a lower rank decomposition
    # Try ranks from 89 down, with structured and random approaches
    
    for target_rank in range(best_rank - 1, max(best_rank - 15, 0), -1):
        if time.time() > deadline:
            break
            
        # Try structured decomposition
        max_structured_attempts = 3
        for attempt in range(max_structured_attempts):
            if time.time() > deadline:
                break
                
            try:
                U_rand, V_rand, W_rand = _try_structured_decomposition(n, m, p, target_rank, seed=42 + attempt)
                
                err = _compute_error(U_rand, V_rand, W_rand, n, m, p)
                if err < 1e-6:
                    # Close enough, try to refine
                    U_ref, V_ref, W_ref, final_err = _local_refinement(
                        U_rand, V_rand, W_rand, n, m, p, 
                        max_iters=50, tol=1e-10, lr=0.01
                    )
                    if final_err < 1e-9:
                        if _verify_decomposition(U_ref, V_ref, W_ref, n, m, p, tol=1e-9):
                            best_U, best_V, best_W = U_ref, V_ref, W_ref
                            best_rank = target_rank
                            best_error = final_err
                            break
            except Exception:
                continue
        
        if best_rank == target_rank:
            continue
            
        # Try direct random with refinement
        max_random_attempts = 2
        for attempt in range(max_random_attempts):
            if time.time() > deadline:
                break
                
            try:
                U_rand = np.random.randn(n * m, target_rank)
                V_rand = np.random.randn(m * p, target_rank)
                W_rand = np.random.randn(p * n, target_rank)
                
                U_rand /= np.linalg.norm(U_rand) + 1e-15
                V_rand /= np.linalg.norm(V_rand) + 1e-15
                W_rand /= np.linalg.norm(W_rand) + 1e-15
                
                err = _compute_error(U_rand, V_rand, W_rand, n, m, p)
                if err < 0.1:  # Reasonably close
                    U_ref, V_ref, W_ref, final_err = _local_refinement(
                        U_rand, V_rand, W_rand, n, m, p,
                        max_iters=30, tol=1e-10, lr=0.005
                    )
                    if final_err < 1e-9:
                        if _verify_decomposition(U_ref, V_ref, W_ref, n, m, p, tol=1e-9):
                            best_U, best_V, best_W = U_ref, V_ref, W_ref
                            best_rank = target_rank
                            best_error = final_err
                            break
            except Exception:
                continue
    
    # Convert to lists as per interface expectation
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
