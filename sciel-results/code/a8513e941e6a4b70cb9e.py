"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_einsum(U, V, W, tol=1e-10):
    """Use einsum for verification."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _greedy_column_removal(U, V, W, deadline, max_attempts=10):
    """Greedy column removal to reduce rank."""
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            order = list(range(current_rank))
        elif attempt == 1:
            order = list(range(current_rank - 1, -1, -1))
        else:
            order = list(range(current_rank))
            random.seed(42 + attempt * 7)
            random.shuffle(order)
        
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        rank_try = best_rank
        
        removed = 0
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx >= rank_try:
                continue
            
            U_new = np.delete(U_try, idx, axis=1)
            V_new = np.delete(V_try, idx, axis=1)
            W_new = np.delete(W_try, idx, axis=1)
            
            if _verify_einsum(U_new, V_new, W_new, tol=1e-9):
                U_try = U_new
                V_try = V_new
                W_try = W_new
                rank_try -= 1
                removed += 1
        
        if rank_try < best_rank:
            best_rank = rank_try
            best_U = U_try.copy()
            best_V = V_try.copy()
            best_W = W_try.copy()
    
    return best_U, best_V, best_W

def _try_structured_decomposition(deadline):
    """
    Try to find a structured decomposition using the fact that 
    C[i,k] = sum_j A[i,j] B[j,k] for i in [0,3), j in [0,5), k in [0,5).
    
    We can think of this as three independent 5x5 matrix multiplications:
    For each i in {0,1,2}: C[i,:] = A[i,:] @ B
    
    The tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    One approach: use a rank-12 decomposition for 5x5 if it exists, 
    giving total rank 3*12 = 36.
    
    However, the known minimal rank for 5x5 is 19 (not 12). 
    Let's try a different structural approach.
    
    Actually, let's try to find any rank reduction from 75 using 
    a more systematic search with random initialization and gradient descent.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Try various target ranks
    target_ranks = [70, 65, 60, 55, 50, 45, 40, 36, 35, 34, 33, 32, 31, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for init_seed in range(5):
            if time.time() > deadline - 10:
                break
            
            random.seed(init_seed * 1000 + target_rank)
            np.random.seed(init_seed * 1000 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement
            # We want T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # 
            # This is a non-convex problem. Let's use a simple approach:
            # Fix two matrices, solve for the third linearly.
            
            # For fixed U and V, we can solve for W:
            # For each (k,i): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
            # Wait, that's not quite right. Let me think again.
            #
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            #
            # For fixed U and V, define M[k,i,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then T[i,j,k] = sum_c M[k,i,c] * W[k*n+i,c]... no, that's not right either.
            #
            # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            #
            # Let's define for each c: 
            #   U_c has shape (n,m), V_c has shape (m,p), W_c has shape (p,n)
            #   T_c[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
            #   T[i,j,k] = sum_c T_c[i,j,k]
            #
            # This is hard to solve linearly. Let's just use the greedy approach 
            # and accept that we may not find a rank below 75 easily.
            
            # Instead, let's try a different approach: 
            # Use the fact that the tensor is separable in a certain way.
            
            # Actually, let's just return the schoolbook if we can't find better.
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try greedy column removal
    U_greedy, V_greedy, W_greedy = _greedy_column_removal(U, V, W, deadline, max_attempts=15)
    
    if U_greedy.shape[1] < best_rank:
        best_rank = U_greedy.shape[1]
        best_U = U_greedy.copy()
        best_V = V_greedy.copy()
        best_W = W_greedy.copy()
    
    # Final verification
    if not _verify_einsum(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
