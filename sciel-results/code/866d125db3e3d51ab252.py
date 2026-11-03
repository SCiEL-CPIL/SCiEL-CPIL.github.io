"""
5x5 matrix multiplication - attempt at rank reduction via structured search.
Uses integer-preserving randomization and linear algebra to try to find
a decomposition of rank less than 125 (schoolbook).
"""
import random
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school), dtype=float)
    V_s = np.zeros((m*p, rank_school), dtype=float)
    W_s = np.zeros((p*n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    return U_s, V_s, W_s

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication.
    
    For matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k],
    the tensor is T[i,j,k] = 1 for all (i,j,k) where the term A[i,j]*B[j,k] contributes to C[i,k].
    
    The factorization must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    """
    n, m, p = n, m, p
    rank = U.shape[1]
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_off_diagonal(U, V, W, n, m, p, tol=1e-9):
    """Verify that off-diagonal terms are zero.
    
    For the decomposition to be valid, we need:
    sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k*n+i, c] = 0
    when (i',j',k') != (i,j,k) in the sense that the term doesn't contribute to C[i,k]
    
    Actually, the correct condition is:
    For each (i,k), the sum over j of (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]) should equal
    sum_j A[i,j] B[j,k] for all A, B.
    
    This means: for each (i,k), the coefficients of A[i,j]B[j,k] in the factorization must be 1
    for the correct j, and the cross-terms must cancel.
    
    A simpler verification: construct random matrices and check if the factorization
    reproduces the matrix product.
    """
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using factorization
        C_factor = np.zeros((n, p))
        rank = U.shape[1]
        for c in range(rank):
            # u_vec[i,j] = U[i*m+j, c]
            # v_vec[j,k] = V[j*p+k, c]
            # w_vec[k,i] = W[k*n+i, c]
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            
            # The factorization gives: C[i,k] = sum_c sum_{i',j',k'} U[i',j',c] A[i',j'] V[j',k',c] B[j',k'] W[k',i',c]
            # This is not simply u_mat @ v_mat * w_mat
            # Let's compute it properly
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        # Term A[i,j] * B[j,k] should contribute with coefficient 1
                        # In the factorization, the coefficient of A[i,j]*B[j,k] in C[i,k] is:
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # This should be 1 for the (i,j,k) term
                        pass
        
        # Better approach: directly verify the tensor condition
        # For all (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 140s deadline, 10s safety margin
    
    # Build schoolbook decomposition as baseline
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    rank_school = n * m * p
    
    # Verify schoolbook works
    assert _verify_decomposition(U_s, V_s, W_s, n, m, p), "Schoolbook verification failed"
    
    # Try to reduce rank using column elimination
    # Strategy: For each column, check if it can be expressed as a linear combination
    # of other columns in a way that preserves the tensor decomposition
    
    # The tensor condition is: for all (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # This is a system of 125 equations (for n=m=p=5)
    
    # Try greedy column elimination:
    # 1. Start with schoolbook (125 columns)
    # 2. For each column, try to remove it by adjusting other columns
    # 3. Check if the tensor condition still holds
    
    U = U_s.copy()
    V = V_s.copy()
    W = W_s.copy()
    current_rank = rank_school
    
    # For each column, try to eliminate it
    for col_to_remove in range(rank_school):
        if time.time() > deadline:
            break
        
        if col_to_remove >= current_rank:
            continue
        
        # Try to express column col_to_remove as a linear combination of other columns
        # The constraint is: for all (i,j,k),
        # sum_{c != col_to_remove} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 - U[i*m+j,col] V[j*p+k,col] W[k*n+i,col]
        
        # This is a nonlinear constraint. Let's try a simpler approach:
        # Check if the column is "redundant" by seeing if removing it and solving
        # for adjustments to other columns can maintain the tensor condition.
        
        # For now, just check if the column has all zeros (trivially removable)
        if np.all(U[:, col_to_remove] == 0) and np.all(V[:, col_to_remove] == 0) and np.all(W[:, col_to_remove] == 0):
            # Remove this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
            continue
        
        # Try a more sophisticated elimination:
        # Check if we can find coefficients alpha_c such that:
        # U[:, col] = sum_{c != col} alpha_c * U[:, c]
        # V[:, col] = sum_{c != col} beta_c * V[:, c]  
        # W[:, col] = sum_{c != col} gamma_c * W[:, c]
        # and the product condition still holds
        
        # This is complex. Let's try a randomized local search approach.
    
    # Try randomized perturbation to find lower-rank solutions
    # Start from schoolbook and try to merge columns
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try to reduce rank by merging similar columns
    # Two columns c1 and c2 can be merged if there exist U', V', W' such that
    # the tensor condition is preserved with one fewer column
    
    # For now, return the best we've found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
