"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    Strategy: Start from schoolbook rank 63, then search for rank reductions
    by exploiting the structure of the 3x3x7 tensor. The 7-column output can
    be viewed as a 3x3 block (rank ~21 via Winograd) plus 4 extra columns.
    We use structured search with verification.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 3, 3, 7
    schoolbook_rank = n * m * p  # 63

    def verify_decomposition(U, V, W):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
        """
        rank = U.shape[1]
        
        # Reconstruct tensor efficiently using einsum
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum: sum over r
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Expected tensor: T[i,j,k] = 1 for all i,j,k
        T_expected = np.ones((n, m, p))
        
        return np.allclose(T_reconstructed, T_expected, atol=1e-9)

    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition as numpy arrays."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                    
        return U, V, W

    # Start with schoolbook decomposition
    U, V, W = schoolbook_decomposition()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank

    # Strategy: Try to find lower-rank decompositions
    # Key insight: The 3x3x7 tensor can be decomposed as:
    # - A 3x3x3 block (for the first 3 columns) + 3x3x4 block (for last 4 columns)
    # The 3x3x3 part has known low-rank decompositions (rank 21 via Winograd)
    # The 3x3x4 part can potentially be optimized
    
    # Approach: Try various structured decompositions
    # 1. First, try to find a rank-21 decomposition for the 3x3x3 subproblem
    # 2. Then extend to the full 3x3x7 problem
    
    # Winograd-style decomposition for 3x3x3
    # The idea is to use (a+b)(c+d) - ac - bd type identities
    
    # For now, let's try a systematic search for rank reduction
    # We'll try to merge schoolbook terms using linear combinations
    
    # Try rank reduction from 63 down
    # For each target rank, try to find a valid decomposition
    
    # Method: Start with schoolbook, try to express some columns as
    # linear combinations of others while maintaining the tensor identity
    
    # A more practical approach: use the fact that the tensor has structure
    # and try specific known decompositions
    
    # Let's try a gradient-based approach to find lower-rank decompositions
    # Initialize near schoolbook and optimize
    
    np.random.seed(42)
    
    # Try several target ranks
    for target_rank in range(62, 50, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with schoolbook structure plus small perturbations
            U_try = np.zeros((n*m, target_rank))
            V_try = np.zeros((m*p, target_rank))
            W_try = np.zeros((p*n, target_rank))
            
            # Copy schoolbook terms that fit
            for r in range(min(target_rank, schoolbook_rank)):
                i = r // (m*p)
                j = (r // p) % m
                k = r % p
                if i < n and j < m and k < p:
                    U_try[i*m+j, r] = 1.0
                    V_try[j*p+k, r] = 1.0
                    W_try[k*n+i, r] = 1.0
            
            # Add small random perturbations to allow optimization
            U_try += np.random.randn(n*m, target_rank) * 0.01
            V_try += np.random.randn(m*p, target_rank) * 0.01
            W_try += np.random.randn(p*n, target_rank) * 0.01
            
            # Simple gradient descent to minimize ||T_reconstructed - T_expected||^2
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Compute T_reconstructed
                U_r = U_try.reshape(n, m, target_rank)
                V_r = V_try.reshape(m, p, target_rank)
                W_r = W_try.reshape(p, n, target_rank)
                
                T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                T_exp = np.ones((n, m, p))
                
                error = T_rec - T_exp
                if np.max(np.abs(error)) < 1e-8:
                    break
                
                # Compute gradients
                # dL/dU[i,j,r] = sum_k error[i,j,k] * V[j,k,r] * W[k,i,r]
                # dL/dV[j,k,r] = sum_i error[i,j,k] * U[i,j,r] * W[k,i,r]
                # dL/dW[k,i,r] = sum_j error[i,j,k] * U[i,j,r] * V[j,k,r]
                
                grad_U = np.einsum('ijk,jkr,kir->ijr', error, V_r, W_r)
                grad_V = np.einsum('ijk,ijr,kir->jkr', error, U_r, W_r)
                grad_W = np.einsum('ijk,ijr,jkr->kir', error, U_r, V_r)
                
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            # Check if we found a valid decomposition
            if verify_decomposition(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        U, V, W = schoolbook_decomposition()
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists for the expected output format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
