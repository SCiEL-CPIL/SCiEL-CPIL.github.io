"""Program for matrix multiplication 2x3x7 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    # Start with schoolbook decomposition (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        
        # Check 1: For all (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check 2: For random invalid (a,b,c), T should be 0
        for _ in range(2000):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a valid multiplication term
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    # Strategy: Use algebraic structure to guide rank reduction
    # The multiplication tensor has a specific Kronecker structure
    # We can try to find linear dependencies among the schoolbook terms
    
    # Approach 1: Try to find a lower rank by exploiting the small dimension n=2
    # Since n=2, the output has only 2 rows, which may allow for rank reduction
    
    # Let's try a structured approach:
    # Group terms by output index k*n+i
    # For each output position (k,i), we need sum_j A[i,j]*B[j,k]
    # This is a linear combination of m=3 products
    
    # Try ranks from 41 down, using random projections with verification
    
    for target_rank in range(41, 20, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            
            # Random projection approach
            U_proj = np.random.randn(target_rank, rank_sb)
            V_proj = np.random.randn(target_rank, rank_sb)
            W_proj = np.random.randn(target_rank, rank_sb)
            
            U_new = U_sb @ U_proj.T  # (nm, target_rank)
            V_new = V_sb @ V_proj.T  # (mp, target_rank)  
            W_new = W_sb @ W_proj.T  # (pn, target_rank)
            
            if verify_decomposition(U_new, V_new, W_new):
                best_rank = target_rank
                best_U = U_new.tolist()
                best_V = V_new.tolist()
                best_W = W_new.tolist()
                break
        
        if best_rank < 42:
            break
    
    # If no improvement found, try a more targeted approach
    if best_rank == 42:
        # Try to find rank 41 by looking for a single linear dependency
        # The tensor T has 42 non-zero entries (one for each (i,j,k) triple)
        # If there's a linear relation among the rank-1 tensors, we can reduce rank
        
        # For each potential rank reduction, try to find coefficients
        # such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
        
        # This is equivalent to finding a null vector in the space spanned by
        # the 42 rank-1 tensors
        
        # Let's try a different approach: use the fact that n=2
        # The output C has 2 rows, so C[i,k] = sum_j A[i,j]*B[j,k]
        # For i=0 and i=1, these are separate linear combinations
        
        # Try to merge some terms that contribute to the same output
        # For a fixed (i,k), the terms are (i,j,k) for j=0,1,2
        # These 3 terms always need to be present for correctness
        
        # But maybe we can share rank-1 components between different (i,k) pairs
        # if the coefficients work out
        
        # Let's try a structured rank 40 decomposition
        # Idea: for some (i,k) pairs, use the same rank component with different weights
        
        for target_rank in [41, 40, 39]:
            if time.time() > deadline - 30:
                break
            
            # Try structured approach: fix most terms, optimize a few
            # Start from schoolbook and try to zero out one column
            for zero_col in range(target_rank, 42):
                if time.time() > deadline - 30:
                    break
                
                # Remove column zero_col and see if we can adjust others
                U_test = np.delete(U_sb, zero_col, axis=1)
                V_test = np.delete(V_sb, zero_col, axis=1)
                W_test = np.delete(W_sb, zero_col, axis=1)
                
                # Now we have rank 41, but the tensor is wrong
                # We need to find a correction term that fixes it
                # This is a linear system problem
                
                # The error tensor is the original rank-1 tensor for (i0,j0,k0)
                # where zero_col corresponds to that triple
                i0 = zero_col // (m * p)
                j0 = (zero_col // p) % m
                k0 = zero_col % p
                
                # We need to find U_corr, V_corr, W_corr such that
                # sum_r U_test[a,r]*V_test[b,r]*W_test[c,r] + 
                # U_corr[a]*V_corr[b]*W_corr[c] = T[a,b,c]
                
                # This is still a rank-1 tensor, so we're not reducing rank
                # We need a different approach
                
                # Let's try: can we express the removed term as a combination
                # of other terms?
                # T[i0,j0,k0] = sum of other terms?
                # Generally no, unless there's a special structure
                
                pass
    
    # If still no improvement, return schoolbook
    if best_rank == 42:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
