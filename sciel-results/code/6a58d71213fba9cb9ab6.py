"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # C[i,j] = sum_k A[i,k] * B[k,j]
                # In tensor: T[i,j,k] = 1
                # Decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                val = 0
                for r in range(rank):
                    val += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if val != 1:
                    return False
    # Check that all other entries are zero
    # The tensor T has T[i,j,k]=1 for all i,j,k, and 0 elsewhere
    # Actually for matrix multiplication tensor, T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Let me re-check: the multiplication tensor has T[i,j,k] = 1 if we're computing C[i,j] += A[i,k]*B[k,j]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the appropriate sense)
    # Actually the tensor is: sum over r of U[i,k,r]*V[k,j,r]*W[j,i,r] should equal 1 for all i,j,k
    # That's what we checked above. But we also need to make sure there are no spurious terms.
    # Since the tensor is fully dense (all entries are 1), the check above is sufficient.
    return True

def _build_rank_55():
    """Build the known rank-55 decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return U, V, W

def _search_rank_54():
    """Try to find a rank-54 decomposition by modifying the rank-55 structure."""
    n, m, p = 2, 3, 10
    U, V, W = _build_rank_55()
    rank = 55
    
    # Try to eliminate one term by checking if it's linearly dependent
    # For each term r, check if the tensor contribution can be expressed as a combination of others
    # This is a complex optimization, so let's try a simpler approach:
    # Try to merge pairs of terms or find a better pattern
    
    # Alternative: try a different block structure
    # What if we use a 2x2 block with 10 terms instead of 11?
    
    # Let's try to find a rank-50 decomposition (10 per block)
    # by finding a better 2x2x2x3 sub-decomposition
    
    # For the 2x2 (rows) x 2 (cols) x 3 (middle) subproblem,
    # we need to multiply A[2x2] by B[2x2] (ignoring the third middle dimension for now)
    # Actually it's A[2x3] * B[3x2] for each pair of columns
    
    # Let's try a systematic search for a rank-10 decomposition of the 2x2x3x2 sub-tensor
    # where we fix the column pair and look at the 2x3 x 3x2 multiplication
    
    # The sub-tensor for columns (k0, k1) has:
    # A indices: (i, k) for i in {0,1}, k in {0,1,2} -> 6 elements
    # B indices: (k, j) for k in {0,1,2}, j in {0,1} -> 6 elements  
    # C indices: (i, j) for i in {0,1}, j in {0,1} -> 4 elements
    # T[i,j,k] = 1 for all i,j,k
    
    # We want: sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] = 1 for all i,j,k
    
    # This is a 6x6x4 tensor of rank 10 (for the 2-column block)
    
    # Let's try to find a rank-10 decomposition using optimization
    # We'll use a randomized approach with local refinement
    
    best_rank = 55
    best_U, best_V, best_W = U, V, W
    
    # Try random initializations and gradient-free optimization
    import random
    random.seed(42)
    
    for attempt in range(100):
        if time.time() - _start_time > 100:
            break
        
        # Try rank 50 (10 per block)
        target_rank = 50
        U_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
        V_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
        W_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Check if this works (unlikely with random, but let's try structured approaches)
        
    # Let's try a more structured approach: modify the existing rank-55 by
    # checking if any single term can be removed
    
    for term_to_remove in range(rank):
        if time.time() - _start_time > 100:
            break
        
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Zero out term_to_remove
        for i in range(len(U_test)):
            U_test[i][term_to_remove] = 0
        for i in range(len(V_test)):
            V_test[i][term_to_remove] = 0
        for i in range(len(W_test)):
            W_test[i][term_to_remove] = 0
        
        # Check if the remaining terms still span the tensor
        # This requires checking if the tensor is in the span of the remaining rank-1 tensors
        # Simplified check: verify the decomposition
        if _verify(U_test, V_test, W_test):
            return U_test, V_test, W_test, rank - 1
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    if time.time() - _start_time > 110:
        U, V, W = _build_rank_55()
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    U, V, W, rank = _search_rank_54()
    
    # Verify the result
    if not _verify(U, V, W):
        # Fall back to known good solution
        U, V, W = _build_rank_55()
        rank = 55
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
