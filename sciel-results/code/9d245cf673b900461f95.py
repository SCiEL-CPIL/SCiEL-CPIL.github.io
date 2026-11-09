"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify decomposition
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to merge two terms r1, r2 into new terms
    # Given current U, V, W with rank R, try to replace terms r1, r2 with 1 term
    # This requires finding a vector that matches the "pattern" of both terms
    # on all indices where they differ
    
    # Better approach: use the fact that for matrix multiplication tensor,
    # we can use structured decompositions based on the algebraic structure
    
    # For 3x5x6, the tensor rank is known to be at most 90 (schoolbook)
    # Let's try to find a lower rank using a more systematic approach
    
    # Strategy: Use the fact that the multiplication tensor T[i,j,k] = delta_{i,j,k}
    # can be decomposed using the structure of the matrix product
    
    # Actually, let me try a different approach: use the known lower bounds
    # and try to find decompositions using algebraic methods
    
    # For now, let's do a more systematic search using the structure of the problem
    
    # Key insight: We can use the fact that for fixed j, the sub-tensor
    # T[i, j, k] is a rank-1 tensor (outer product of e_i and e_k)
    # So we need to decompose the sum over j of these rank-1 tensors
    
    # Let's try to find a decomposition using a "block" approach
    # where we group terms by their j-index
    
    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    # Try to reduce rank by merging terms
    # Two terms can be merged if there exist coefficients such that
    # the new term matches both old terms on all relevant indices
    
    # For terms r1 and r2, they have:
    # U[:, r1], U[:, r2], V[:, r1], V[:, r2], W[:, r1], W[:, r2]
    
    # A merge is possible if there's a vector w_new such that:
    # For all (i,j,k): U[i*m+j, r1]*V[j*p+k, r1]*W[k*n+i, r1] +
    #                   U[i*m+j, r2]*V[j*p+k, r2]*W[k*n+i, r2]
    # = U_new[i*m+j]*V_new[j*p+k]*w_new[k*n+i]
    
    # This is generally not possible, but we can try specific structured merges
    
    # Alternative: use the known fact that matrix multiplication tensors
    # can sometimes be decomposed using fewer terms via algebraic tricks
    
    # Let's try a random search with a smarter initialization
    # Use the schoolbook as a starting point and try small perturbations
    
    # Actually, let me try to use the structure more directly
    # The tensor for 3x5x6 matrix multiplication has a specific structure
    
    # Let's try to find a rank-1 decomposition of a related problem first
    # and build up from there
    
    # For now, return schoolbook as the best we can do quickly
    # The rank is 90
    
    # Try a few random searches to see if we can do better
    vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    # Try ranks from 85 down to 70 with random search
    for target_rank in range(min(best_rank - 1, 85), 69, -1):
        if time.time() > deadline - 30:
            break
        
        # Use a structured random search
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            
            # Generate random matrices with half-integer entries
            U_rand = np.random.choice(vals, size=(n*m, target_rank))
            V_rand = np.random.choice(vals, size=(m*p, target_rank))
            W_rand = np.random.choice(vals, size=(p*n, target_rank))
            
            # Verify
            if verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                U, V, W = U_rand, V_rand, W_rand
                break
        
        if best_rank < 90:
            break
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
