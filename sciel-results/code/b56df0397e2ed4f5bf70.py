"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank by checking linear dependencies
    # The tensor flattened: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Let's try to see if we can reduce rank by finding that some combinations are redundant
    # Strategy: try rank from schoolbook-1 down to some minimum, and check if a decomposition exists
    
    # For efficiency, let's use the structure of matrix multiplication
    # Key observation: for fixed (i,k), the sum over j gives (AB)[i,k]
    # So T[i,j,k] = delta_{j, j} which is always 1 in the "multiplication" sense
    
    # Actually, let me try a different approach: use the fact that we can sometimes
    # combine terms when there are symmetries
    
    # Let's try rank 74 first
    # We can attempt to find if any linear combination of the 75 terms equals zero
    
    # Build the tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try to find minimal rank by attempting decomposition
    # Use the approach: for a given rank r, check if there exist U,V,W such that
    # sum_r U[:,r] * V[:,r] * W[:,r] = T (outer product sum)
    
    # This is a hard problem in general, but for small cases we can try
    # Let's try a greedy/iterative approach or use known results
    
    # For 3x5x5, let me try to see if rank 73 or 74 is achievable
    # by looking for linear dependencies in the "slice" structure
    
    # Alternative: use the fact that the matrix multiplication tensor for (n,m,p)
    # has rank at most n*m*p - something when there are small dimensions
    
    # Let me try a systematic search for rank reduction
    # Start with schoolbook and try to eliminate one term at a time
    
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb
    
    # Try to find if we can reduce by 1
    # For each column c, check if U[:,c], V[:,c], W[:,c] can be expressed
    # as a linear combination of the other columns in a way that preserves the tensor
    
    # This is equivalent to checking if the 75 rank-1 tensors are linearly independent
    # Flatten each rank-1 tensor into a vector of length (n*m)*(m*p)*(p*n) = 15*25*15 = 5625
    
    # That's too large. Let me use a smarter approach.
    
    # For matrix multiplication, the tensor has a special structure.
    # Let me try to use the known result or do a targeted search.
    
    # Actually, for this problem, let me just return the schoolbook if we can't
    # find something better quickly. But let's try one more thing.
    
    # Try to see if rank 74 works by attempting to find a dependency
    # among the 75 terms
    
    # Create a matrix where each row is one "equation" from the tensor
    # and each column is one term in the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For the schoolbook, U[a,r] is 1 only when r corresponds to (i,j) with a=i*m+j
    # Similarly for V and W
    
    # Let me try a different tactic: use random initialization and gradient-free search
    # for a lower rank decomposition
    
    # Given time constraints, let me try a few specific ranks
    
    for target_rank in [74, 73, 72]:
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # This is NP-hard in general, but let's try for small reductions
        
        # Approach: start with schoolbook and try to merge terms
        # For rank 74, we need to find two terms that can be combined
        
        # Let me try: can we find coefficients such that
        # sum_{r=1}^{74} u_r v_r w_r = T?
        
        # This is a system of 5625 equations with 3*15*74 + 3*25*74 + 3*15*74 = 
        # 3450 + 5550 + 3450 = 12450 unknowns (if we don't fix scaling)
        # Actually, we have gauge freedom, so it's more complex
        
        # Let me try a simpler check: is the tensor rank actually less than 75?
        # We can check the rank of certain flattenings
        
        # Flatten T as (n*m) x (m*p*p*n)
        T_flat1 = T.reshape(n * m, m * p * p * n)
        rank1 = np.linalg.matrix_rank(T_flat1)
        
        # Flatten T as (m*p) x (n*m*p*n)
        T_flat2 = T.reshape(m * p, n * m * p * n)
        rank2 = np.linalg.matrix_rank(T_flat2)
        
        # Flatten T as (p*n) x (n*m*m*p)
        T_flat3 = T.reshape(p * n, n * m * m * p)
        rank3 = np.linalg.matrix_rank(T_flat3)
        
        # The tensor rank is at least max(rank1, rank2, rank3)
        # For schoolbook, these should all be min(n*m, m*p*p*n) etc.
        
        # rank1 = min(15, 25*15) = 15
        # rank2 = min(25, 15*15) = 25  
        # rank3 = min(15, 15*25) = 15
        
        # So lower bound is 25, which is much less than 75
        # This doesn't help directly
        
        # Let me try a constructive approach for rank 74
        # If we can find that one of the 75 basis tensors is in the span of the others,
        # we can reduce the rank
        
        # Check: is the set of 75 rank-1 tensors linearly independent?
        # Represent each as a vector in R^{15*25*15}
        
        # This is a 5625 x 75 matrix, which is manageable
        M = np.zeros((n * m * m * p * p * n, rank_sb))
        for r in range(rank_sb):
            # Get the (i,j,k) for this term
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            # The rank-1 tensor has 1 at position (i*m+j, j*p+k, k*n+i)
            idx = (i * m + j) * (m * p) * (p * n) + (j * p + k) * (p * n) + (k * n + i)
            M[idx, r] = 1
        
        # Check if the 75 columns are linearly independent
        # They should be, since each has a unique non-zero position
        # So the tensor rank is exactly 75? No, that's not right.
        
        # The issue is that the tensor rank is not the same as the rank of this matrix
        # because the decomposition requires the specific factorization form
        
        # Let me just return schoolbook for now, as finding a lower rank
        # requires more sophisticated methods
        
        pass
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
