"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Efficient verification using einsum
    def verify(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    # Strategy: Try to reduce rank by finding linear dependencies
    # or by using structured algebraic decompositions
    
    # Approach 1: Try ranks from 119 down, using structured initialization
    # For each rank, try to build a valid decomposition
    
    # Key insight: We can try to "merge" pairs of schoolbook terms
    # If U1*V1*W1 + U2*V2*W2 can be replaced by a single term or
    # if some terms are linearly dependent, we can reduce rank.
    
    # Approach 2: Use the fact that for small matrices, we can try
    # all possible "patterns" of non-zero entries
    
    # Let's try a smarter approach: use the structure of the problem.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-1 term is U[i,j]*V[j,k]*W[k,i].
    # We need sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k.
    
    # This is equivalent to finding a low-rank factorization of the
    # "flattened" tensor.
    
    # For now, let's try random searches with the fast verification
    # and structured initializations.
    
    # Try structured perturbations of the schoolbook decomposition
    # Idea: set some entries to 0 and see if we can compensate
    
    # Actually, let's try a different approach: use the known fact that
    # M_{4,5,6} might have rank < 120. Try to find it via:
    # 1. Start with rank 120
    # 2. Try to zero out one column at a time and see if the remaining
    #    119 columns still span the tensor (i.e., if the 120th column
    #    is in the span of the others)
    
    # This is equivalent to checking if the 120 rank-1 tensors are
    # linearly independent.
    
    # Build the tensor as a vector and check linear independence
    # The tensor has n*m*p = 120 entries, all equal to 1.
    # Each rank-1 term contributes a vector with a 1 at position (i,j,k)
    # and 0 elsewhere.
    
    # So the 120 rank-1 vectors are the standard basis vectors e_1, ..., e_120.
    # They are clearly linearly independent, so the rank is exactly 120
    # for this particular decomposition.
    
    # But wait - that's the rank of the *set* of rank-1 tensors, not the
    # tensor rank. The tensor rank is the minimum number of rank-1 tensors
    # needed to sum to T.
    
    # The tensor T has all entries equal to 1. Can we write it as a sum
    # of fewer than 120 rank-1 tensors?
    
    # A rank-1 tensor U⊗V⊗W has entries U[i,j]*V[j,k]*W[k,i].
    # For this to be non-zero, we need U[i,j]≠0, V[j,k]≠0, W[k,i]≠0.
    
    # The question is: what is the minimum number of such rank-1 tensors
    # that sum to the all-ones tensor?
    
    # This is the tensor rank of the all-ones tensor in the space of
    # 3-tensors with the specific index structure of matrix multiplication.
    
    # For the "full" tensor (no structure), the rank of the all-ones
    # tensor of size n×m×p is max(n,m,p) if we allow arbitrary rank-1
    # tensors. But here, the rank-1 tensors have the specific form
    # U[i,j]*V[j,k]*W[k,i], which is more restrictive.
    
    # Let's try a computational approach: use the fast verification
    # to search for lower-rank decompositions.
    
    # Try ranks from 119 down to 100
    for target_rank in range(119, 99, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            
            # Use small integer/half-integer values
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            U = np.random.choice(vals, size=(n * m, target_rank))
            V = np.random.choice(vals, size=(m * p, target_rank))
            W = np.random.choice(vals, size=(p * n, target_rank))
            
            if verify(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
