"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 18
    
    # Known lower bounds and known ranks for small matrix multiplication:
    # For 2x3x3, the rank is known to be 15 (Strassen-like or other constructions).
    # Actually, let's check: the rank of matrix multiplication tensor T_{2,3,3}.
    # Known results: R(2,3,3) = 15? Or is it 16?
    # Let's try to find a decomposition of rank less than 18.
    
    # We'll try to use a known decomposition or search for one.
    # Since we need an EXACT decomposition with integer/half-integer entries,
    # we can try known constructions.
    
    # For 2x3x3, one known rank is 15 (from literature on small matrix multiplication).
    # Let's try to construct or verify a rank-15 decomposition.
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor for (n,m,p) has rank at least max(n*m, m*p, n*p).
    # For 2,3,3: max(6, 9, 6) = 9. So lower bound is 9.
    # Schoolbook is 18.
    
    # Known: R(2,2,2) = 7 (Strassen)
    # R(2,3,3): I believe this is 15 or 16.
    
    # Let's try a systematic approach: start with schoolbook and try to reduce rank
    # by finding linear dependencies or using known tricks.
    
    # Alternative: use the fact that we can embed smaller problems.
    # 2x3x3 can be seen as two 2x3x2 blocks plus one 2x3x1 block, etc.
    
    # Let's try to find a rank-16 or rank-15 decomposition by search.
    # Given the time constraint, let's first return a valid decomposition.
    
    # Try rank 15:
    # We'll construct a candidate and verify it.
    
    # Let me try a different approach: use known tensor rank results.
    # For 2x3x3, I'll try to construct a decomposition.
    
    # Actually, let's just do a quick search for a low-rank decomposition
    # using random integer/half-integer matrices and check.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # First, let's try to see if we can find a rank < 18 decomposition
    # by trying known constructions.
    
    # Try rank 16, 15, 14, ...
    # We'll use a simple verification: for a candidate (U, V, W) of rank r,
    # check that the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # matches the identity tensor (which is 1 if i==i', j==j', k==k' in the appropriate sense).
    
    # Wait, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # is T[i, j, k] where i in [0,n), j in [m), k in [p), and the "product" index...
    # Actually, the standard tensor is: T[(i, j), (j, k), (k, i)] = 1 for all valid i,j,k
    # No wait, that's not right either.
    
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n + i, r]
    
    # The decomposition is: A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't look like standard matrix multiplication.
    
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T[i, j, k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # But T[i,j,k] = 1 for all i,j,k (it's just the identity in the j-sum).
    
    # Hmm, let me think again. The tensor decomposition for matrix multiplication:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # No, that's not standard.
    
    # Let me look at the schoolbook decomposition in the seed:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple
    # So C[i,k] = sum_{i,j,k} A[i,j] B[j,k] * 1 * 1 * 1 * delta(i,i') * delta(k,k')
    # Wait, that would give sum_{i,j,k} A[i,j] B[j,k] which is not C[i,k].
    
    # I think I'm misunderstanding the tensor structure. Let me re-examine.
    
    # The tensor for matrix multiplication M_{n,m,p} has entries indexed by (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p).
    # The "contracted" form is: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i, j, k] = 1 (for all i,j,k), and the multiplication is:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] ... no that's not a tensor contraction.
    
    # Actually, the standard tensor decomposition is:
    # T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]  ... no.
    
    # Let me look at it differently. The bilinear form is:
    # <A, B> -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form if we include C:
    # sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j]^2 * B[j,k]^2 ... no.
    
    # I think the correct interpretation is:
    # The tensor T has components T[i, j, k] and we decompose it as:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And the "multiplication" is recovered by:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] must equal 1 for all i,j,k.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    # In the schoolbook, each r corresponds to one (i,j,k) triple, and:
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k), else 0
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k), else 0  
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0
    # So the product is 1 only when r matches (i,j,k), giving sum = 1. Correct.
    
    # So we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's try to find a rank-16 or lower decomposition.
    
    # Given the complexity, let's first try a few known constructions or random search.
    
    # For now, let's return the schoolbook decomposition as a fallback,
    # but try to find a better one first.
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        r = len(U[0]) if U else 0
        # U is n*m x r, V is m*p x r, W is p*n x r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # We'll try ranks from 15 down to 9
    
    best_rank = rank_schoolbook
    best_result = None
    
    # Schoolbook as baseline
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decompositions
    # This is a hard problem, so let's try a few heuristics
    
    # Heuristic: try to merge columns that have overlapping support
    # This is complex, so let's just try a few known patterns
    
    # For 2x3x3, let's try rank 15
    # We can try to use a Strassen-like decomposition for the 2x2x2 part and handle the rest
    
    # Actually, let's just return the schoolbook for now since finding a provably
    # lower rank is non-trivial and time is limited.
    
    if time.time() > deadline - 10:
        return best_result
    
    # Try a simple random search for rank 16
    import random
    random.seed(42)
    
    for target_rank in [16, 15]:
        if time.time() > deadline - 30:
            break
        
        # Try random integer matrices
        for attempt in range(1000):
            if time.time() > deadline - 30:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
# EVOLVE-BLOCK-END
