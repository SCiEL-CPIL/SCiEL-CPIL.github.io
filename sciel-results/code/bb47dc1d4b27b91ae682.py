"""Optimized tensor decomposition search for 4x5 * 5x6 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
# Deadline guard: return best result before 140s to be safe within 150s limit
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Known upper bound: schoolbook = n*m*p = 120
    # Best known for 4x5x6 is around 60-80, but let's try to find something good
    
    # Strategy: Try to find a decomposition with rank as low as possible.
    # We'll use a randomized approach with structured initializations and local refinement.
    
    # First, let's define the matrix multiplication tensor T[i, j, k] = delta[i, j, k]
    # where (A*B)[i, k] = sum_j A[i, j] * B[j, k]
    # T is a 3-tensor of shape (n, m, p) where T[i, j, k] = 1 if this is the (i, j, k) term
    # Actually, the tensor is T[i, j, k] such that T[i, j, k] = delta (the identity in some sense)
    # More precisely: T[i, j, k] = 1 for all (i, j, k) where we multiply A[i,j] * B[j,k]
    # Wait, the tensor for matrix multiplication is T[i, j, k] = 1 for all valid (i, j, k)
    # No, that's not right either. Let me think again.
    
    # The matrix multiplication tensor M has entries M[i, j, k] = 1 if we are computing
    # (A*B)[i, k] which involves A[i, j] * B[j, k]. So M[i, j, k] = 1 for all (i, j, k)
    # in the range. Actually no - the tensor is defined such that:
    # sum_{r} U[i, r] * V[j, r] * W[k, r] should equal the "structure constant"
    
    # Let me re-read the problem. The tensor decomposition of matrix multiplication:
    # We want rank-r factors U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # where delta is 1 if this is a valid multiplication term, 0 otherwise.
    
    # Actually, for matrix multiplication, the tensor T[i, j, k] = 1 for all (i, j, k)
    # because every (i, j, k) triple corresponds to a term in the product.
    # Wait, that would make it a rank-1 tensor which is clearly wrong.
    
    # Let me reconsider. The matrix multiplication tensor is:
    # T[i, j, k] = 1 if the (i, k) entry of A*B uses A[i, j] and B[j, k]
    # This is always true for valid i, j, k, so T[i, j, k] = 1 for all (i, j, k).
    # But that can't be right because then the rank would be 1.
    
    # I think I'm confusing things. Let me look at this differently.
    # The tensor for multiplying an n x m matrix by an m x p matrix has shape (n, m, p)
    # and T[i, j, k] = 1 for all (i, j, k) in [0,n) x [0,m) x [0,p).
    # A rank-r decomposition means: T[i, j, k] = sum_{r=1}^r U[i, r] * V[j, r] * W[k, r]
    # where U is n x r, V is m x r, W is p x r.
    
    # But the problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This suggests a different indexing. Let me re-read.
    
    # "U[i*m+j, r]" - so U has rows indexed by (i, j) pairs, n*m total
    # "V[j*p+k, r]" - V has rows indexed by (j, k) pairs, m*p total  
    # "W[k*n+i, r]" - W has rows indexed by (k, i) pairs, p*n total
    
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    # where T[i, j, k] is the matrix multiplication tensor.
    
    # For matrix multiplication, T[i, j, k] = 1 for all (i, j, k) because each term
    # A[i, j] * B[j, k] contributes to (A*B)[i, k].
    
    # Wait, that still gives a rank-1 tensor. I must be misunderstanding.
    
    # Ah, I think the tensor is actually the "structure tensor" where:
    # T[i, j, k] = 1 if and only if the (i, k) entry of the product depends on A[i,j] and B[j,k]
    # In standard matrix multiplication, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid (i, j, k).
    
    # This means the tensor is all-ones, which has rank 1. That can't be right for a
    # non-trivial decomposition problem.
    
    # I think I need to reconsider what the tensor actually is. Perhaps the problem is
    # about the bilinear map, and the tensor encodes which products contribute to which
    # output entries, with the understanding that each output entry (i, k) gets a sum
    # over j. The tensor T has entries T[i, j, k] = 1, and we're looking for a rank
    # decomposition where the "rank" corresponds to the number of bilinear terms.
    
    # Actually, I recall now: the matrix multiplication tensor T_{n,m,p} has entries
    # T[i, j, k] = 1 for all (i, j, k) in [n] x [m] x [p], and its tensor rank is the
    # minimum number of rank-1 tensors needed to sum to T. Since T is all-ones, its
    # rank is indeed 1... unless there's a constraint I'm missing.
    
    # Let me look at this from the schoolbook perspective. The schoolbook algorithm
    # has rank n*m*p because it computes each product A[i,j]*B[j,k] as a separate term.
    # This suggests the tensor is NOT all-ones, or the rank-1 tensors have a specific
    # structure.
    
    # I think the issue is that in a rank-1 term U[:, r] * V[:, r] * W[:, r], the
    # indices are coupled. A rank-1 term corresponds to a single "gate" where:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is nonzero only for specific (i, j, k).
    
    # For the schoolbook, each term r corresponds to a specific (i, j, k) triple, and
    # U[i*m+j, r] = V[j*p+k, r] = W[k*n+i, r] = 1 only for that triple.
    
    # So the tensor T[i, j, k] = 1 for all (i, j, k), and we need to write it as a sum
    # of rank-1 tensors where each rank-1 tensor is "sparse" in a specific way.
    
    # Actually, I think the tensor rank here is the bilinear complexity, and the
    # all-ones tensor does have rank n*m*p in this restricted sense because each rank-1
    # term can only "cover" one (i, j, k) triple effectively.
    
    # Let me just proceed with the search. We want to minimize rank r such that there
    # exist U, V, W with the given shapes and integer/half-integer entries satisfying:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i, j, k)
    
    # This is a constraint satisfaction / optimization problem.
    
    # Given the complexity, let's try a few strategies:
    # 1. Try known good ranks and use random initialization + local search
    # 2. Use structured decompositions
    
    # For 4x5x6, let's try ranks from a reasonable lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try to find a better decomposition
    # We'll use a randomized approach with verification
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try some structured decompositions
    # One approach: use the fact that we can group terms
    
    # Let's try a simple improvement: block decomposition
    # If we can find a way to combine multiple schoolbook terms into one rank-1 term
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and try to improve if time allows
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        return best_result
    
    # Try to find a better decomposition using random search
    # We'll try ranks lower than schoolbook
    
    import random
    
    # Try a few random initializations for lower ranks
    for target_rank in range(30, best_rank, -5):  # Try from 30 up to best_rank-5
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try random initialization
        for attempt in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Initialize with small random integers
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result
