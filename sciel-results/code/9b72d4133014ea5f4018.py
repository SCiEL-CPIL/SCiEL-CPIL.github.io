"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_strassen_block(n, m, p):
    """
    Try to use Strassen-like block decomposition.
    For 2x5x6, we can try to partition into blocks.
    
    One approach: treat as multiplying (2x5) by (5x6).
    Partition A into columns: A = [A1 | A2 | A3 | A4 | A5] where each Ai is 2x1
    Partition B into rows: B = [B1; B2; B3; B4; B5] where each Bj is 1x6
    
    C = A1*B1 + A2*B2 + A3*B3 + A4*B4 + A5*B5
    
    Each Ai*Bj is a 2x6 outer product (rank-1 in some sense, but it's 2x6 matrix).
    
    Actually, let's think about this differently.
    
    For small matrices, we can try to find a decomposition by considering
    the tensor as a 3-tensor of size 2x5x6 and finding its rank.
    
    Known results:
    - Rank of matrix multiplication tensor M_{n,m,p} >= max(n*m, m*p, p*n)
    - For M_{2,5,6}: lower bound = max(10, 30, 12) = 30
    - Upper bound: 60 (schoolbook)
    
    Let's try to find a rank-30 decomposition or close to it.
    
    One known technique: if we can write the multiplication as a sum of fewer
    products by exploiting linear dependencies.
    """
    pass

def _greedy_tensor_rank(n, m, p, max_rank, time_limit=30):
    """
    Try to find a tensor decomposition of rank <= max_rank using a greedy/random approach.
    """
    start_time = time.time()
    target_rank = max_rank
    
    # The tensor T[i,j,k] = 1 for all i,j,k (this is the all-ones tensor for our case)
    # Wait, that's not right. The matrix multiplication tensor for M_{n,m,p} has:
    # T[i,j,k] = 1 if and only if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # In standard matrix multiplication, every T[i,j,k] = 1.
    
    # So we need to decompose the all-ones tensor of size n x m x p.
    # Rank of all-ones tensor of size a x b x c is... well, it's not trivially 1
    # unless we allow scaling. Actually, the all-ones tensor IS rank 1!
    # T[i,j,k] = 1 = 1*1*1 for all i,j,k.
    
    # Wait, that means the matrix multiplication tensor is NOT the all-ones tensor.
    # Let me reconsider.
    
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # (M * (A ⊗ B))[i,k] = C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k]
    # where T[i,j,k] = 1 for all valid i,j,k.
    
    # Hmm, but that would make it rank 1. That can't be right.
    
    # I think I'm confusing myself. Let me think about this more carefully.
    
    # The tensor for matrix multiplication is such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: for each (i,k), C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1.
    
    # But this is NOT a single trilinear form. It's a system of np bilinear forms.
    # The "tensor" in tensor decomposition context refers to the structure that encodes
    # all these bilinear forms simultaneously.
    
    # In the context of tensor rank for matrix multiplication:
    # We want to find rank r and matrices U (nm x r), V (mp x r), W (pn x r) such that:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is exactly what the schoolbook does with r = n*m*p.
    
    # So the question is: what is the minimum r such that we can find U, V, W
    # satisfying these constraints?
    
    # This is the rank of the matrix multiplication tensor M_{n,m,p}.
    
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,3,2}) = ? 
    # For M_{2,5,6}, we need to search.
    
    # Let's try a random/greedy approach to find a low-rank decomposition.
    
    # Strategy: Start with schoolbook (rank 60) and try to reduce rank by
    # finding linear dependencies among the terms.
    
    # Alternative: Use the fact that for fixed (i,k), we need:
    # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # 
    # Wait, that's not right either. Let me re-examine.
    #
    # The constraint is: for each triple (i,j,k),
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # There are n*m*p = 60 such constraints.
    #
    # We want to minimize rank r = number of columns in U, V, W.
    
    # Let's try to use a simple approach: initialize with random values and
    # use iterative refinement, or try specific structured decompositions.
    
    # For now, let's just return the schoolbook decomposition as a baseline.
    # The search for better decompositions is complex.
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 120  # Leave safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    # For 2x5x6, the lower bound is max(2*5, 5*6, 6*2) = max(10, 30, 12) = 30
    
    # Let's try a structured approach:
    # Partition the 5 intermediate indices into groups and use Strassen-like tricks.
    
    # Approach: Try rank 30 first (the lower bound)
    # If that fails, try 31, 32, etc.
    
    # Actually, let's try a different strategy: use the known result that
    # rank(M_{n,m,p}) can sometimes be computed via algebraic methods.
    
    # For now, let's try to use a simple optimization:
    # Start with a random initialization at a lower rank and see if we can satisfy
    # the constraints.
    
    # This is a hard combinatorial problem, so let's try a few approaches.
    
    # Approach 1: Try to find a decomposition of rank r by solving the system
    # of polynomial equations. This is NP-hard in general, but for small r
    # we can try random initializations.
    
    # Let's try ranks from 30 upward.
    for target_rank in range(30, min(rank_schoolbook + 1, 45)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this works (unlikely for random, but let's try)
            # Actually, random initialization won't work. We need a smarter approach.
            
            # Let's try a structured approach instead.
            pass
    
    # Approach 2: Use the fact that for 2x5x6, we can try to exploit the structure.
    # 
    # Let's think about it this way:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0] + A[0,4]*B[4,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1] + A[0,4]*B[4,1]
    # ... etc.
    #
    # The key insight is that we can share computations across different (i,k) pairs.
    
    # For now, let's just return the schoolbook decomposition.
    # A proper search would require more sophisticated algorithms.
    
    return best
# EVOLVE-BLOCK-END
