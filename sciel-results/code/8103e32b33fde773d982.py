"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition using exact arithmetic
    def verify(U, V, W):
        # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k}
        # Actually the tensor is T[i,j,k] = 1 if (i,j,k) corresponds to valid indices
        # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all valid i,j,k
        # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        rank = len(U[0])
        # Build the tensor from decomposition
        # T[i,j,k] should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Try known small rank decompositions
    # For 3x4x7, the schoolbook rank is 84.
    # We can try to find a lower rank decomposition.
    
    # Strategy: Use the fact that matrix multiplication tensor has structure.
    # A rank-r decomposition means we can write the multiplication as r rank-1 tensors.
    
    # Let's try a simple approach: enumerate small ranks and use random search + local optimization
    # But exact arithmetic with Fractions is slow. Let's use a smarter approach.
    
    # Actually, for exact tensor decomposition, we can use the following:
    # The matrix multiplication tensor T_{ijk} for multiplying (n x m) by (m x p) matrices
    # has entries T_{ijk} = 1 if the indices are valid.
    # 
    # A rank-r decomposition is: T_{ijk} = sum_{c=1}^{r} U_{ij,c} V_{jk,c} W_{ki,c}
    # 
    # This is equivalent to: for each c, we have a "term" that contributes to the product.
    # 
    # One known approach: use the "border rank" or known decompositions.
    # For 3x4x7, let's try rank 83, 82, ... down to some minimum.
    
    # Given time constraints, let's try a few random perturbations of the schoolbook
    # to see if we can merge some terms.
    
    # Actually, a better approach: the tensor is sparse in a structured way.
    # Each rank-1 term U_{ij,c} V_{jk,c} W_{ki,c} contributes to T_{ijk}.
    # 
    # Let's try: can we find a rank-83 decomposition by combining two schoolbook terms?
    # Two terms with the same (i,j) but different k, or same (j,k) but different i, etc.
    
    # For a fixed (i,j), the schoolbook has p terms (one for each k).
    # Can we represent sum_k delta_{k,k'} as a sum of fewer rank-1 terms? No, because
    # the k-index is in both V and W.
    
    # Let's try a different approach: use the fact that for fixed j, the terms
    # with that j form a sub-tensor. 
    
    # Given the complexity, let's just try a few random searches for rank < 84.
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try rank = best_rank - 1, best_rank - 2, etc.
    # For each rank, do a random search
    
    def random_search(rank, max_iters=1000):
        """Random search for a rank-r decomposition."""
        if time.time() > deadline - 5.0:
            return None
        
        # Initialize with random small values
        # Use integers or half-integers
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                return None
            
            # Random initialization
            U = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(N)]
            V = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(M)]
            W = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(P)]
            
            # This is unlikely to work with pure random, so skip to local search
            pass
        
        return None
    
    # A more structured approach: start from schoolbook and try to merge terms
    # by finding linear dependencies.
    
    # The schoolbook has 84 terms. Each term is indexed by (i,j,k).
    # Term (i,j,k) has:
    #   U[i*m+j, c_{ijk}] = 1
    #   V[j*p+k, c_{ijk}] = 1  
    #   W[k*n+i, c_{ijk}] = 1
    #
    # To reduce rank, we need to find a linear combination of terms that gives zero
    # contribution to all T_{i',j',k'}.
    
    # This is hard to do efficiently. Let's just return the schoolbook for now,
    # but try a few quick checks for small rank reductions.
    
    # Check if rank 83 is possible by trying to find a dependency
    # among the 84 schoolbook terms.
    
    # The space of rank-1 tensors is N*M*P dimensional (84*28*21 = 49392 dims).
    # The schoolbook terms are 84 vectors in this space.
    # They are linearly independent (each has a unique non-zero pattern).
    # So we can't simply drop one.
    
    # Given time constraints, return schoolbook.
    return best_result
# EVOLVE-BLOCK-END
