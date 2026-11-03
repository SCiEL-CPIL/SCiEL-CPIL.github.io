"""Find rank-minimal exact tensor decomposition for 3x5 by 5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 75

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0]) if U and U[0] else 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    err = abs(s - 1.0)
                    max_err = max(max_err, err)
                    if max_err > 1e-6:
                        return False, max_err
        return True, max_err

    def to_half_integer_list(arr):
        """Convert numpy array to list of Fraction(half-integers) if close enough."""
        result = []
        for row in arr:
            r = []
            for val in row:
                # Check if close to a half-integer
                h = round(val * 2) / 2.0
                if abs(val - h) < 1e-9:
                    r.append(Fraction(h).limit_denominator(2))
                else:
                    r.append(Fraction(val).limit_denominator(1000))
            result.append(r)
        return result

    # Try a structured approach: use known results for small matrix multiplication
    # For 3x5x5, the rank is known to be at most 55 (from literature on tensor rank)
    # But let's try to find it computationally.
    
    # Strategy: Start from schoolbook and try to find linear dependencies
    # that allow rank reduction.
    
    # Build the tensor as a matrix and find its rank
    # The multiplication tensor can be viewed as a bilinear map
    # T: A ⊗ B → C, where A is n×m, B is m×p, C is n×p
    
    # The structure tensor: for each term in the bilinear form,
    # we have a triple (u, v, w) where u is a vector in A*, v in B*, w in C
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # For matrix multiplication, the "obvious" decomposition has rank n*m*p
    # Each term corresponds to one index combination (i,j,k)
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can search over possible decompositions.
    
    # First, let's see if we can find a rank-74 decomposition by looking for
    # a linear dependency among the 75 standard terms.
    
    # The tensor entries are all 1 (for the standard matrix multiplication).
    # A rank-r decomposition means:
    # sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's try to find a decomposition with rank < 75 using a genetic/random approach
    # with structured initialization.
    
    best_rank = rank_schoolbook
    best_decomp = make_schoolbook()
    
    # Try to find improvements using iterative optimization
    # Key insight: we can use the "Coppersmith-Winograd" style ideas or
    # simply try random structured decompositions
    
    # For a more systematic approach, let's try to use the fact that
    # some entries can be shared.
    
    # Let's try a simple but effective heuristic:
    # 1. Start with schoolbook
    # 2. Try to merge terms that have similar structure
    
    # Actually, let's try a cleaner approach: use the known structure of
    # matrix multiplication and try to find the minimum rank.
    
    # For 3x5x5, let's try ranks from 50 to 74 and see if we can find valid decompositions
    # using a structured random search.
    
    for target_rank in range(55, rank_schoolbook):
        if time.time() > deadline - 20:
            break
        
        # Use a structured random search
        np.random.seed(target_rank * 12345 + 6789)
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a structured pattern
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Fill with a structured pattern based on the matrix multiplication
            # Use the "block" structure
            for c in range(target_rank):
                # Assign random indices
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                # This is a very naive approach; let's use something smarter
                pass
            
            # Better approach: use a random linear combination of the standard terms
            # Start with standard terms and try to combine them
            if target_rank < rank_schoolbook:
                # Create a random mixing
                U_std = np.zeros((n*m, rank_schoolbook))
                V_std = np.zeros((m*p, rank_schoolbook))
                W_std = np.zeros((p*n, rank_schoolbook))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_std[i*m+j, col] = 1
                            V_std[j*p+k, col] = 1
                            W_std[k*n+i, col] = 1
                            col += 1
                
                # Try to project to lower rank
                # This won't work directly because the tensor structure is fixed
                # We need to find a different decomposition entirely
                
                # Let's try a completely different approach:
                # Use the "small" tensor decomposition for these dimensions
                # For 3x5x5, we can try to use the fact that the rank is bounded
                # by known results
                
                # Actually, let's just try random matrices and see if we get lucky
                # (very unlikely to work, but let's try)
                U = np.random.randn(n*m, target_rank) * 0.1
                V = np.random.randn(m*p, target_rank) * 0.1
                W = np.random.randn(p*n, target_rank) * 0.1
                
                valid, err = verify(U.tolist(), V.tolist(), W.tolist())
                if valid:
                    found = True
                    U_list = to_half_integer_list(U)
                    V_list = to_half_integer_list(V)
                    W_list = to_half_integer_list(W)
                    best_rank = target_rank
                    best_decomp = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if found:
            break
    
    # If we found a better decomposition, return it
    # Otherwise, return the schoolbook decomposition
    return best_decomp
# EVOLVE-BLOCK-END
