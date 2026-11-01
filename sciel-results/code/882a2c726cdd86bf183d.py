"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Schoolbook gives rank 36. Let's try to find a lower rank decomposition.
    # Strategy: Try rank 24, 25, ..., 35 in order, using a structured approach.
    
    # For rank r, we need U (9 x r), V (12 x r), W (12 x r)
    # such that for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Let's try a specific construction for rank 24.
    # The idea: group by the middle index j. For each j, we have n*p = 12 terms.
    # Total: 3 * 12 = 36. To get to 24, we need to eliminate 12 terms.
    
    # One approach: use a basis transformation on the V matrix.
    # For fixed i and k, C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as a bilinear form in A[i,:] and B[:,k]
    
    # Let's try a direct construction. For rank 24, we can use the following:
    # Split the 3 j-values into groups and use shared terms.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that the tensor T[i,j,k] = 1 can be decomposed as:
    # T[i,j,k] = sum_c u_c[i,j] v_c[j,k] w_c[k,i]
    
    # For rank 24, one known construction uses the structure of the rectangular matrix.
    # Let's try to find a valid rank-24 decomposition by solving a linear system.
    
    # Alternative: try to find a rank-r decomposition by iterative refinement.
    # Start with a random initialization and use gradient descent or a similar method.
    
    # For now, let's try a few specific constructions and verify them.
    
    # Construction 1: Rank 24 using a specific pattern
    # Based on the idea of sharing terms across different j values
    
    target_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
            
        # Try to construct a rank-target_rank decomposition
        # Use a simple heuristic: start with schoolbook and try to merge terms
        
        # For rank < 36, we need to find a valid decomposition.
        # Let's try a random search with verification for small ranks.
        
        if target_rank >= 30:
            # For higher ranks, we can use a more direct approach
            # Start with schoolbook and try to find linear dependencies
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill in the first 36 terms with schoolbook, then try to zero out some
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if column < target_rank:
                            U[i * m + j][column] = 1
                            V[j * p + k][column] = 1
                            W[k * n + i][column] = 1
                            column += 1
                        # If column >= target_rank, we skip this term
                        # This won't work directly, so we need a different approach
            
            # This simple truncation won't work. We need a proper decomposition.
            # Let's try a different approach for these ranks.
            continue
        
        # For rank 24-29, try a structured construction
        # Use the fact that we can potentially share terms across j values
        
        # One known approach: use a "ladder" decomposition
        # For 3x3x4, we can try to use 8 terms per "layer" instead of 12
        
        # Let's try a specific rank-24 construction based on algebraic structure
        if target_rank == 24:
            # Try a specific pattern
            U = [[0] * 24 for _ in range(9)]
            V = [[0] * 24 for _ in range(12)]
            W = [[0] * 24 for _ in range(12)]
            
            # Fill in a candidate decomposition
            # This is a heuristic; if it doesn't verify, we move on
            
            # Pattern: use 8 terms for each of 3 groups
            # Group 1: j=0, terms 0-7
            # Group 2: j=1, terms 8-15
            # Group 3: j=2, terms 16-23
            
            for j in range(3):
                base = j * 8
                for i in range(3):
                    for k in range(4):
                        # This doesn't directly give 8 terms per j
                        pass
            
            # This simple pattern won't work. Let's try a different approach.
            # Use a basis transformation.
            
            # For now, skip this and try other ranks.
            continue
    
    # If we couldn't find a lower rank decomposition, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
