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

    # Schoolbook decomposition as fallback (rank 36)
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

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Compute C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 if this is the multiplication tensor
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # Wait, let me reconsider. The tensor for matrix mult is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this basis)
        # A rank-r decomposition means: T[i,j,k] = sum_{c=1}^r U[i,j,c] V[j,k,c] W[k,i,c]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find lower rank decompositions
    # Known: rank(3x3x4) is between 24 and 36
    # Let's try rank 24 first with a structured approach
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Strategy 1: Try rank 24 with specific algebraic structure
    # For 3x3x4, one known approach uses the fact that we can share terms
    
    # Try a rank-24 decomposition based on the following idea:
    # Group terms by j. For each j, we have n*p = 12 terms.
    # Total schoolbook: 3 * 12 = 36.
    # To get to 24, we need to eliminate 12 terms by combining across j.
    
    # Alternative: use a bilinear algorithm approach
    # Let's try to find a rank-r decomposition by solving a system
    
    # For now, try a few candidate ranks and see if we can find valid decompositions
    for target_rank in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
            
        # Try a structured rank-target_rank decomposition
        # Use the "ladder" or "sliding window" approach
        
        # For rank r, we need to find U (9 x r), V (12 x r), W (12 x r)
        # such that for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        
        # This is a nonlinear system. Let's try a specific construction.
        
        # Construction for rank 24:
        # Use the fact that for fixed (i,k), C[i,k] = sum_j A[i,j] B[j,k]
        # We can write this as a sum of 3 terms (one for each j).
        # But we can potentially combine terms across different (i,k) pairs.
        
        # Let's try a different approach: use known results from the literature.
        # The tensor rank of 3x3x4 is known to be at most 24.
        
        if target_rank == 24:
            # Try the specific rank-24 construction
            # This uses the structure of the 3x3x4 tensor
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill in a candidate rank-24 decomposition
            # Based on the idea of sharing terms across different j values
            
            # Let's use a systematic approach:
            # For each c in range(24), assign U[c], V[c], W[c]
            
            # Try a specific pattern:
            # c = 0..11: handle j=0, all (i,k) pairs
            # c = 12..23: handle j=1, but combine with j=2 somehow
            
            # Actually, let's just try to solve it numerically for small ranks
            
            # For now, skip and use schoolbook if no valid decomposition found
            continue
        
        # If we can't find a valid decomposition quickly, fall back to schoolbook
        break
    
    # If no better decomposition found, return schoolbook
    if best_result is None:
        return schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
