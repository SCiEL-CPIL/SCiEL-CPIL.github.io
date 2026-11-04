"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p
    
    # Construct the schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find lower rank decompositions using structured search
    # For 2x3 * 3x5, known lower bounds suggest rank >= 25
    # Try ranks from 25 to 29
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that U,V,W with given rank exactly decompose the multiplication tensor."""
        # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal delta for all i,j,k
        # The tensor T[i,j,k] = 1 if we're computing element (i,k) from row i, col k
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Wait, no. The matrix multiplication tensor is defined as:
        # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid i,j,k
        # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        # But wait, the indexing is: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions with lower rank
    # Key insight: we can share some terms if the structure allows
    
    # Strategy: Try to find decompositions where some rank-1 terms are shared
    # For 2x3 * 3x5, let's try various structured approaches
    
    # Approach 1: Try rank 25-29 with random initialization and verification
    # This is expensive, so we'll use a smarter approach
    
    # Approach 2: Use the fact that for small matrices, we can enumerate possible structures
    
    # Let's try a specific structured approach for 2x3 * 3x5
    # The tensor has 2*3*5 = 30 non-zero entries, each = 1
    # A rank-r decomposition means we can write each of the 30 entries as a sum of r products
    
    # For now, let's try to see if we can find a rank < 30 decomposition
    # by trying to merge some of the schoolbook terms
    
    # If time permits, try random search for lower rank
    if time.time() < deadline - 5:
        # Try rank 29
        for target_rank in range(29, 24, -1):
            if time.time() > deadline - 10:
                break
            
            # Try a few random attempts
            max_attempts = 100 if target_rank >= 28 else 50
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization with some structure
                # Use rational numbers to maintain exactness
                U_try = [[Fraction(0)] * target_rank for _ in range(n * m)]
                V_try = [[Fraction(0)] * target_rank for _ in range(m * p)]
                W_try = [[Fraction(0)] * target_rank for _ in range(p * n)]
                
                # Initialize with schoolbook pattern, then try to merge
                # Start with first target_rank schoolbook terms
                for r in range(target_rank):
                    i = r // (m * p)
                    j = (r // p) % m
                    k = r % p
                    U_try[i*m+j][r] = Fraction(1)
                    V_try[j*p+k][r] = Fraction(1)
                    W_try[k*n+i][r] = Fraction(1)
                
                # Try to adjust to cover remaining terms
                # This is a complex optimization problem, so let's just verify if the
                # truncated schoolbook works (it won't, but let's check the framework)
                
                # Actually, let's try a different approach: use known results
                # For 2x3 * 3x5, the minimum rank is known to be 25 (or close)
                # Let's try to construct a specific rank-25 decomposition
                
                if target_rank == 25:
                    # Try a specific structured rank-25 decomposition
                    # This requires careful construction
                    pass
                
                # For now, break out if we can't find one quickly
                break
            
            if found:
                best_rank = target_rank
                break
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": best_U[:best_rank] if isinstance(best_U[0], list) else best_U, 
                "V": best_V[:best_rank] if isinstance(best_V[0], list) else best_V,
                "W": best_W[:best_rank] if isinstance(best_W[0], list) else best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
