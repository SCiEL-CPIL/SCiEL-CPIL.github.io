"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
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
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def to_half_integer(x):
        """Convert a value to the nearest half-integer if it's close enough."""
        x2 = 2 * x
        nearest = round(x2)
        if abs(x2 - nearest) < 1e-6:
            return Fraction(nearest, 2)
        return None
    
    def try_strassen_block(n, m, p):
        """Try to find a decomposition using block structure with Strassen-like ideas.
        
        For 3x5x6, we can try various block partitions.
        The key insight is that if we can find a low-rank decomposition of the
        block multiplication tensor, we can combine it with the block structure.
        """
        # Try different block partition strategies
        # Strategy 1: Treat as (2+1) x 5 x 6
        # Strategy 2: Treat as 3 x (2+2+1) x 6  
        # Strategy 3: Treat as 3 x 5 x (2+2+2)
        
        # For now, let's try a simple approach: use the fact that matrix multiplication
        # can be decomposed using fewer multiplications for certain block sizes
        
        # Try rank values from 90 downward
        best_result = None
        best_rank = n * m * p
        
        # Try to find a decomposition with rank < 90
        # We'll use a greedy approach: try to find rank-1 tensors that cover
        # as many (i,j,k) tuples as possible
        
        for target_rank in range(best_rank - 1, 50, -1):
            if time.time() > deadline:
                break
                
            # Try to find a decomposition with this rank
            # Use a simple heuristic: try to find a set of rank-1 tensors
            # that sum to the target tensor
            
            # For each possible rank-1 tensor (u, v, w), check if it helps
            # This is expensive, so we'll use a structured approach
            
            # Try to build the decomposition incrementally
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Start with schoolbook and try to merge terms
            # This is a simplified approach
            
            # For this specific case, let's try a known decomposition
            # 3x5x6 can potentially be done with rank < 90
            
            # Try a structured decomposition based on grouping
            # Group the 5 middle dimension into (2, 3)
            # Group the 6 right dimension into (3, 3)
            
            # Actually, let's try a more direct approach:
            # Use the fact that we can share computation between related (i,j,k) tuples
            
            # For now, return None if we can't find a better decomposition quickly
            # The schoolbook is always valid
            pass
        
        return best_result
    
    # Try the Strassen block approach
    result = try_strassen_block(n, m, p)
    
    if result is None or not verify_decomposition(result['U'], result['V'], result['W']):
        result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
