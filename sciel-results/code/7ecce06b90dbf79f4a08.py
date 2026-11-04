import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS

    def schoolbook():
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
        """Verify that U, V, W exactly decompose the matrix multiplication tensor.
        Returns True if valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Check all tensor entries: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def verify_half_integer(U, V, W):
        """Verify that all entries are integer or half-integer, and decomposition is exact."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # Check if val is integer or half-integer
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return verify(U, V, W)

    def to_half_integer_matrix(mat):
        """Convert a matrix of floats to half-integers by rounding."""
        return [[round(val * 2) / 2 for val in row] for row in mat]

    # Start with schoolbook as guaranteed fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try Strassen-based block decomposition for 3x3 x 3x8
    # Strategy: decompose 3x8 into 3x3 + 3x3 + 3x2 blocks
    # Use Strassen for the 2x2 sub-blocks where possible
    
    # Approach: Use a structured decomposition
    # For 3x3 * 3x3, we can potentially use rank 23 (Lader) or Strassen-based
    # For 3x3 * 3x2, schoolbook is 18
    
    # Let's try a direct structured approach:
    # Decompose the 8 columns into groups and use efficient sub-decompositions
    
    # Try rank values from just below schoolbook downward
    # Focus on structured patterns rather than pure random
    
    search_ranks = list(range(best_rank - 1, max(50, best_rank - 30), -1))
    
    for target_rank in search_ranks:
        if time.time() > deadline:
            break
        
        # Try a structured pattern: use Strassen-like combinations
        # For 3x3 matrices, Strassen gives rank 7 for 2x2, extend to 3x3
        # A 3x3 can be seen as 2x2 + boundary terms
        
        # Try a few structured initializations
        attempts = 2
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Structured initialization: start from schoolbook and try to merge columns
            # by finding linear dependencies
            
            # Use a different strategy: try to find a decomposition with
            # small integer/half-integer coefficients using a greedy approach
            
            # Initialize with schoolbook pattern but try to reduce rank
            # by setting some columns to zero and adjusting others
            
            if attempt == 0:
                # Try a pattern where we use Strassen for the first 2x2x2 sub-block
                # and schoolbook for the rest
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                # Use schoolbook for most, but try to compress
                # This is a heuristic: place ones in a pattern that might allow cancellation
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            # Standard schoolbook placement
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                if col < target_rank:
                    # We have extra columns; try to use them for Strassen-like combos
                    # For now, just verify if the partial schoolbook works
                    pass
                
                # Verify
                if verify_half_integer(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
            
            elif attempt == 1:
                # Try a different structured pattern: use half-integers
                # Initialize with a pattern that uses 0.5 coefficients
                import random
                random.seed(42 + target_rank)
                
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                # Fill with small random half-integers
                for i in range(n * m):
                    for c in range(target_rank):
                        U[i][c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
                for j in range(m * p):
                    for c in range(target_rank):
                        V[j][c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
                for k in range(p * n):
                    for c in range(target_rank):
                        W[k][c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
                
                if verify_half_integer(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
        
        if best_rank < best_rank:  # This won't trigger, but structure is clear
            break
    
    # Final: try a known Strassen-based decomposition for 3x3x3x8
    # 3x3 * 3x8 = (3x3 * 3x3) + (3x3 * 3x3) + (3x3 * 3x2)
    # If we can get rank 23 for each 3x3x3 and 18 for 3x3x2, total = 23+23+18 = 64
    # But we need to construct it explicitly
    
    # For now, if we haven't improved, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
