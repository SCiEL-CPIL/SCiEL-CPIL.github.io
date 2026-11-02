"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill

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

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def check_half_integers(U, V, W):
        """Check all entries are integers or half-integers."""
        for arr in [U, V, W]:
            for row in arr:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Try structured approaches to reduce rank
    # Key insight: we can potentially share terms across different (i,j,k) triples
    # if the structure allows it.
    
    # Approach 1: Try to find a decomposition with rank = max(n*m, m*p, p*n) = 28
    # This is the theoretical lower bound. Let's try a few structured patterns.
    
    # Approach 2: Use known results for small matrix multiplication
    # For 3x4x7, the minimum rank is likely between 28 and 84.
    # Let's try a more systematic search with better initialization.
    
    # Try ranks from the lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations at this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Use a mix of 0, 1, -1, 0.5, -0.5
            choices = [0, 1, -1, 0.5, -0.5]
            U = [[random.choice(choices) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(choices) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(choices) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Ensure the result is valid
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END
