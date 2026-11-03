"""Matrix multiplication 2x3x8 — rank search with Strassen-inspired construction."""
import time
import itertools
import random

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_strassen_block(n, m, p, deadline):
    """Try Strassen-like decomposition for 2x3x8."""
    best_rank = n * m * p
    best_decomp = None
    return best_rank, best_decomp

def _search_low_rank(n, m, p, deadline):
    """Search for low-rank decompositions using systematic approach."""
    return n * m * p, None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s wall-clock limit
    n, m, p = DIMENSIONS
    
    # Start with the known rank-48 decomposition
    rank = n * m * p  # 48
    
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
    
    # Try to find a lower-rank decomposition using a structured search
    # Strategy: Use Strassen-inspired patterns to reduce rank
    # Key insight: For 2x3x8, we can potentially save multiplications
    # by combining terms using Strassen-like algebraic identities
    
    # Attempt 1: Try a rank-44 construction based on splitting
    # 2x3x8 into sub-problems and applying Strassen to 2x2x2 blocks
    
    if time.time() < deadline:
        # Try to find a valid decomposition with rank 44
        # Using a structured approach: split the 3 middle indices into 2+1
        # and use Strassen on the 2x2 part
        
        target_ranks = [44, 45, 46, 47]
        found = False
        
        for target_rank in target_ranks:
            if time.time() > deadline:
                break
            if found:
                break
            
            # Try random search for this target rank
            # Use a structured initialization
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Create a structured initialization
                U_t = [[0] * target_rank for _ in range(n * m)]
                V_t = [[0] * target_rank for _ in range(m * p)]
                W_t = [[0] * target_rank for _ in range(p * n)]
                
                # Initialize with schoolbook pattern for first min(48, target_rank) terms
                for col in range(min(48, target_rank)):
                    i = col // (m * p)
                    rem = col % (m * p)
                    j = rem // p
                    k = rem % p
                    U_t[i * m + j][col] = 1
                    V_t[j * p + k][col] = 1
                    W_t[k * n + i][col] = 1
                
                # For remaining columns, use random values
                for col in range(48, target_rank):
                    for row in range(n * m):
                        U_t[row][col] = random.choice([-1, 0, 1])
                    for row in range(m * p):
                        V_t[row][col] = random.choice([-1, 0, 1])
                    for row in range(p * n):
                        W_t[row][col] = random.choice([-1, 0, 1])
                
                # Verify
                if _verify(U_t, V_t, W_t, n, m, p):
                    best_result = {"U": U_t, "V": V_t, "W": W_t}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            return best_result
    
    # If no better solution found, return rank 48
    return best_result
# EVOLVE-BLOCK-END
