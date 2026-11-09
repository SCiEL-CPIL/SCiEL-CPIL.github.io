import time
import sys
import random
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a low-rank exact tensor decomposition for matrix multiplication.
    Falls back to schoolbook if no better solution is found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition."""
        rank = schoolbook_rank
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
    
    def verify_decomposition(U, V, W):
        """
        Verify that the decomposition exactly decomposes the matrix multiplication tensor.
        For all i, j, k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        """
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    def try_structured_rank(rank):
        """
        Try to find a rank-r decomposition using structured patterns based on
        decomposing the middle dimension into smaller blocks.
        """
        if rank >= schoolbook_rank:
            return schoolbook_decomposition()
        
        # Strategy: Use the fact that we can decompose 5 = 2 + 3
        # For 2x5 * 5x6, we can think of it as:
        # (2x5) * (5x6) where the 5 in the middle is split
        # Use block decomposition with known small matrix multiplication ranks
        
        # Known: 2x2 * 2x2 has rank 7 (Strassen)
        # Known: 2x3 * 3x2 has rank 7
        # Known: 3x3 * 3x3 has rank 23
        
        # For 2x5 * 5x6, we can use:
        # Split 5 into 2+3, 6 into 2+4
        # Then we need: (2x2)*(2x2), (2x3)*(3x2), (2x4)*(4x2), (3x2)*(2x4), (3x3)*(3x3), (3x4)*(4x3)
        # But this gets complex. Let's try a different approach.
        
        # Use randomized search with structured initialization
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with structured patterns
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Use a mixture of random and structured patterns
            for r in range(rank):
                if r < rank // 2:
                    # Structured: assign based on modular patterns
                    for i in range(n):
                        for j in range(m):
                            if (i + j) % 3 == r % 3:
                                U[i * m + j][r] = 1 if (i * j) % 2 == 0 else -1
                    for j in range(m):
                        for k in range(p):
                            if (j + k) % 3 == r % 3:
                                V[j * p + k][r] = 1 if (j * k) % 2 == 0 else -1
                    for k in range(p):
                        for i in range(n):
                            if (k + i) % 3 == r % 3:
                                W[k * n + i][r] = 1 if (k * i) % 2 == 0 else -1
                else:
                    # Random
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k][r] = random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = random.choice([-1, 0, 1])
            
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_pure_random_rank(rank):
        """
        Try to find a rank-r decomposition using pure randomized search.
        """
        if rank >= schoolbook_rank:
            return schoolbook_decomposition()
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try progressively lower ranks
    best_result = schoolbook_decomposition()
    
    # Theoretical lower bound
    min_possible = max(n * p, m * n, m * p)
    
    # Try structured search first, then random
    for target_rank in range(min_possible, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try structured approach
        result = try_structured_rank(target_rank)
        if result is not None:
            best_result = result
            break
        
        # Try random approach
        result = try_pure_random_rank(target_rank)
        if result is not None:
            best_result = result
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
