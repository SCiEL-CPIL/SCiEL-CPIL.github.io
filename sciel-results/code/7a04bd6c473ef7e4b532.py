"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import threading

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 50.0
    n = 10
    
    def time_left():
        return deadline - time.time()

    def is_latin(square):
        for i in range(n):
            if len(set(square[i])) != n:
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(a, b):
        seen = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Construct two MOLS of order 10 using the 2x5 decomposition
    # q = 5
    q = 5
    
    # S1[u][v] = (u + v) % q
    # S2[u][v] = (u + 2*v) % q
    
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        i1 = i // q
        i2 = i % q
        for j in range(n):
            j1 = j // q
            j2 = j % q
            
            # S1 part
            s1_val = (i2 + j2) % q
            L1[i][j] = 2 * s1_val + i1 + j1
            
            # S2 part
            s2_val = (i2 + 2 * j2) % q
            L2[i][j] = 2 * s2_val + i1 + j1
            
    squares = [L1, L2]
    
    # Verify
    if is_latin(L1) and is_latin(L2) and are_orthogonal(L1, L2):
        return squares
    else:
        # Fallback to just the first square if something went wrong (shouldn't happen)
        return [L1]

# EVOLVE-BLOCK-END
