"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(n):
            j1 = j // 2
            j2 = j % 2
            
            # L1 symbol components
            s1_c1 = (i1 + j1) % 5
            s1_c2 = (i2 + j2) % 2
            L1[i][j] = 5 * s1_c1 + s1_c2
            
            # L2 symbol components
            s2_c1 = (i1 + 2 * j1) % 5
            s2_c2 = j2
            L2[i][j] = 5 * s2_c1 + s2_c2
            
    return [L1, L2]
# EVOLVE-BLOCK-END
