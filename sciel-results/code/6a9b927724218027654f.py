"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = r_sb
    
    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank solutions using randomized search
    # For 2x4x6, try ranks from 47 down to 30
    for target_rank in range(47, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations with small integer values
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            vals = [-1, 0, 1]
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
        else:
            break
    
    return best_result
# EVOLVE-BLOCK-END
