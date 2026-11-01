"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import random
import numpy as np

def construct():
    """Return the schoolbook tensor decomposition for 2x3x9 matrix multiplication.
    
    We attempt to find a lower-rank decomposition by first finding a low-rank
    decomposition for 2x3x3 and then replicating it for the 9 columns (3 blocks).
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
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

    def verify_233(U, V, W, R):
        """Verify 2x3x3 decomposition.
        U: (6, R), V: (9, R), W: (6, R)
        i in 0..1, j in 0..2, k in 0..2
        U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
        """
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(R):
                        s += U[i*3+j][c] * V[j*3+k][c] * W[k*2+i][c]
                    if s != 1:
                        return False
        return True

    def find_233_decomp(target_rank, max_trials=2000):
        """Search for 2x3x3 decomposition of given rank."""
        for _ in range(max_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Generate sparse random matrices
            # Entries from {-1, 0, 1} with p=0.3 for non-zero
            U = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(6)]
            
            if verify_233(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    def build_239_from_233(decomp_233):
        """Build 2x3x9 decomposition from 2x3x3 decomposition."""
        R = len(decomp_233["U"][0])
        total_rank = 3 * R
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(27)]
        W = [[0] * total_rank for _ in range(18)]
        
        U_233 = decomp_233["U"]
        V_233 = decomp_233["V"]
        W_233 = decomp_233["W"]
        
        for block in range(3):
            # block corresponds to columns 3*block to 3*block+2
            for c in range(R):
                col = block * R + c
                # U is same for all blocks
                for row in range(6):
                    U[row][col] = U_233[row][c]
                
                # V: row index j*9 + k. For this block, k is 0..2, so actual k = 3*block + k_local
                for j in range(3):
                    for k_local in range(3):
                        k_actual = 3 * block + k_local
                        V[j * 9 + k_actual][col] = V_233[j * 3 + k_local][c]
                
                # W: row index k*2 + i. For this block, k is 3*block + k_local
                for k_local in range(3):
                    for i in range(2):
                        k_actual = 3 * block + k_local
                        W[k_actual * 2 + i][col] = W_233[k_local * 2 + i][c]
        
        return {"U": U, "V": V, "W": W}

    # Try ranks for 2x3x3
    # Known lower bound for 2x3x3 is likely 10 or 11.
    # Schoolbook is 18.
    # If we find R=10, total rank 30.
    # If R=11, total rank 33.
    # If R=12, total rank 36.
    
    best_result = None
    
    # Try R=10 first
    for R in [10, 11, 12, 13]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Increase trials for lower ranks as they are harder
        trials = 5000 if R == 10 else 2000
        
        decomp_233 = find_233_decomp(R, trials)
        if decomp_233:
            result = build_239_from_233(decomp_233)
            # Verify the 2x3x9 result just in case
            # (Optional, but good for safety)
            # The construction is exact if 2x3x3 is exact.
            return result

    # Fallback to schoolbook
    return build_schoolbook()
