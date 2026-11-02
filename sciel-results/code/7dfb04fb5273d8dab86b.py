import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Rank is 27: 9 for first 3 cols, 9 for next 3 cols, 9 for last col
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(offset, a_indices, b_indices, c_indices):
        """
        Add a block of rank 9 (schoolbook) for a 3x3x3 or 3x3x1 subproblem.
        a_indices: list of 9 global A indices (for 3x3) or 9 global A indices (for 3x1, but only 3 are used? No, 3x1 uses 9 A elements? No, A is 3x3, so 9 elements. B is 3x1, so 3 elements. C is 3x1, so 3 elements.)
        Actually, for 3x3x3:
        A has 9 elements. B has 9 elements. C has 9 elements.
        For 3x3x1:
        A has 9 elements. B has 3 elements. C has 3 elements.
        """
        # This function assumes the subproblem is 3x3x3 or similar.
        # We will handle the mapping externally.
        pass

    # Block 1: k=0,1,2 (3x3x3)
    # A indices: 0..8
    # B indices: B[j, k] for j in 0..2, k in 0..2. Global index j*7+k.
    # C indices: C[i, k] for i in 0..2, k in 0..2. Global index k*3+i.
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A index
                a_idx = i * 3 + j
                # B index
                b_idx = j * 7 + k
                # C index
                c_idx = k * 3 + i
                
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    # Block 2: k=3,4,5 (3x3x3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A index
                a_idx = i * 3 + j
                # B index
                b_idx = j * 7 + (k + 3)
                # C index
                c_idx = (k + 3) * 3 + i
                
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    # Block 3: k=6 (3x3x1)
    for i in range(3):
        for j in range(3):
            # A index
            a_idx = i * 3 + j
            # B index
            b_idx = j * 7 + 6
            # C index
            c_idx = 6 * 3 + i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
