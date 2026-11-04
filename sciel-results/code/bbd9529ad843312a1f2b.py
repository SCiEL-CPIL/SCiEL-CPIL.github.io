"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2 for the 2x2 blocks, and handle 1x6 separately
    
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen Coefficients
    # A, B, C are local 2x2 matrices
    # M is the multiplication index 0..6
    
    # A_coeffs[mi] = dict of (i, j) -> coeff
    # B_coeffs[mi] = dict of (j, k) -> coeff
    # C_coeffs[mi] = dict of (i, k) -> coeff
    
    A_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},            # M3
        {(1,1): 1},            # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}  # M7
    ]
    
    B_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},            # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},            # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1}   # M7
    ]
    
    C_coeffs = [
        {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): -1},                       # M2
        {(0,1): 1, (1,1): 1},                        # M3
        {(0,0): 1, (1,0): 1},                        # M4
        {(0,0): -1, (0,1): 1},                       # M5
        {(1,1): 1},                                  # M6
        {(0,0): 1, (1,1): -1}                        # M7
    ]
    
    # Process Part 1: j in [0, 1]
    # Process Part 2: j in [2, 3]
    
    for part in range(2):
        j_offset = part * 2
        # p is split into 3 chunks of 2
        for k_chunk in range(3):
            k_offset = k_chunk * 2
            for mi in range(7):
                if time.time() > deadline:
                    break
                # Update U
                for (i, j), coeff in A_coeffs[mi].items():
                    U[i * m + (j_offset + j), col] += coeff
                
                # Update V
                for (j, k), coeff in B_coeffs[mi].items():
                    V[(j_offset + j) * p + (k_offset + k), col] += coeff
                
                # Update W
                for (i, k), coeff in C_coeffs[mi].items():
                    W[(k_offset + k) * n + i, col] += coeff
                
                col += 1
        
        if time.time() > deadline:
            break
    
    # Process Part 3: j in [4, 4] (Schoolbook for 2x1x6)
    if time.time() <= deadline:
        j_offset = 4
        for i in range(n):
            for k in range(p):
                if time.time() > deadline:
                    break
                # A[i, 4]
                U[i * m + j_offset, col] = 1
                # B[4, k]
                V[j_offset * p + k, col] = 1
                # C[k, i]
                W[k * n + i, col] = 1
                col += 1
            if time.time() > deadline:
                break
    
    # If we ran out of time, we might have an incomplete decomposition.
    # However, the logic is fast enough.
    # Convert to lists as required by the interface? 
    # The interface says: {'U', 'V', 'W'}: factor matrices... with integer/half-integer entries.
    # It doesn't explicitly say numpy arrays, but usually lists are safer if not specified, 
    # or numpy arrays are accepted. The seed uses lists. 
    # Let's return lists to be safe, matching the seed style.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
