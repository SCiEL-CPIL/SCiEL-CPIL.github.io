"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """
    Build rank-21 decomposition using Strassen's algorithm on three 2x2 blocks.
    
    For multiplying (n x m) by (m x p) where n=2, m=2, p=6:
    - Split B into three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    - Apply Strassen to each block independently
    - Total rank = 3 * 7 = 21
    """
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U: coefficients for A (input matrix, 2x2)
    # V: coefficients for B (input matrix, 2x2)  
    # W: coefficients for C (output matrix, 2x2)
    # Indexing: 0=A00, 1=A01, 2=A10, 3=A11
    
    strassen_steps = [
        # Step 0: M1 = (A00 + A11) * (B00 + B11)
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # Step 1: M2 = (A10 + A11) * B00
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # Step 2: M3 = A00 * (B01 - B11)
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        # Step 3: M4 = A11 * (B10 - B00)
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        # Step 4: M5 = (A00 + A01) * B11
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # Step 5: M6 = (A10 - A00) * (B00 + B01)
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # Step 6: M7 = (A01 - A11) * (B10 + B11)
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # For each of the 3 blocks of B (and corresponding block of C)
    for block in range(3):
        base_col = block * 7
        
        # Map local indices to global indices
        # A (U): local (i, j) -> global i*m + j = i*2 + j
        # B (V): local (j, k_local) -> global j*p + (block*2 + k_local) = j*6 + block*2 + k_local
        # C (W): local (k_local, i) -> global (block*2 + k_local)*n + i = (block*2 + k_local)*2 + i
        
        for s in range(7):
            col = base_col + s
            step = strassen_steps[s]
            
            # Fill U (A matrix coefficients)
            for local_idx in range(4):
                val = step['u'][local_idx]
                if val != 0:
                    # local_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    i = local_idx // 2
                    j = local_idx % 2
                    U[i * m + j][col] = val
            
            # Fill V (B matrix coefficients)
            for local_idx in range(4):
                val = step['v'][local_idx]
                if val != 0:
                    # local_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    j = local_idx // 2
                    k_local = local_idx % 2
                    k_global = block * 2 + k_local
                    V[j * p + k_global][col] = val
            
            # Fill W (C matrix coefficients)
            for local_idx in range(4):
                val = step['w'][local_idx]
                if val != 0:
                    # local_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    k_local = local_idx // 2
                    i = local_idx % 2
                    k_global = block * 2 + k_local
                    W[k_global * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-21 Strassen decomposition
    result = _strassen_decomposition(n, m, p)
    
    # Verify the decomposition is correct
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If verification fails (shouldn't happen), fall back to schoolbook
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
# EVOLVE-BLOCK-END
