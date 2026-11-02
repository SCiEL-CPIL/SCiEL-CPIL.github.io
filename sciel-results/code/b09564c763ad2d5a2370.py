"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_factors():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    W[3, 6] = 1 # C11
    
    return U, V, W

def _get_schoolbook_factors(n, m, p):
    """Returns U, V, W for n x m x p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid decomposition.
    # Partition J={0,1,2} into J1={0,1}, J2={2}
    # Partition K={0..6} into K1={0,1}, K2={2,3}, K3={4,5}, K4={6}
    
    # Blocks:
    # 1. J1 x K1 (2x2): Strassen
    # 2. J1 x K2 (2x2): Strassen
    # 3. J1 x K3 (2x2): Strassen
    # 4. J1 x K4 (2x1): Schoolbook
    # 5. J2 x K1 (1x2): Schoolbook
    # 6. J2 x K2 (1x2): Schoolbook
    # 7. J2 x K3 (1x2): Schoolbook
    # 8. J2 x K4 (1x1): Schoolbook
    
    strassen_U, strassen_V, strassen_W = _get_strassen_factors()
    
    # Total rank: 7*3 + 4 + 4 + 4 + 2 = 39
    total_rank = 39
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    
    # Block 1: J1={0,1}, K1={0,1}
    # U indices: i*3+j. i in 0..1, j in 0..1.
    # V indices: j*7+k. j in 0..1, k in 0..1.
    # W indices: k*2+i. k in 0..1, i in 0..1.
    
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + j, col_offset + c] = strassen_U[i * 2 + j, c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * p + k, col_offset + c] = strassen_V[j * 2 + k, c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k * n + i, col_offset + c] = strassen_W[k * 2 + i, c]
    col_offset += 7
    
    # Block 2: J1={0,1}, K2={2,3}
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + j, col_offset + c] = strassen_U[i * 2 + j, c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * p + (k + 2), col_offset + c] = strassen_V[j * 2 + k, c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[(k + 2) * n + i, col_offset + c] = strassen_W[k * 2 + i, c]
    col_offset += 7
    
    # Block 3: J1={0,1}, K3={4,5}
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + j, col_offset + c] = strassen_U[i * 2 + j, c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * p + (k + 4), col_offset + c] = strassen_V[j * 2 + k, c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[(k + 4) * n + i, col_offset + c] = strassen_W[k * 2 + i, c]
    col_offset += 7
    
    # Block 4: J1={0,1}, K4={6} (2x1 Schoolbook)
    # Rank 4
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i * m + j, col_offset] = 1
                V[j * p + 6, col_offset] = 1
                W[6 * n + i, col_offset] = 1
                col_offset += 1
    col_offset += 0 # Already incremented inside loop? No, I need to be careful.
    # Let's re-do Block 4 with explicit loop
    
    # Reset col_offset logic to be safe
    # Actually, I will just fill the remaining blocks with schoolbook
    
    # Block 4: J1={0,1}, K4={6}
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i * m + j, col_offset] = 1
                V[j * p + 6, col_offset] = 1
                W[6 * n + i, col_offset] = 1
                col_offset += 1
                
    # Block 5: J2={2}, K1={0,1} (1x2 Schoolbook)
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i * m + (j + 2), col_offset] = 1
                V[(j + 2) * p + k, col_offset] = 1
                W[k * n + i, col_offset] = 1
                col_offset += 1

    # Block 6: J2={2}, K2={2,3} (1x2 Schoolbook)
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i * m + (j + 2), col_offset] = 1
                V[(j + 2) * p + (k + 2), col_offset] = 1
                W[(k + 2) * n + i, col_offset] = 1
                col_offset += 1

    # Block 7: J2={2}, K3={4,5} (1x2 Schoolbook)
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i * m + (j + 2), col_offset] = 1
                V[(j + 2) * p + (k + 4), col_offset] = 1
                W[(k + 4) * n + i, col_offset] = 1
                col_offset += 1

    # Block 8: J2={2}, K4={6} (1x1 Schoolbook)
    for i in range(2):
        for j in range(1):
            for k in range(1):
                U[i * m + (j + 2), col_offset] = 1
                V[(j + 2) * p + 6, col_offset] = 1
                W[6 * n + i, col_offset] = 1
                col_offset += 1

    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
