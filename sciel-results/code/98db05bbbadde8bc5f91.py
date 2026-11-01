import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Blocks:
    # J1 = {0, 1}, J2 = {2, 3}
    # K1 = {0, 1}, K2 = {2, 3}, K3 = {4}
    
    # 4 Strassen blocks (2x2x2)
    # 2 Schoolbook blocks (2x2x1)
    
    rank = 4 * 7 + 2 * 4
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen block definition
    # Inputs: j_offset, k_offset
    # j_offset is 0 or 2
    # k_offset is 0 or 2
    
    def add_strassen_block(j_off, k_off):
        nonlocal col
        # Local indices
        # A rows: i in {0, 1}
        # A cols: j in {j_off, j_off+1}
        # B rows: j in {j_off, j_off+1}
        # B cols: k in {k_off, k_off+1}
        # C rows: i in {0, 1}
        # C cols: k in {k_off, k_off+1}
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00, C11 -> 1
        U[0 * m + (j_off + 0), col] = 1
        U[1 * m + (j_off + 1), col] = 1
        V[(j_off + 0) * p + (k_off + 0), col] = 1
        V[(j_off + 1) * p + (k_off + 1), col] = 1
        W[(k_off + 0) * n + 0, col] = 1
        W[(k_off + 1) * n + 1, col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[1 * m + (j_off + 0), col] = 1
        U[1 * m + (j_off + 1), col] = 1
        V[(j_off + 0) * p + (k_off + 0), col] = 1
        W[(k_off + 0) * n + 1, col] = 1
        col += 1
        
        # M3: A00(B10-B11)
        U[0 * m + (j_off + 0), col] = 1
        V[(j_off + 1) * p + (k_off + 0), col] = 1
        V[(j_off + 1) * p + (k_off + 1), col] = -1
        W[(k_off + 0) * n + 0, col] = 1
        col += 1
        
        # M4: A11(B01-B00)
        U[1 * m + (j_off + 1), col] = 1
        V[(j_off + 0) * p + (k_off + 1), col] = 1
        V[(j_off + 0) * p + (k_off + 0), col] = -1
        W[(k_off + 0) * n + 1, col] = 1
        W[(k_off + 1) * n + 1, col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[0 * m + (j_off + 0), col] = 1
        U[0 * m + (j_off + 1), col] = 1
        V[(j_off + 1) * p + (k_off + 1), col] = 1
        W[(k_off + 0) * n + 0, col] = 1
        W[(k_off + 1) * n + 0, col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1 * m + (j_off + 0), col] = 1
        U[0 * m + (j_off + 0), col] = -1
        V[(j_off + 0) * p + (k_off + 0), col] = 1
        V[(j_off + 0) * p + (k_off + 1), col] = 1
        W[(k_off + 0) * n + 1, col] = 1
        W[(k_off + 1) * n + 1, col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0 * m + (j_off + 1), col] = 1
        U[1 * m + (j_off + 1), col] = -1
        V[(j_off + 1) * p + (k_off + 0), col] = 1
        V[(j_off + 1) * p + (k_off + 1), col] = 1
        W[(k_off + 0) * n + 0, col] = 1
        W[(k_off + 1) * n + 0, col] = 1
        col += 1

    def add_schoolbook_2x2x1_block(j_off, k_idx):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # C_i = A_i0 B_0 + A_i1 B_1
        # 4 terms
        # T1: A00 B0 C0
        U[0 * m + (j_off + 0), col] = 1
        V[(j_off + 0) * p + k_idx, col] = 1
        W[k_idx * n + 0, col] = 1
        col += 1
        
        # T2: A01 B1 C0
        U[0 * m + (j_off + 1), col] = 1
        V[(j_off + 1) * p + k_idx, col] = 1
        W[k_idx * n + 0, col] = 1
        col += 1
        
        # T3: A10 B0 C1
        U[1 * m + (j_off + 0), col] = 1
        V[(j_off + 0) * p + k_idx, col] = 1
        W[k_idx * n + 1, col] = 1
        col += 1
        
        # T4: A11 B1 C1
        U[1 * m + (j_off + 1), col] = 1
        V[(j_off + 1) * p + k_idx, col] = 1
        W[k_idx * n + 1, col] = 1
        col += 1

    # Add blocks
    add_strassen_block(0, 0) # J1, K1
    add_strassen_block(0, 2) # J1, K2
    add_strassen_block(2, 0) # J2, K1
    add_strassen_block(2, 2) # J2, K2
    
    add_schoolbook_2x2x1_block(0, 4) # J1, K3
    add_schoolbook_2x2x1_block(2, 4) # J2, K3

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
