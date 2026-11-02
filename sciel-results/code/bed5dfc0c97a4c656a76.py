"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
- Decompose the 15 columns of B into 7 blocks of 2 columns and 1 block of 1 column.
- Each 2-column block corresponds to a 2x2x2 matrix multiplication, which has rank 7 (Strassen).
- The 1-column block corresponds to a 2x2x1 matrix multiplication, which has rank 4 (schoolbook).
- Total rank = 7 * 7 + 4 = 53.
- This decomposition is exact and uses integer entries.
"""

import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Total rank calculation
    num_strassen_blocks = p // 2  # 7 blocks
    num_schoolbook_blocks = p % 2 # 1 block
    rank = num_strassen_blocks * 7 + num_schoolbook_blocks * 4
    # rank = 7 * 7 + 1 * 4 = 53
    
    # Allocate factor matrices
    # U: (n*m, rank) = (4, 53)
    # V: (m*p, rank) = (30, 53)
    # W: (p*n, rank) = (30, 53)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V rows (local): 0:B00, 1:B01, 2:B10, 3:B11
    # W rows (local): 0:C00, 1:C01, 2:C10, 3:C11
    
    # Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # W reconstruction:
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    strassen_W = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Schoolbook coefficients for 2x2x1
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V rows (local): 0:B00, 1:B10 (since p=1, only one column)
    # W rows (local): 0:C00, 1:C10
    
    # Terms:
    # Q1 = A00 * B00
    # Q2 = A01 * B10
    # Q3 = A10 * B00
    # Q4 = A11 * B10
    
    # W reconstruction:
    # C00 = Q1 + Q2
    # C10 = Q3 + Q4
    
    schoolbook_U = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    schoolbook_V = [
        [1, 0, 1, 0], # B00
        [0, 1, 0, 1]  # B10
    ]
    
    schoolbook_W = [
        [1, 1, 0, 0], # C00
        [0, 0, 1, 1]  # C10
    ]
    
    # Fill the global matrices
    col_offset = 0
    for b in range(num_strassen_blocks):
        # Columns k0 = 2*b, k1 = 2*b + 1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for r in range(7):
            global_r = col_offset + r
            
            # U
            for i in range(n * m):
                U[i][global_r] = strassen_U[i][r]
                
            # V
            # V row index is j*p + k
            # Local V indices: 0->B0k0, 1->B0k1, 2->B1k0, 3->B1k1
            # Map local 0 to global j=0, k=k0 -> 0*p + k0
            # Map local 1 to global j=0, k=k1 -> 0*p + k1
            # Map local 2 to global j=1, k=k0 -> 1*p + k0
            # Map local 3 to global j=1, k=k1 -> 1*p + k1
            
            v_map = [
                0 * p + k0,
                0 * p + k1,
                1 * p + k0,
                1 * p + k1
            ]
            
            for l, v_row_idx in enumerate(v_map):
                V[v_row_idx][global_r] = strassen_V[l][r]
                
            # W
            # W row index is k*n + i
            # Local W indices: 0->C0k0, 1->C0k1, 2->C1k0, 3->C1k1
            # Map local 0 to global k=k0, i=0 -> k0*n + 0
            # Map local 1 to global k=k1, i=0 -> k1*n + 0
            # Map local 2 to global k=k0, i=1 -> k0*n + 1
            # Map local 3 to global k=k1, i=1 -> k1*n + 1
            
            w_map = [
                k0 * n + 0,
                k1 * n + 0,
                k0 * n + 1,
                k1 * n + 1
            ]
            
            for l, w_row_idx in enumerate(w_map):
                W[w_row_idx][global_r] = strassen_W[l][r]
                
        col_offset += 7
        
    for b in range(num_schoolbook_blocks):
        # Column k = 2*num_strassen_blocks + b
        k = 2 * num_strassen_blocks + b
        
        for r in range(4):
            global_r = col_offset + r
            
            # U
            for i in range(n * m):
                U[i][global_r] = schoolbook_U[i][r]
                
            # V
            # V row index is j*p + k
            # Local V indices: 0->B0k, 1->B1k
            # Map local 0 to global j=0, k=k -> 0*p + k
            # Map local 1 to global j=1, k=k -> 1*p + k
            
            v_map = [
                0 * p + k,
                1 * p + k
            ]
            
            for l, v_row_idx in enumerate(v_map):
                V[v_row_idx][global_r] = schoolbook_V[l][r]
                
            # W
            # W row index is k*n + i
            # Local W indices: 0->C0k, 1->C1k
            # Map local 0 to global k=k, i=0 -> k*n + 0
            # Map local 1 to global k=k, i=1 -> k*n + 1
            
            w_map = [
                k * n + 0,
                k * n + 1
            ]
            
            for l, w_row_idx in enumerate(w_map):
                W[w_row_idx][global_r] = schoolbook_W[l][r]
                
        col_offset += 4
        
    return {"U": U, "V": V, "W": W}
