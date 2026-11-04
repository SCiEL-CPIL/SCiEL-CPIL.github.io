import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 28 decomposition using 4 Strassen algorithms
    # A is 2x4, B is 4x4
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B11, B12, B21, B22 (each 2x2)
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # We will use 4 Strassen decompositions:
    # P1 = A1*B11
    # P2 = A2*B21
    # P3 = A1*B12
    # P4 = A2*B22
    
    # Each Strassen uses 7 multiplications. Total 28.
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for A (2x4):
    # A1: rows 0,1 cols 0,1 -> indices 0,1,2,3
    # A2: rows 0,1 cols 2,3 -> indices 4,5,6,7
    # Actually, A is 2x4.
    # Row 0: A[0,0], A[0,1], A[0,2], A[0,3]
    # Row 1: A[1,0], A[1,1], A[1,2], A[1,3]
    # A1 = [[A00, A01], [A10, A11]]
    # A2 = [[A02, A03], [A12, A13]]
    
    # Indices for B (4x4):
    # B11: rows 0,1 cols 0,1
    # B12: rows 0,1 cols 2,3
    # B21: rows 2,3 cols 0,1
    # B22: rows 2,3 cols 2,3
    
    # Strassen for X (2x2) * Y (2x2)
    # X = [[x0, x1], [x2, x3]]
    # Y = [[y0, y1], [y2, y3]]
    # M1 = x0*(y0+y3)
    # M2 = (x0+x2)*y3
    # M3 = (x1+x3)*y0
    # M4 = x3*(y0-y3)
    # M5 = (x0+x3)*(y2-y1)
    # M6 = (x0-x1)*(y1+y3)
    # M7 = (x2-x3)*(y0+y2)
    
    # C = [[M1+M4-M5+M6, M3+M5+M6], [M2-M4+M5+M7, M1-M2+M3-M7]]
    
    def add_strassen(u_offset, v_offset, w_offset, a_indices, b_indices, c_indices):
        # a_indices: 4 indices for A block (x0, x1, x2, x3)
        # b_indices: 4 indices for B block (y0, y1, y2, y3)
        # c_indices: 4 indices for C block (c00, c01, c10, c11)
        
        # M1
        U[a_indices[0], u_offset] += 1
        V[b_indices[0], u_offset] += 1
        V[b_indices[3], u_offset] += 1
        W[c_indices[0], u_offset] += 1
        W[c_indices[3], u_offset] -= 1
        
        # M2
        U[a_indices[0], u_offset+1] += 1
        U[a_indices[2], u_offset+1] += 1
        V[b_indices[3], u_offset+1] += 1
        W[c_indices[1], u_offset+1] += 1
        W[c_indices[2], u_offset+1] -= 1
        
        # M3
        U[a_indices[1], u_offset+2] += 1
        U[a_indices[3], u_offset+2] += 1
        V[b_indices[0], u_offset+2] += 1
        W[c_indices[0], u_offset+2] += 1
        W[c_indices[3], u_offset+2] -= 1
        
        # M4
        U[a_indices[3], u_offset+3] += 1
        V[b_indices[0], u_offset+3] += 1
        V[b_indices[3], u_offset+3] -= 1
        W[c_indices[0], u_offset+3] += 1
        W[c_indices[1], u_offset+3] -= 1
        
        # M5
        U[a_indices[0], u_offset+4] += 1
        U[a_indices[3], u_offset+4] += 1
        V[b_indices[2], u_offset+4] += 1
        V[b_indices[1], u_offset+4] -= 1
        W[c_indices[1], u_offset+4] += 1
        W[c_indices[2], u_offset+4] += 1
        
        # M6
        U[a_indices[0], u_offset+5] += 1
        U[a_indices[1], u_offset+5] -= 1
        V[b_indices[1], u_offset+5] += 1
        V[b_indices[3], u_offset+5] += 1
        W[c_indices[0], u_offset+5] += 1
        W[c_indices[1], u_offset+5] += 1
        
        # M7
        U[a_indices[2], u_offset+6] += 1
        U[a_indices[3], u_offset+6] -= 1
        V[b_indices[0], u_offset+6] += 1
        V[b_indices[2], u_offset+6] += 1
        W[c_indices[2], u_offset+6] += 1
        W[c_indices[3], u_offset+6] -= 1

    # A1 indices: 0,1,2,3 (A00, A01, A10, A11)
    # A2 indices: 4,5,6,7 (A02, A03, A12, A13)
    
    # B11 indices: 0,1,4,5 (B00, B01, B10, B11)
    # B12 indices: 2,3,6,7 (B02, B03, B12, B13)
    # B21 indices: 8,9,12,13 (B20, B21, B30, B31)
    # B22 indices: 10,11,14,15 (B22, B23, B32, B33)
    
    # C1 indices: 0,1,2,3 (C00, C01, C10, C11)
    # C2 indices: 4,5,6,7 (C02, C03, C12, C13)
    
    a1 = [0, 1, 2, 3]
    a2 = [4, 5, 6, 7]
    
    b11 = [0, 1, 4, 5]
    b12 = [2, 3, 6, 7]
    b21 = [8, 9, 12, 13]
    b22 = [10, 11, 14, 15]
    
    c1 = [0, 1, 2, 3]
    c2 = [4, 5, 6, 7]
    
    add_strassen(0, 0, 0, a1, b11, c1)
    add_strassen(7, 0, 0, a2, b21, c1)
    add_strassen(14, 0, 0, a1, b12, c2)
    add_strassen(21, 0, 0, a2, b22, c2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
