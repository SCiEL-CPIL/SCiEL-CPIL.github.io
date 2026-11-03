"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 52 decomposition (2 blocks of rank 26)
    rank = 52
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in zip(u_indices, [1]*len(u_indices)):
            U[idx][col] += val
        for idx, val in zip(v_indices, [1]*len(v_indices)):
            V[idx][col] += val
        for idx, val in zip(w_indices, [1]*len(w_indices)):
            W[idx][col] += val

    # We need to construct the rank-26 decomposition for one 3x3x3 block
    # and apply it twice (for cols 0-2 and 3-5 of B)
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # Local A indices: 0,1,2,3 -> Global A: 4,5,7,8 (for block 22)
    # Local B indices: 0,1,2,3 -> Global B: 4,5,7,8 (for block 22)
    # Local C indices: 0,1,2,3 -> Global C: 4,5,7,8 (for block 22)
    
    # We will define the terms for one 3x3x3 block relative to local 0..8 indices
    # Then map to global.
    
    # Block 1: B cols 0-2, C cols 0-2
    # Block 2: B cols 3-5, C cols 3-5
    
    col_offset = 0
    
    for block in range(2):
        # Offset for B and C
        b_offset = block * 3
        c_offset = block * 3
        
        # 1. A22 B22 (Rank 7, Strassen)
        # A22: 4,5,7,8
        # B22: 4+b_offset, 5+b_offset, 7+b_offset, 8+b_offset
        # C22: 4+c_offset, 5+c_offset, 7+c_offset, 8+c_offset
        
        a_22 = [4, 5, 7, 8]
        b_22 = [4+b_offset, 5+b_offset, 7+b_offset, 8+b_offset]
        c_22 = [4+c_offset, 5+c_offset, 7+c_offset, 8+c_offset]
        
        # M1: (a+d)(e+h)
        U[a_22[0]][col_offset] = 1; U[a_22[3]][col_offset] = 1
        V[b_22[0]][col_offset] = 1; V[b_22[3]][col_offset] = 1
        W[c_22[0]][col_offset] = 1; W[c_22[3]][col_offset] = 1
        col_offset += 1
        
        # M2: (c+d)e
        U[a_22[2]][col_offset] = 1; U[a_22[3]][col_offset] = 1
        V[b_22[0]][col_offset] = 1
        W[c_22[2]][col_offset] = 1 # C21 local index 2 is global c_22[2]? 
        # Local C: 00, 01, 10, 11 -> 4,5,7,8
        # C21 is local 10 -> global 7? 
        # c_22 = [4,5,7,8]. Index 2 is 7. Correct.
        col_offset += 1
        
        # M3: a(f-h)
        U[a_22[0]][col_offset] = 1
        V[b_22[1]][col_offset] = 1; V[b_22[3]][col_offset] = -1
        W[c_22[1]][col_offset] = 1 # C12 local 01 -> 5
        col_offset += 1
        
        # M4: d(g-f)
        U[a_22[3]][col_offset] = 1
        V[b_22[2]][col_offset] = 1; V[b_22[1]][col_offset] = -1
        W[c_22[2]][col_offset] = 1 # C21
        W[c_22[3]][col_offset] = -1 # C22
        col_offset += 1
        
        # M5: (a+b)h
        U[a_22[0]][col_offset] = 1; U[a_22[1]][col_offset] = 1
        V[b_22[3]][col_offset] = 1
        W[c_22[0]][col_offset] = -1 # C11
        W[c_22[1]][col_offset] = 1 # C12
        col_offset += 1
        
        # M6: (c-a)(e+f)
        U[a_22[2]][col_offset] = 1; U[a_22[0]][col_offset] = -1
        V[b_22[0]][col_offset] = 1; V[b_22[1]][col_offset] = 1
        W[c_22[3]][col_offset] = 1 # C22
        col_offset += 1
        
        # M7: (b-d)(g+h)
        U[a_22[1]][col_offset] = 1; U[a_22[3]][col_offset] = -1
        V[b_22[2]][col_offset] = 1; V[b_22[3]][col_offset] = 1
        W[c_22[0]][col_offset] = 1 # C11
        W[c_22[3]][col_offset] = -1 # C22
        col_offset += 1
        
        # 2. A22 B21 (Rank 4, Schoolbook)
        # A22: 4,5,7,8
        # B21: 3+b_offset, 6+b_offset (Col 0 of B, Rows 1,2)
        # C21: 3+c_offset, 6+c_offset (Col 0 of C, Rows 1,2)
        
        b_21 = [3+b_offset, 6+b_offset]
        c_21 = [3+c_offset, 6+c_offset]
        
        # A22[0]*B21[0] -> C21[0]
        U[a_22[0]][col_offset] = 1; V[b_21[0]][col_offset] = 1; W[c_21[0]][col_offset] = 1
        col_offset += 1
        # A22[1]*B21[0] -> C21[1]
        U[a_22[1]][col_offset] = 1; V[b_21[0]][col_offset] = 1; W[c_21[1]][col_offset] = 1
        col_offset += 1
        # A22[2]*B21[0] -> C21[0]
        U[a_22[2]][col_offset] = 1; V[b_21[0]][col_offset] = 1; W[c_21[0]][col_offset] = 1
        col_offset += 1
        # A22[3]*B21[0] -> C21[1]
        U[a_22[3]][col_offset] = 1; V[b_21[0]][col_offset] = 1; W[c_21[1]][col_offset] = 1
        col_offset += 1
        
        # 3. A21 B12 (Rank 4, Schoolbook)
        # A21: 3,6 (Col 0 of A, Rows 1,2)
        # B12: 1+b_offset, 2+b_offset (Row 0 of B, Cols 1,2)
        # C12: 1+c_offset, 2+c_offset (Row 0 of C, Cols 1,2)
        
        a_21 = [3, 6]
        b_12 = [1+b_offset, 2+b_offset]
        c_12 = [1+c_offset, 2+c_offset]
        
        # A21[0]*B12[0] -> C12[0]
        U[a_21[0]][col_offset] = 1; V[b_12[0]][col_offset] = 1; W[c_12[0]][col_offset] = 1
        col_offset += 1
        # A21[0]*B12[1] -> C12[1]
        U[a_21[0]][col_offset] = 1; V[b_12[1]][col_offset] = 1; W[c_12[1]][col_offset] = 1
        col_offset += 1
        # A21[1]*B12[0] -> C12[0]
        U[a_21[1]][col_offset] = 1; V[b_12[0]][col_offset] = 1; W[c_12[0]][col_offset] = 1
        col_offset += 1
        # A21[1]*B12[1] -> C12[1]
        U[a_21[1]][col_offset] = 1; V[b_12[1]][col_offset] = 1; W[c_12[1]][col_offset] = 1
        col_offset += 1
        
        # 4. A21 B11 (Rank 2, Schoolbook)
        # A21: 3,6
        # B11: 0+b_offset
        # C11: 0+c_offset
        
        b_11 = [0+b_offset]
        c_11 = [0+c_offset]
        
        U[a_21[0]][col_offset] = 1; V[b_11[0]][col_offset] = 1; W[c_11[0]][col_offset] = 1
        col_offset += 1
        U[a_21[1]][col_offset] = 1; V[b_11[0]][col_offset] = 1; W[c_11[0]][col_offset] = 1
        col_offset += 1
        
        # 5. A12 B22 (Rank 4, Schoolbook)
        # A12: 1,2 (Row 0 of A, Cols 1,2)
        # B22: 4+b_offset, 5+b_offset, 7+b_offset, 8+b_offset
        # C22: 4+c_offset, 5+c_offset, 7+c_offset, 8+c_offset
        # Wait, C indices for A12*B22:
        # A12 is i=0, j=1,2. B22 is j=1,2, k=1,2.
        # Result C is i=0, k=1,2.
        # C indices: 3+c_offset (k=1,i=0), 6+c_offset (k=2,i=0).
        
        a_12 = [1, 2]
        c_12_out = [3+c_offset, 6+c_offset]
        
        # A12[0]*B22[0] -> C[3]
        U[a_12[0]][col_offset] = 1; V[b_22[0]][col_offset] = 1; W[c_12_out[0]][col_offset] = 1
        col_offset += 1
        # A12[0]*B22[1] -> C[6]
        U[a_12[0]][col_offset] = 1; V[b_22[1]][col_offset] = 1; W[c_12_out[1]][col_offset] = 1
        col_offset += 1
        # A12[1]*B22[2] -> C[3]
        U[a_12[1]][col_offset] = 1; V[b_22[2]][col_offset] = 1; W[c_12_out[0]][col_offset] = 1
        col_offset += 1
        # A12[1]*B22[3] -> C[6]
        U[a_12[1]][col_offset] = 1; V[b_22[3]][col_offset] = 1; W[c_12_out[1]][col_offset] = 1
        col_offset += 1
        
        # 6. A12 B21 (Rank 2, Schoolbook)
        # A12: 1,2
        # B21: 3+b_offset, 6+b_offset
        # C11: 0+c_offset (i=0, k=0)
        
        U[a_12[0]][col_offset] = 1; V[b_21[0]][col_offset] = 1; W[c_11[0]][col_offset] = 1
        col_offset += 1
        U[a_12[1]][col_offset] = 1; V[b_21[1]][col_offset] = 1; W[c_11[0]][col_offset] = 1
        col_offset += 1
        
        # 7. A11 B12 (Rank 2, Schoolbook)
        # A11: 0
        # B12: 1+b_offset, 2+b_offset
        # C12: 3+c_offset, 6+c_offset (i=0, k=1,2)
        
        a_11 = [0]
        
        U[a_11[0]][col_offset] = 1; V[b_12[0]][col_offset] = 1; W[c_12_out[0]][col_offset] = 1
        col_offset += 1
        U[a_11[0]][col_offset] = 1; V[b_12[1]][col_offset] = 1; W[c_12_out[1]][col_offset] = 1
        col_offset += 1
        
        # 8. A11 B11 (Rank 1, Schoolbook)
        # A11: 0
        # B11: 0+b_offset
        # C11: 0+c_offset
        
        U[a_11[0]][col_offset] = 1; V[b_11[0]][col_offset] = 1; W[c_11[0]][col_offset] = 1
        col_offset += 1

    if col_offset != rank:
        # Fallback if logic error
        return {"U": [[0]*54 for _ in range(9)], "V": [[0]*54 for _ in range(18)], "W": [[0]*54 for _ in range(18)]}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
