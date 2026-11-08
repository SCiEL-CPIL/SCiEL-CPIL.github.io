import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Total rank is 48
    # 3 blocks of 2x2x2 (rank 7) + 3 blocks of 2x3x2 (rank 9)
    rank = 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_off, j_off, k_off, c_start):
        nonlocal col
        # 7 terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[i_off*5 + j_off][c_start] += 1
        U[(i_off+1)*5 + (j_off+1)][c_start] += 1
        V[(j_off)*6 + k_off][c_start] += 1
        V[(j_off+1)*6 + (k_off+1)][c_start] += 1
        W[(k_off)*2 + i_off][c_start] += 1
        W[(k_off+1)*2 + (i_off+1)][c_start] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_off+1)*5 + j_off][col] += 1
        U[(i_off+1)*5 + (j_off+1)][col] += 1
        V[j_off*6 + k_off][col] += 1
        W[(k_off+1)*2 + (i_off+1)][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_off*5 + j_off][col] += 1
        V[(j_off)*6 + (k_off+1)][col] += 1
        V[(j_off+1)*6 + (k_off+1)][col] -= 1
        W[k_off*2 + i_off][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i_off+1)*5 + (j_off+1)][col] += 1
        V[(j_off+1)*6 + (k_off+1)][col] += 1
        V[j_off*6 + k_off][col] -= 1
        W[(k_off+1)*2 + (i_off+1)][col] += 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[i_off*5 + j_off][col] += 1
        U[(i_off+1)*5 + j_off][col] += 1
        V[(j_off+1)*6 + (k_off+1)][col] += 1
        W[(k_off+1)*2 + i_off][col] += 1
        col += 1
        
        # M6 = (A01-A00)(B00+B10)
        U[i_off*5 + (j_off+1)][col] += 1
        U[i_off*5 + j_off][col] -= 1
        V[j_off*6 + k_off][col] += 1
        V[(j_off+1)*6 + k_off][col] += 1
        W[k_off*2 + i_off][col] += 1
        col += 1
        
        # M7 = (A10-A11)(B01+B11)
        U[(i_off+1)*5 + j_off][col] += 1
        U[(i_off+1)*5 + (j_off+1)][col] -= 1
        V[j_off*6 + (k_off+1)][col] += 1
        V[(j_off+1)*6 + (k_off+1)][col] += 1
        W[(k_off+1)*2 + (i_off+1)][col] += 1
        col += 1

    # Helper to add a 2x3x2 block (Rank 9)
    # Splits into 2x2x2 (Strassen) + 2x1x2 (Naive)
    def add_2x3x2(i_off, j_off, k_off, c_start):
        # 2x2x2 part on j_off, j_off+1
        add_strassen(i_off, j_off, k_off, c_start)
        
        # 2x1x2 part on j_off+2
        # C_i0 += A_i2 B_20
        # C_i1 += A_i2 B_21
        
        # Term 1: A_i2 B_20 -> C_i0
        U[i_off*5 + (j_off+2)][col] += 1
        U[(i_off+1)*5 + (j_off+2)][col] += 1
        V[(j_off+2)*6 + k_off][col] += 1
        W[k_off*2 + i_off][col] += 1
        W[k_off*2 + (i_off+1)][col] += 1 # Wait, C_i0 and C_i1?
        # M = A_i2 * B_20.
        # This contributes to C_i0.
        # W index is k*n + i.
        # For C_i0, k=0, i=i.
        # So W[k_off*2 + i_off] += 1.
        # And W[k_off*2 + (i_off+1)] += 1?
        # No, A_i2 is for i=0 and i=1.
        # U has 1 for i=0 and i=1.
        # V has 1 for j=2, k=0.
        # W should have 1 for k=0, i=0 AND k=0, i=1?
        # No, the term is A_i2 B_20.
        # If i=0, term is A_02 B_20. Contributes to C_00.
        # If i=1, term is A_12 B_20. Contributes to C_10.
        # So W should be 1 at (k=0, i=0) and (k=0, i=1).
        # My previous line: W[k_off*2 + i_off] += 1 and W[k_off*2 + (i_off+1)] += 1.
        # This is correct.
        col += 1
        
        # Term 2: A_i2 B_21 -> C_i1
        U[i_off*5 + (j_off+2)][col] += 1
        U[(i_off+1)*5 + (j_off+2)][col] += 1
        V[(j_off+2)*6 + (k_off+1)][col] += 1
        W[(k_off+1)*2 + i_off][col] += 1
        W[(k_off+1)*2 + (i_off+1)][col] += 1
        col += 1

    # Block 1: J1={0,1}, K1={0,1}
    add_strassen(0, 0, 0, col)
    # Block 2: J1={0,1}, K2={2,3}
    add_strassen(0, 0, 2, col)
    # Block 3: J1={0,1}, K3={4,5}
    add_strassen(0, 0, 4, col)
    
    # Block 4: J2={2,3,4}, K1={0,1}
    add_2x3x2(0, 2, 0, col)
    # Block 5: J2={2,3,4}, K2={2,3}
    add_2x3x2(0, 2, 2, col)
    # Block 6: J2={2,3,4}, K3={4,5}
    add_2x3x2(0, 2, 4, col)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
