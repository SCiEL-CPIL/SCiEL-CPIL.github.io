"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2x13 matrix multiplication
    # The 2x2x13 tensor is decomposed into six 2x2x2 blocks (for columns 0-11) 
    # plus one 2x2x1 block (for column 12), using Strassen for the 2x2x2 blocks.
    n, m, p = DIMENSIONS  # 2, 2, 13
    rank = 46  # 6 blocks * 7 (Strassen rank for 2x2x2) + 1 block * 4 (schoolbook for 2x2x1)
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # Strassen decomposition for 2x2x2:
    # U (4x7), V (4x7), W (4x7) with indices 0..3 for each matrix
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen U, V, W for 2x2x2 (local indices 0..3)
    # U rows: A00=0, A10=1, A11=2, A01=3 (order: i*m+j, i=0,1; j=0,1)
    # Wait, let me be careful with indexing.
    # U index: i*m + j where i is row of A (0,1), j is col of A (0,1)
    # So U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
    # V index: j*p + k where j is row of B (0,1), k is col of B (0,1)
    # So V[0] = B00, V[1] = B01, V[2] = B10, V[3] = B11
    # W index: k*n + i where k is col of C (0,1), i is row of C (0,1)
    # So W[0] = C00, W[1] = C10, W[2] = C01, W[3] = C11
    
    # Strassen U (4x7):
    # M1: A00 + A11 -> U[0] += 1, U[3] += 1
    # M2: A10 + A11 -> U[2] += 1, U[3] += 1
    # M3: A00 -> U[0] += 1
    # M4: A11 -> U[3] += 1
    # M5: A00 + A01 -> U[0] += 1, U[1] += 1
    # M6: A10 - A00 -> U[2] += 1, U[0] -= 1
    # M7: A01 - A11 -> U[1] += 1, U[3] -= 1
    
    U_strassen = [[0]*7 for _ in range(4)]
    # M1
    U_strassen[0][0] = 1
    U_strassen[3][0] = 1
    # M2
    U_strassen[2][1] = 1
    U_strassen[3][1] = 1
    # M3
    U_strassen[0][2] = 1
    # M4
    U_strassen[3][3] = 1
    # M5
    U_strassen[0][4] = 1
    U_strassen[1][4] = 1
    # M6
    U_strassen[2][5] = 1
    U_strassen[0][5] = -1
    # M7
    U_strassen[1][6] = 1
    U_strassen[3][6] = -1
    
    # Strassen V (4x7):
    # M1: B00 + B11 -> V[0] += 1, V[3] += 1
    # M2: B00 -> V[0] += 1
    # M3: B01 - B11 -> V[1] += 1, V[3] -= 1
    # M4: B10 - B00 -> V[2] += 1, V[0] -= 1
    # M5: B11 -> V[3] += 1
    # M6: B00 + B01 -> V[0] += 1, V[1] += 1
    # M7: B10 + B11 -> V[2] += 1, V[3] += 1
    
    V_strassen = [[0]*7 for _ in range(4)]
    # M1
    V_strassen[0][0] = 1
    V_strassen[3][0] = 1
    # M2
    V_strassen[0][1] = 1
    # M3
    V_strassen[1][2] = 1
    V_strassen[3][2] = -1
    # M4
    V_strassen[2][3] = 1
    V_strassen[0][3] = -1
    # M5
    V_strassen[3][4] = 1
    # M6
    V_strassen[0][5] = 1
    V_strassen[1][5] = 1
    # M7
    V_strassen[2][6] = 1
    V_strassen[3][6] = 1
    
    # Strassen W (4x7):
    # C00 = M1 + M4 - M5 + M7 -> W[0] = M1+M4-M5+M7
    # C10 = M2 + M4 -> W[1] = M2+M4
    # C01 = M3 + M5 -> W[2] = M3+M5
    # C11 = M1 - M2 + M3 + M6 -> W[3] = M1-M2+M3+M6
    
    W_strassen = [[0]*7 for _ in range(4)]
    # C00
    W_strassen[0][0] = 1  # M1
    W_strassen[0][3] = 1  # M4
    W_strassen[0][4] = -1 # M5
    W_strassen[0][6] = 1  # M7
    # C10
    W_strassen[1][1] = 1  # M2
    W_strassen[1][3] = 1  # M4
    # C01
    W_strassen[2][2] = 1  # M3
    W_strassen[2][4] = 1  # M5
    # C11
    W_strassen[3][0] = 1  # M1
    W_strassen[3][1] = -1 # M2
    W_strassen[3][2] = 1  # M3
    W_strassen[3][5] = 1  # M6
    
    # Now build the global U, V, W
    # For each of the 6 Strassen blocks (k=0..11 in pairs), and 1 schoolbook block (k=12)
    
    # Process 6 Strassen blocks
    for b in range(6):
        base_col = b * 7
        # U: same for all blocks, just place in U[0..3][base_col..base_col+6]
        for i in range(4):
            for c in range(7):
                U[i][base_col + c] += U_strassen[i][c]
        
        # V: local V indices 0..3 correspond to B0k, B0k+1, B1k, B1k+1
        # Global V index: j*p + k where j is row of B (0,1), k is col of B
        # Local V[0] -> B0,2b -> global index 0*13 + 2b = 2b
        # Local V[1] -> B0,2b+1 -> global index 0*13 + 2b+1 = 2b+1
        # Local V[2] -> B1,2b -> global index 1*13 + 2b = 13+2b
        # Local V[3] -> B1,2b+1 -> global index 1*13 + 2b+1 = 13+2b+1
        v_map = [2*b, 2*b+1, 13+2*b, 13+2*b+1]
        for c in range(7):
            for local_v in range(4):
                val = V_strassen[local_v][c]
                if val != 0:
                    V[v_map[local_v]][base_col + c] += val
        
        # W: local W indices 0..3 correspond to C0,2b; C1,2b; C0,2b+1; C1,2b+1
        # Global W index: k*n + i where k is col of C, i is row of C
        # Local W[0] -> C0,2b -> global index 2b*2 + 0 = 4b
        # Local W[1] -> C1,2b -> global index 2b*2 + 1 = 4b+1
        # Local W[2] -> C0,2b+1 -> global index (2b+1)*2 + 0 = 4b+2
        # Local W[3] -> C1,2b+1 -> global index (2b+1)*2 + 1 = 4b+3
        w_map = [4*b, 4*b+1, 4*b+2, 4*b+3]
        for c in range(7):
            for local_w in range(4):
                val = W_strassen[local_w][c]
                if val != 0:
                    W[w_map[local_w]][base_col + c] += val
    
    # Process the last schoolbook block (k=12)
    base_col = 6 * 7  # = 42
    # Schoolbook for 2x2x1: rank 4
    # U: A00, A01, A10, A11 -> indices 0,1,2,3
    # V: B0, B1 -> indices 0,1 (since k=12 only)
    # W: C0, C1 -> indices 0,1 (since k=12 only)
    
    # Column 42: A00 * B0 -> C0
    U[0][base_col] = 1
    V[12][base_col] = 1  # B row 0, col 12 -> index 0*13+12 = 12
    W[24][base_col] = 1  # C row 0, col 12 -> index 12*2+0 = 24
    
    # Column 43: A01 * B1 -> C0
    U[1][base_col + 1] = 1
    V[13][base_col + 1] = 1  # B row 0, col 13 -> index 0*13+13 = 13
    W[24][base_col + 1] = 1  # C row 0, col 13 -> index 13*2+0 = 26
    # Wait, k is always 12 for this block. Let me reconsider.
    # For 2x2x1, k=12 only.
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices: B012=12, B112=25 (B row j, col 12 -> j*13+12)
    # W indices: C012=24, C112=25 (C row i, col 12 -> 12*2+i)
    
    # Let me redo this more carefully.
    # For 2x2x1 with k=12:
    # C012 = A00*B012 + A01*B112
    # C112 = A10*B012 + A11*B112
    
    # Column 42: A00 * B012 -> C012
    U[0][base_col] = 1
    V[12][base_col] = 1
    W[24][base_col] = 1
    
    # Column 43: A01 * B112 -> C012
    U[1][base_col + 1] = 1
    V[25][base_col + 1] = 1
    W[24][base_col + 1] = 1
    
    # Column 44: A10 * B012 -> C112
    U[2][base_col + 2] = 1
    V[12][base_col + 2] = 1
    W[25][base_col + 2] = 1
    
    # Column 45: A11 * B112 -> C112
    U[3][base_col + 3] = 1
    V[25][base_col + 3] = 1
    W[25][base_col + 3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
