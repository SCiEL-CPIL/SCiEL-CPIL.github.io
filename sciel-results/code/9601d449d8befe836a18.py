"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    # Block structure:
    # A: 4x5, split rows [0:2, 2:4], cols [0:2, 2:5]
    # B: 5x6, split rows [0:2, 2:5], cols [0:2, 2:4, 4:6]
    # C: 4x6, split rows [0:2, 2:4], cols [0:2, 2:4, 4:6]
    
    # 6 output blocks C[x][y] where x in {0,1}, y in {0,1,2}
    # C[x][y] = A[x][0] * B[0][y] + A[x][1] * B[1][y]
    # A[x][0] is 2x2, B[0][y] is 2x2 -> Strassen (rank 7)
    # A[x][1] is 2x3, B[1][y] is 3x2 -> schoolbook (rank 12)
    
    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
        A indices: (i,j) i,j in {0,1} -> flat index i*2+j
        B indices: (j,k) j,k in {0,1} -> flat index j*2+k
        C indices: (i,k) i,k in {0,1} -> flat index i*2+k (trace convention: k*n+i with n=2)
        Wait, the trace convention is W[k*n+i, r]. For 2x2, n=2, so W[k*2+i].
        """
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2x2 A -> 4 elements
        V = [[0]*rank for _ in range(4)]  # 2x2 B -> 4 elements
        W = [[0]*rank for _ in range(4)]  # 2x2 C -> 4 elements, W[k*2+i]
        
        # A flat: A[i*2+j]
        # B flat: B[j*2+k]
        # C flat: C[k*2+i] (trace convention)
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[0], A11 = A[3]
        # B00 = B[0], B11 = B[3]
        U[0][0] = 1; U[3][0] = 1  # A00 + A11
        V[0][0] = 1; V[3][0] = 1  # B00 + B11
        # C00 += M1, C11 += M1
        W[0][0] = 1  # C00: k=0,i=0 -> W[0*2+0]=W[0]
        W[3][0] = 1  # C11: k=1,i=1 -> W[1*2+1]=W[3]
        
        # M2 = (A10+A11)B00
        # A10 = A[2], A11 = A[3]
        # B00 = B[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M2
        W[2][1] = 1  # C10: k=0,i=1 -> W[0*2+1]=W[1]... wait
        
        # Let me re-check the trace convention. W[k*n+i, r] where n=2.
        # C[i,k] is stored at W[k*2+i].
        # C00: i=0,k=0 -> W[0]
        # C01: i=0,k=1 -> W[2]
        # C10: i=1,k=0 -> W[1]
        # C11: i=1,k=1 -> W[3]
        
        # Let me redo:
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1   # C00
        W[3][0] = 1   # C11
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1   # C10
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1  # B10 - B11
        W[2][2] = 1   # C01
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1  # B01 - B00
        W[0][3] = 1   # C00
        W[1][3] = 1   # C10
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1  # C00 -= M5
        W[2][4] = 1   # C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1   # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1   # C00 += M7
        
        return U, V, W
    
    # Schoolbook 2x3 * 3x2 = 2x2
    def schoolbook_2x3_3x2():
        """Returns U, V, W for 2x3 * 3x2 = 2x2 (rank 12).
        A: 2x3, flat index i*3+j, i in {0,1}, j in {0,1,2}
        B: 3x2, flat index j*2+k, j in {0,1,2}, k in {0,1}
        C: 2x2, flat index k*2+i (trace convention with n=2)
        """
        rank = 12
        U = [[0]*rank for _ in range(6)]  # 2x3 A
        V = [[0]*rank for _ in range(6)]  # 3x2 B
        W = [[0]*rank for _ in range(4)]  # 2x2 C, W[k*2+i]
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i*3+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    # Now build the full decomposition
    # 6 blocks, each with Strassen (7) + schoolbook (12) = 19
    total_rank = 6 * 19
    U = [[0]*total_rank for _ in range(20)]  # 4*5 = 20
    V = [[0]*total_rank for _ in range(30)]  # 5*6 = 30
    W = [[0]*total_rank for _ in range(24)]  # 6*4 = 24, W[k*4+i]
    
    # Block offsets:
    # A: 4x5
    #   A[x][0]: rows x*2:(x+1)*2, cols 0:2 -> 2x2
    #   A[x][1]: rows x*2:(x+1)*2, cols 2:5 -> 2x3
    # B: 5x6
    #   B[0][y]: rows 0:2, cols y*2:(y+1)*2 -> 2x2
    #   B[1][y]: rows 2:5, cols y*2:(y+1)*2 -> 3x2
    # C: 4x6
    #   C[x][y]: rows x*2:(x+1)*2, cols y*2:(y+1)*2 -> 2x2
    
    col_offset = 0
    for x in range(2):
        for y in range(3):
            # Strassen part: A[x][0] * B[0][y]
            sU, sV, sW = strassen_2x2x2()
            for i in range(4):
                for r in range(7):
                    U[x*5 + i][col_offset + r] += sU[i][r]
            for i in range(4):
                for r in range(7):
                    V[y*2 + i][col_offset + r] += sV[i][r]
            for i in range(4):
                for r in range(7):
                    # C index: k*4 + i, where C is 2x2 block at rows x*2:(x+1)*2, cols y*2:(y+1)*2
                    # sW[i] corresponds to C[k*2+i] in local 2x2
                    # Global C index: row = x*2 + i_local_row, col = y*2 + i_local_col
                    # But sW is indexed by k*2+i (local), where i is local row, k is local col
                    # Global: row = x*2 + i, col = y*2 + k
                    # W global index: col*4 + row = (y*2+k)*4 + (x*2+i)
                    # sW index: k*2 + i
                    local_idx = i
                    k_local = local_idx // 2
                    i_local = local_idx % 2
                    global_row = x*2 + i_local
                    global_col = y*2 + k_local
                    W[global_col*4 + global_row][col_offset + r] += sW[i][r]
            col_offset += 7
            
            # Schoolbook part: A[x][1] * B[1][y]
            bU, bV, bW = schoolbook_2x3_3x2()
            for i in range(6):
                for r in range(12):
                    # A[x][1] flat index: i_local = i, global = x*5 + 2 + i
                    U[x*5 + 2 + i][col_offset + r] += bU[i][r]
            for i in range(6):
                for r in range(12):
                    # B[1][y] flat index: i_local = i, global = 2 + y*2 + i
                    V[2 + y*2 + i][col_offset + r] += bV[i][r]
            for i in range(4):
                for r in range(12):
                    # C block: local index i = k_local*2 + i_local
                    local_idx = i
                    k_local = local_idx // 2
                    i_local = local_idx % 2
                    global_row = x*2 + i_local
                    global_col = y*2 + k_local
                    W[global_col*4 + global_row][col_offset + r] += bW[i][r]
            col_offset += 12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
