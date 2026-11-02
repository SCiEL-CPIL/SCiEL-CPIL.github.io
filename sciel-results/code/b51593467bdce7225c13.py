    def block_decomposition():
        U_full = np.zeros((12, 44))
        V_full = np.zeros((16, 44))
        W_full = np.zeros((12, 44))
        
        sU, sV, sW = strassen_2x2()
        
        # Define the 4 blocks
        # b=0: A(0,1) * B(0,1; 0,1) -> C(:, 0,1)
        # b=1: A(0,1) * B(0,1; 2,3) -> C(:, 2,3)
        # b=2: A(2,3) * B(2,3; 0,1) -> C(:, 0,1)
        # b=3: A(2,3) * B(2,3; 2,3) -> C(:, 2,3)
        
        for b in range(4):
            col_offset = b * 11
            
            # Determine slices
            if b < 2:
                a_col_start = 0
                b_row_start = 0
            else:
                a_col_start = 2
                b_row_start = 2
                
            if b % 2 == 0:
                b_col_start = 0
                c_col_start = 0
            else:
                b_col_start = 2
                c_col_start = 2
            
            # Local U: 3 rows, 2 cols of A
            # Local V: 2 rows, 2 cols of B
            # Local W: 3 rows, 2 cols of C
            
            # Construct local U, V, W for 3x2x2
            U_loc = np.zeros((6, 11))
            V_loc = np.zeros((4, 11))
            W_loc = np.zeros((6, 11))
            
            # Part 1: Strassen for rows 0,1 of A (local indices 0,1,2,3 in U_loc? No, U_loc is 6 rows)
            # A_local:
            # (0,0) -> U_loc[0]
            # (0,1) -> U_loc[1]
            # (1,0) -> U_loc[2]
            # (1,1) -> U_loc[3]
            # (2,0) -> U_loc[4]
            # (2,1) -> U_loc[5]
            
            # sU is (4,7). Maps to A[0,0], A[0,1], A[1,0], A[1,1]
            U_loc[0:4, 0:7] = sU
            
            # sV is (4,7). Maps to B[0,0], B[0,1], B[1,0], B[1,1]
            V_loc[0:4, 0:7] = sV
            
            # sW is (4,7). Maps to C[0,0], C[0,1], C[1,0], C[1,1]
            # C_local:
            # (0,0) -> W_loc[0]
            # (0,1) -> W_loc[1]
            # (1,0) -> W_loc[2]
            # (1,1) -> W_loc[3]
            # (2,0) -> W_loc[4]
            # (2,1) -> W_loc[5]
            
            W_loc[0:4, 0:7] = sW
            
            # Part 2: Trivial for row 2 of A
            # Products 7-10 (indices 7,8,9,10)
            
            # P7: A[2,0] * B[0,0] -> C[2,0]
            U_loc[4, 7] = 1
            V_loc[0, 7] = 1
            W_loc[4, 7] = 1
            
            # P8: A[2,0] * B[0,1] -> C[2,1]
            U_loc[4, 8] = 1
            V_loc[1, 8] = 1
            W_loc[5, 8] = 1
            
            # P9: A[2,1] * B[1,0] -> C[2,0]
            U_loc[5, 9] = 1
            V_loc[2, 9] = 1
            W_loc[4, 9] = 1
            
            # P10: A[2,1] * B[1,1] -> C[2,1]
            U_loc[5, 10] = 1
            V_loc[3, 10] = 1
            W_loc[5, 10] = 1
            
            # Map to global matrices
            
            # U: A[i, j] -> U_full[i*4 + j, col]
            # i in 0..2, j in a_col_start..a_col_start+1
            for i in range(3):
                for jj in range(2):
                    global_row = i * 4 + (a_col_start + jj)
                    local_row = i * 2 + jj
                    U_full[global_row, col_offset:col_offset+11] = U_loc[local_row, :]
                    
            # V: B[j, k] -> V_full[j*4 + k, col]
            # j in b_row_start..b_row_start+1, k in b_col_start..b_col_start+1
            for jj in range(2):
                for kk in range(2):
                    global_row = (b_row_start + jj) * 4 + (b_col_start + kk)
                    local_row = jj * 2 + kk
                    V_full[global_row, col_offset:col_offset+11] = V_loc[local_row, :]
                    
            # W: C[i, k] -> W_full[k*3 + i, col] (Trace convention W[k*n+i, r])
            # i in 0..2, k in c_col_start..c_col_start+1
            for i in range(3):
                for kk in range(2):
                    global_row = (c_col_start + kk) * 3 + i
                    local_row = i * 2 + kk
                    W_full[global_row, col_offset:col_offset+11] = W_loc[local_row, :]
                    
        return U_full, V_full, W_full
