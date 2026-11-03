"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    # Decompose 2x5x5 into blocks: rows (2), cols of A (5), cols of B (5)
    # A is n x m = 2x5, B is m x p = 5x5, C is n x p = 2x5
    # Block structure: m = 2+2+1, p = 2+2+1
    
    # We'll use Strassen for 2x2x2 blocks and direct for 2x1x1
    # R(2,2,2) = 7, R(2,1,1) = 2
    # Total: 4*7 + 2*2 = 32
    
    # Strassen 2x2x2 decomposition
    # U for 2x2 input (rows of A block), V for 2x2 input (cols of B block), W for 2x2 output
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        """Return U, V, W matrices for 2x2x2 Strassen decomposition.
        U: 2x7 (input A rows), V: 2x7 (input B cols), W: 4x7 (output C)
        Indexing: A is 2x2, B is 2x2, C is 2x2
        U[row, col] means coefficient for A[row, 0 or 1] in product col
        Actually, for matrix multiplication tensor:
        U[i*m+j, c] for A[i,j], V[j*p+k, c] for B[j,k], W[k*n+i, c] for C[k,n+i]
        For 2x2x2: m=2, p=2, n=2
        U: 4x7, V: 4x7, W: 4x7
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        # a00 = A[0,0], a11 = A[1,1]
        # b00 = B[0,0], b11 = B[1,1]
        U[0*2+0][0] = 1  # a00
        U[1*2+1][0] = 1  # a11
        V[0*2+0][0] = 1  # b00
        V[1*2+1][0] = 1  # b11
        # c00 += M1, c11 += M1
        W[0*2+0][0] = 1  # c[0,0]
        W[1*2+1][0] = 1  # c[1,1]
        
        # M2 = (a10+a11)(b00)
        U[1*2+0][1] = 1  # a10
        U[1*2+1][1] = 1  # a11
        V[0*2+0][1] = 1  # b00
        # c10 += M2
        W[0*2+1][1] = 1  # c[1,0]
        
        # M3 = (a00)(b01-b11)
        U[0*2+0][2] = 1  # a00
        V[0*2+1][2] = 1  # b01
        V[1*2+1][2] = -1  # b11
        # c01 += M3, c11 += M3
        W[1*2+0][2] = 1  # c[0,1]
        W[1*2+1][2] = 1  # c[1,1]
        
        # M4 = (a11)(b10-b00)
        U[1*2+1][3] = 1  # a11
        V[1*2+0][3] = 1  # b10
        V[0*2+0][3] = -1  # b00
        # c00 += M4, c10 += M4
        W[0*2+0][3] = 1  # c[0,0]
        W[0*2+1][3] = 1  # c[1,0]
        
        # M5 = (a00+a01)(b11)
        U[0*2+0][4] = 1  # a00
        U[0*2+1][4] = 1  # a01
        V[1*2+1][4] = 1  # b11
        # c00 -= M5, c01 += M5
        W[0*2+0][4] = -1  # c[0,0]
        W[1*2+0][4] = 1  # c[0,1]
        
        # M6 = (a10-a00)(b00+b01)
        U[1*2+0][5] = 1  # a10
        U[0*2+0][5] = -1  # a00
        V[0*2+0][5] = 1  # b00
        V[0*2+1][5] = 1  # b01
        # c11 += M6
        W[1*2+1][5] = 1  # c[1,1]
        
        # M7 = (a01-a11)(b10+b11)
        U[0*2+1][6] = 1  # a01
        U[1*2+1][6] = -1  # a11
        V[1*2+0][6] = 1  # b10
        V[1*2+1][6] = 1  # b11
        # c00 += M7
        W[0*2+0][6] = 1  # c[0,0]
        
        return U, V, W
    
    def direct_2x1x1():
        """Return U, V, W for 2x1x1 direct multiplication.
        A is 2x1, B is 1x1, C is 2x1
        U: 2x2, V: 1x2, W: 2x2
        U[i*1+j, c] = U[i, c], V[j*1+k, c] = V[k, c], W[k*1+i, c] = W[i, c]
        c0 = a0*b0, c1 = a1*b0
        """
        rank = 2
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(1)]
        W = [[0]*rank for _ in range(2)]
        
        U[0][0] = 1  # a0
        V[0][0] = 1  # b0
        W[0][0] = 1  # c0
        
        U[1][1] = 1  # a1
        V[0][1] = 1  # b0
        W[1][1] = 1  # c1
        
        return U, V, W
    
    # Block structure for 2x5x5:
    # A rows: 2 (i=0,1)
    # A cols / B rows: 5 (j=0,1,2,3,4) -> blocks: [0,1], [2,3], [4]
    # B cols: 5 (k=0,1,2,3,4) -> blocks: [0,1], [2,3], [4]
    
    # Block (j_block, k_block) where j_block in {0,1,2}, k_block in {0,1,2}
    # j_block 0: j in [0,1], size 2
    # j_block 1: j in [2,3], size 2
    # j_block 2: j in [4], size 1
    # Similarly for k_block
    
    # For each (j_block, k_block):
    # If both size 2: use Strassen (rank 7)
    # If either size 1: use direct (rank 2)
    
    total_rank = 0
    all_U = []
    all_V = []
    all_W = []
    
    j_offsets = [0, 2, 4]  # start index for each j_block
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4]  # start index for each k_block
    k_sizes = [2, 2, 1]
    
    for jb in range(3):
        for kb in range(3):
            js = j_sizes[jb]
            ks = k_sizes[kb]
            jo = j_offsets[jb]
            ko = k_offsets[kb]
            
            if js == 2 and ks == 2:
                U_blk, V_blk, W_blk = strassen_2x2x2()
                # U_blk: 4x7, where row = i*2 + j_local, i in [0,1], j_local in [0,1]
                # V_blk: 4x7, where row = j_local*2 + k_local, j_local in [0,1], k_local in [0,1]
                # W_blk: 4x7, where row = k_local*2 + i, k_local in [0,1], i in [0,1]
                
                # Map to global indices
                # U global row = i * m + j_global = i * 5 + (jo + j_local)
                # V global row = j_global * p + k_global = (jo + j_local) * 5 + (ko + k_local)
                # W global row = k_global * n + i = (ko + k_local) * 2 + i
                
                for r in range(4):
                    U_global_row = (r // 2) * 5 + (jo + (r % 2))
                    V_global_row = (jo + (r // 2)) * 5 + (ko + (r % 2))
                    W_global_row = (ko + (r % 2)) * 2 + (r // 2)
                    
                    for c in range(7):
                        if U_blk[r][c] != 0:
                            all_U.append([U_global_row, total_rank + c, U_blk[r][c]])
                        if V_blk[r][c] != 0:
                            all_V.append([V_global_row, total_rank + c, V_blk[r][c]])
                        if W_blk[r][c] != 0:
                            all_W.append([W_global_row, total_rank + c, W_blk[r][c]])
                
                total_rank += 7
            
            elif js == 2 and ks == 1:
                # A block is 2x2, B block is 2x1
                # This is 2x2x1: rank 4 (direct)
                # Actually, let's use direct multiplication
                # C is 2x1
                # c_i = sum_j A[i,j] * B[j,0]
                # Use rank 4: one term per (i,j)
                rank = 4
                for i in range(2):
                    for j_local in range(2):
                        c_idx = total_rank + i * 2 + j_local
                        j_global = jo + j_local
                        # U[i*5 + j_global, c_idx] = 1
                        # V[j_global*5 + ko, c_idx] = 1
                        # W[ko*2 + i, c_idx] = 1
                        all_U.append([(i * 5 + j_global), c_idx, 1])
                        all_V.append([(j_global * 5 + ko), c_idx, 1])
                        all_W.append([(ko * 2 + i), c_idx, 1])
                total_rank += rank
                
            elif js == 1 and ks == 2:
                # A block is 2x1, B block is 1x2
                # This is 2x1x2: rank 4
                rank = 4
                for i in range(2):
                    for k_local in range(2):
                        c_idx = total_rank + i * 2 + k_local
                        j_global = jo
                        k_global = ko + k_local
                        all_U.append([(i * 5 + j_global), c_idx, 1])
                        all_V.append([(j_global * 5 + k_global), c_idx, 1])
                        all_W.append([(k_global * 2 + i), c_idx, 1])
                total_rank += rank
                
            elif js == 1 and ks == 1:
                # A block is 2x1, B block is 1x1
                # This is 2x1x1: rank 2
                U_blk, V_blk, W_blk = direct_2x1x1()
                for r in range(2):
                    U_global_row = r * 5 + jo
                    V_global_row = jo * 5 + ko
                    W_global_row = ko * 2 + r
                    
                    for c in range(2):
                        if U_blk[r][c] != 0:
                            all_U.append([U_global_row, total_rank + c, U_blk[r][c]])
                        if V_blk[0][c] != 0:
                            all_V.append([V_global_row, total_rank + c, V_blk[0][c]])
                        if W_blk[r][c] != 0:
                            all_W.append([W_global_row, total_rank + c, W_blk[r][c]])
                
                total_rank += 2
    
    # Build final matrices
    U = [[0]*total_rank for _ in range(10)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(10)]
    
    for row, col, val in all_U:
        U[row][col] = val
    for row, col, val in all_V:
        V[row][col] = val
    for row, col, val in all_W:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

