import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 96
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Attempt to construct the Rank 84 solution based on Block Strassen
    # Decompose into 6 output blocks (2x3 grid), each computed as sum of two 2x2x2 products
    # Each 2x2x2 product uses Strassen (rank 7). Total rank 12 * 7 = 84.
    
    try:
        rank_84 = 84
        U_84 = [[0.0] * rank_84 for _ in range(n * m)]
        V_84 = [[0.0] * rank_84 for _ in range(m * p)]
        W_84 = [[0.0] * rank_84 for _ in range(p * n)]
        
        col = 0
        for i_block in range(2):  # i index of 2x2 block in A and C
            for j_block in range(3):  # j index of 2x2 block in B and C
                
                # Global row indices for A block i_block: rows 2*i_block, 2*i_block+1
                # Global col indices for B block j_block: cols 2*j_block, 2*j_block+1
                # Global row indices for C block: rows 2*i_block, 2*i_block+1
                # Global col indices for C block: cols 2*j_block, 2*j_block+1
                
                a_row0 = 2 * i_block
                a_row1 = 2 * i_block + 1
                b_col0 = 2 * j_block
                b_col1 = 2 * j_block + 1
                c_row0 = 2 * i_block
                c_row1 = 2 * i_block + 1
                c_col0 = 2 * j_block
                c_col1 = 2 * j_block + 1
                
                # Term 1: A_i0 * B_0j
                # A_i0: rows a_row0..a_row1, cols 0..1
                # B_0j: rows 0..1, cols b_col0..b_col1
                # C_ij: rows c_row0..c_row1, cols c_col0..c_col1
                
                # Strassen M1 = (a00+a11)(b00+b11)
                U_84[a_row0][col] += 1  # a00
                U_84[a_row1][col] += 1  # a11
                V_84[0][col] += 1       # b00 (row 0, col b_col0)
                V_84[1][col] += 1       # b11 (row 1, col b_col1)
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M2 = (a10+a11)b00
                U_84[a_row1][col] += 1  # a10 (row a_row1, col 0)
                U_84[a_row1][col] += 1  # a11 (row a_row1, col 1)
                V_84[0][col] += 1       # b00
                W_84[c_row1 * p + c_col0][col] += 1  # C10
                col += 1
                
                # Strassen M3 = a00(b10-b11)
                U_84[a_row0][col] += 1  # a00
                V_84[1][col] += 1       # b10 (row 1, col b_col0)
                V_84[1][col] -= 1       # b11 (row 1, col b_col1)
                W_84[c_row0 * p + c_col1][col] += 1  # C01
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M4 = a11(b01-b00)
                U_84[a_row1][col] += 1  # a11
                V_84[0][col] += 1       # b01 (row 0, col b_col1)
                V_84[0][col] -= 1       # b00 (row 0, col b_col0)
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col0][col] += 1  # C10
                col += 1
                
                # Strassen M5 = (a00+a01)b11
                U_84[a_row0][col] += 1  # a00
                U_84[a_row0][col] += 1  # a01 (row a_row0, col 1)
                V_84[1][col] += 1       # b11
                W_84[c_row0 * p + c_col0][col] -= 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C01
                col += 1
                
                # Strassen M6 = (a10-a00)(b00+b01)
                U_84[a_row1][col] += 1  # a10
                U_84[a_row0][col] -= 1  # a00
                V_84[0][col] += 1       # b00
                V_84[0][col] += 1       # b01
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M7 = (a01-a11)(b10+b11)
                U_84[a_row0][col] += 1  # a01
                U_84[a_row1][col] -= 1  # a11
                V_84[1][col] += 1       # b10
                V_84[1][col] += 1       # b11
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Term 2: A_i1 * B_1j
                # A_i1: rows a_row0..a_row1, cols 2..3
                # B_1j: rows 2..3, cols b_col0..b_col1
                
                # Strassen M1' = (a00'+a11')(b00'+b11')
                U_84[a_row0][col] += 1  # a00' (row a_row0, col 2)
                U_84[a_row1][col] += 1  # a11' (row a_row1, col 3)
                V_84[2][col] += 1       # b00' (row 2, col b_col0)
                V_84[3][col] += 1       # b11' (row 3, col b_col1)
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M2' = (a10'+a11')b00'
                U_84[a_row1][col] += 1  # a10' (row a_row1, col 2)
                U_84[a_row1][col] += 1  # a11' (row a_row1, col 3)
                V_84[2][col] += 1       # b00'
                W_84[c_row1 * p + c_col0][col] += 1  # C10
                col += 1
                
                # Strassen M3' = a00'(b10'-b11')
                U_84[a_row0][col] += 1  # a00'
                V_84[3][col] += 1       # b10' (row 3, col b_col0)
                V_84[3][col] -= 1       # b11' (row 3, col b_col1)
                W_84[c_row0 * p + c_col1][col] += 1  # C01
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M4' = a11'(b01'-b00')
                U_84[a_row1][col] += 1  # a11'
                V_84[2][col] += 1       # b01' (row 2, col b_col1)
                V_84[2][col] -= 1       # b00' (row 2, col b_col0)
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col0][col] += 1  # C10
                col += 1
                
                # Strassen M5' = (a00'+a01')b11'
                U_84[a_row0][col] += 1  # a00'
                U_84[a_row0][col] += 1  # a01' (row a_row0, col 3)
                V_84[3][col] += 1       # b11'
                W_84[c_row0 * p + c_col0][col] -= 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C01
                col += 1
                
                # Strassen M6' = (a10'-a00')(b00'+b01')
                U_84[a_row1][col] += 1  # a10'
                U_84[a_row0][col] -= 1  # a00'
                V_84[2][col] += 1       # b00'
                V_84[2][col] += 1       # b01'
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
                
                # Strassen M7' = (a01'-a11')(b10'+b11')
                U_84[a_row0][col] += 1  # a01'
                U_84[a_row1][col] -= 1  # a11'
                V_84[3][col] += 1       # b10'
                V_84[3][col] += 1       # b11'
                W_84[c_row0 * p + c_col0][col] += 1  # C00
                W_84[c_row1 * p + c_col1][col] += 1  # C11
                col += 1
        
        # Verify the rank 84 solution
        if col == 84:
            U_np = np.array(U_84)
            V_np = np.array(V_84)
            W_np = np.array(W_84)
            
            valid = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for l in range(m):
                            val = 0.0
                            for r in range(84):
                                val += W_np[k * n + i, r] * U_np[i * m + j, r] * V_np[l * p + k, r]
                            expected = 1.0 if j == l else 0.0
                            if abs(val - expected) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_solution = {
                    "U": [[float(v) for v in row] for row in U_84],
                    "V": [[float(v) for v in row] for row in V_84],
                    "W": [[float(v) for v in row] for row in W_84]
                }
                best_rank = 84
    except Exception as e:
        pass

    return best_solution
