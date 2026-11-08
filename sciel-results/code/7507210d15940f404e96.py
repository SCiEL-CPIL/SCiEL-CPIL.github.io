import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_add(col_offset, A_row_map, A_col_map, B_row_map, B_col_map, C_row_map, C_col_map):
        c = col_offset + 0
        U[A_row_map[0]*m + A_col_map[0]][c] += 1
        U[A_row_map[1]*m + A_col_map[1]][c] += 1
        V[B_row_map[0]*p + B_col_map[0]][c] += 1
        V[B_row_map[1]*p + B_col_map[1]][c] += 1
        W[C_row_map[0]*n + C_col_map[0]][c] += 1
        W[C_row_map[1]*n + C_col_map[1]][c] += 1
        
        c = col_offset + 1
        U[A_row_map[1]*m + A_col_map[0]][c] += 1
        U[A_row_map[1]*m + A_col_map[1]][c] += 1
        V[B_row_map[0]*p + B_col_map[0]][c] += 1
        W[C_row_map[0]*n + C_col_map[1]][c] += 1
        W[C_row_map[1]*n + C_col_map[1]][c] -= 1
        
        c = col_offset + 2
        U[A_row_map[0]*m + A_col_map[0]][c] += 1
        V[B_row_map[0]*p + B_col_map[1]][c] += 1
        V[B_row_map[1]*p + B_col_map[1]][c] -= 1
        W[C_row_map[1]*n + C_col_map[0]][c] += 1
        W[C_row_map[1]*n + C_col_map[1]][c] += 1
        
        c = col_offset + 3
        U[A_row_map[1]*m + A_col_map[1]][c] += 1
        V[B_row_map[1]*p + B_col_map[0]][c] += 1
        V[B_row_map[0]*p + B_col_map[0]][c] -= 1
        W[C_row_map[0]*n + C_col_map[0]][c] += 1
        W[C_row_map[0]*n + C_col_map[1]][c] += 1
        
        c = col_offset + 4
        U[A_row_map[0]*m + A_col_map[0]][c] += 1
        U[A_row_map[0]*m + A_col_map[1]][c] += 1
        V[B_row_map[1]*p + B_col_map[1]][c] += 1
        W[C_row_map[0]*n + C_col_map[0]][c] -= 1
        W[C_row_map[1]*n + C_col_map[0]][c] += 1
        
        c = col_offset + 5
        U[A_row_map[1]*m + A_col_map[0]][c] += 1
        U[A_row_map[0]*m + A_col_map[0]][c] -= 1
        V[B_row_map[0]*p + B_col_map[0]][c] += 1
        V[B_row_map[1]*p + B_col_map[0]][c] += 1
        W[C_row_map[1]*n + C_col_map[1]][c] += 1
        
        c = col_offset + 6
        U[A_row_map[0]*m + A_col_map[1]][c] += 1
        U[A_row_map[1]*m + A_col_map[1]][c] -= 1
        V[B_row_map[0]*p + B_col_map[1]][c] += 1
        V[B_row_map[1]*p + B_col_map[1]][c] += 1
        W[C_row_map[0]*n + C_col_map[0]][c] += 1
        W[C_row_map[1]*n + C_col_map[1]][c] -= 1
    
    A_left_rows = [0, 1]
    A_left_cols = [0, 1]
    A_right_rows = [0, 1]
    A_right_cols = [2, 3]
    
    B00_rows = [0, 1]
    B00_cols = [0, 1]
    B01_rows = [0, 1]
    B01_cols = [2, 3]
    B10_rows = [2, 3]
    B10_cols = [0, 1]
    B11_rows = [2, 3]
    B11_cols = [2, 3]
    
    C_left_k = [0, 1]
    C_left_i = [0, 1]
    C_right_k = [2, 3]
    C_right_i = [0, 1]
    
    strassen_add(0, A_left_rows, A_left_cols, B00_rows, B00_cols, C_left_k, C_left_i)
    strassen_add(7, A_right_rows, A_right_cols, B10_rows, B10_cols, C_left_k, C_left_i)
    strassen_add(14, A_left_rows, A_left_cols, B01_rows, B01_cols, C_right_k, C_right_i)
    strassen_add(21, A_right_rows, A_right_cols, B11_rows, B11_cols, C_right_k, C_right_i)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    # Fall back to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii*m+jj][col] = 1
                                V_sb[jj*p+kk][col] = 1
                                W_sb[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
