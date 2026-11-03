import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    def strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = 1
        
        # M3 = A11(B21-B22)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = A22(B12-B11)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1; W[1][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[2][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    def schoolbook_221():
        """Return U, V, W for 2x2x1 schoolbook decomposition (rank 4)"""
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W
    
    total_rank = 50
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    for mid_block in range(2):
        for out_block in range(2):
            if out_block == 0:
                for sub2 in range(2):
                    U_sub, V_sub, W_sub = strassen_222()
                    for r in range(4):
                        i_loc = r // 2
                        j_loc = r % 2
                        U_global_row = i_loc * m + (2 * mid_block + j_loc)
                        for c in range(7):
                            U[U_global_row][rank_offset + c] = U_sub[r][c]
                    
                    for r in range(4):
                        j_loc = r // 2
                        k_loc = r % 2
                        V_global_row = (2 * mid_block + j_loc) * p + (4 * sub2 + k_loc)
                        for c in range(7):
                            V[V_global_row][rank_offset + c] = V_sub[r][c]
                    
                    for r in range(4):
                        k_loc = r // 2
                        i_loc = r % 2
                        W_global_row = (4 * sub2 + k_loc) * n + i_loc
                        for c in range(7):
                            W[W_global_row][rank_offset + c] = W_sub[r][c]
                    
                    rank_offset += 7
            else:
                # 2x2x2 for outputs 4,5
                U_sub, V_sub, W_sub = strassen_222()
                for r in range(4):
                    i_loc = r // 2
                    j_loc = r % 2
                    U_global_row = i_loc * m + (2 * mid_block + j_loc)
                    for c in range(7):
                        U[U_global_row][rank_offset + c] = U_sub[r][c]
                
                for r in range(4):
                    j_loc = r // 2
                    k_loc = r % 2
                    V_global_row = (2 * mid_block + j_loc) * p + (4 + k_loc)
                    for c in range(7):
                        V[V_global_row][rank_offset + c] = V_sub[r][c]
                
                for r in range(4):
                    k_loc = r // 2
                    i_loc = r % 2
                    W_global_row = (4 + k_loc) * n + i_loc
                    for c in range(7):
                        W[W_global_row][rank_offset + c] = W_sub[r][c]
                
                rank_offset += 7
                
                # 2x2x1 for output 6
                U_sub, V_sub, W_sub = schoolbook_221()
                for r in range(4):
                    i_loc = r // 2
                    j_loc = r % 2
                    U_global_row = i_loc * m + (2 * mid_block + j_loc)
                    for c in range(4):
                        U[U_global_row][rank_offset + c] = U_sub[r][c]
                
                for r in range(2):
                    j_loc = r
                    V_global_row = (2 * mid_block + j_loc) * p + 6
                    for c in range(4):
                        V[V_global_row][rank_offset + c] = V_sub[r][c]
                
                for r in range(2):
                    i_loc = r
                    W_global_row = 6 * n + i_loc
                    for c in range(4):
                        W[W_global_row][rank_offset + c] = W_sub[r][c]
                
                rank_offset += 4
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < 140:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                    if abs(val - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
