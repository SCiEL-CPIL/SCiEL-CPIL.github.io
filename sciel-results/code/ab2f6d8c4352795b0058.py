"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Build the rank-44 decomposition using Strassen for 2x2x2 blocks
    # A is 3x4, B is 4x4, result is 3x4
    # Split A into [A1|A2] (each 3x2), B into [[B11|B12],[B21|B22]] (each 2x2)
    # Z1 = A1*B11 + A2*B21 (each 3x2x2, rank 11)
    # Z2 = A1*B12 + A2*B22 (each 3x2x2, rank 11)
    # Total rank = 44
    
    # Strassen rank-7 decomposition of 2x2x2 MM tensor
    # u indexed by (i,j): u[0]=(0,0), u[1]=(0,1), u[2]=(1,0), u[3]=(1,1)
    # v indexed by (j,k): v[0]=(0,0), v[1]=(0,1), v[2]=(1,0), v[3]=(1,1)
    # w indexed by (k,i): w[0]=(0,0), w[1]=(0,1), w[2]=(1,0), w[3]=(1,1)
    
    strassen_u = [
        [1, 0, 0, -1],  # (a-d)
        [1, 0, 0, 1],   # (a+d)
        [0, 0, 0, 1],   # d
        [1, 0, 0, 0],   # a
        [1, 1, 0, 0],   # (a+b)
        [0, 0, 1, -1],  # (c-d)
        [0, 1, -1, 0],  # (b-c)
    ]
    strassen_v = [
        [0, 0, 1, 1],   # (g+h)
        [1, 0, 0, 0],   # e
        [-1, 1, 0, 0],  # (f-e)
        [1, 0, 0, -1],  # (e-h)
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # (e+f)
        [0, 0, 1, 1],   # (g+h)
    ]
    strassen_w = [
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
        [1, 1, 0, 0],   # C[0,0]+C[0,1]
        [0, 0, 1, 1],   # C[1,0]+C[1,1]
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
        [1, 0, 1, 0],   # C[0,0]+C[1,0]
        [0, 0, 1, 1],   # C[1,0]+C[1,1]
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
    ]
    
    def verify_2x2x2(u_list, v_list, w_list):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for c in range(7):
                        ui = u_list[c][i*2+j]
                        vj = v_list[c][j*2+k]
                        wk = w_list[c][k*2+i]
                        total += ui * vj * wk
                    if total != 1:
                        return False
        return True
    
    strassen_valid = verify_2x2x2(strassen_u, strassen_v, strassen_w)
    
    if strassen_valid and (time.time() - start_time) < 100:
        # Build rank-44 decomposition
        rank = 44
        U = [[0] * rank for _ in range(n * m)]  # 12 x 44
        V = [[0] * rank for _ in range(m * p)]  # 16 x 44
        W = [[0] * rank for _ in range(p * n)]  # 12 x 44
        
        # 4 blocks of 3x2x2, each rank 11 (7 Strassen + 4 trivial)
        # Block (ai, bi) where ai in {0,1} (which half of A's columns), bi in {0,1} (which half of B's columns)
        # A1 = columns 0,1 of A; A2 = columns 2,3 of A
        # B11 = rows 0,1 cols 0,1; B12 = rows 0,1 cols 2,3
        # B21 = rows 2,3 cols 0,1; B22 = rows 2,3 cols 2,3
        
        for ai in range(2):
            for bi in range(2):
                base_col = (ai * 2 + bi) * 11  # 11 columns per block
                # Strassen part (top 2 rows of A_part)
                for c in range(7):
                    for i in range(2):  # top 2 rows
                        for j in range(2):
                            u_val = strassen_u[c][i * 2 + j]
                            if u_val != 0:
                                U[i * 4 + ai * 2 + j][base_col + c] = u_val
                    for j in range(2):
                        for k in range(2):
                            v_val = strassen_v[c][j * 2 + k]
                            if v_val != 0:
                                V[(bi * 2 + j) * 4 + bi * 2 + k][base_col + c] = v_val
                    for k in range(2):
                        for i in range(2):
                            w_val = strassen_w[c][k * 2 + i]
                            if w_val != 0:
                                W[(bi * 2 + k) * 3 + i][base_col + c] = w_val
                
                # Trivial part (bottom 1 row of A_part)
                for j in range(2):
                    for k in range(2):
                        col = base_col + 7 + j * 2 + k
                        U[2 * 4 + ai * 2 + j][col] = 1  # i=2
                        V[(bi * 2 + j) * 4 + bi * 2 + k][col] = 1
                        W[(bi * 2 + k) * 3 + 2][col] = 1
        
        # Verify full decomposition
        def verify_full(U_mat, V_mat, W_mat, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(len(U_mat[0])):
                            total += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                        if total != 1:
                            return False
            return True
        
        if verify_full(U, V, W, n, m, p):
            # Try to reduce rank by removing redundant terms
            # Check if any term can be removed
            best_U, best_V, best_W = U, V, W
            best_rank = 44
            
            # Try removing each term and checking if the decomposition still holds
            # This is a greedy approach - try removing terms one at a time
            for attempt in range(10):
                removed = False
                for c in range(best_rank):
                    if time.time() > deadline - 10:
                        break
                    
                    # Create a copy with term c zeroed out
                    test_U = [row[:] for row in best_U]
                    test_V = [row[:] for row in best_V]
                    test_W = [row[:] for row in best_W]
                    
                    # Zero out column c
                    for i in range(n * m):
                        test_U[i][c] = 0
                    for j in range(m * p):
                        test_V[j][c] = 0
                    for k in range(p * n):
                        test_W[k][c] = 0
                    
                    # Check if decomposition still holds
                    valid = True
                    for i in range(n):
                        if not valid:
                            break
                        for j in range(m):
                            if not valid:
                                break
                            for k in range(p):
                                total = 0
                                for cc in range(best_rank):
                                    total += test_U[i * m + j][cc] * test_V[j * p + k][cc] * test_W[k * n + i][cc]
                                if total != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            break
                    
                    if valid:
                        # Remove term c
                        best_U = [row[:c] + row[c+1:] for row in test_U]
                        best_V = [row[:c] + row[c+1:] for row in test_V]
                        best_W = [row[:c] + row[c+1:] for row in test_W]
                        best_rank -= 1
                        removed = True
                        break
                
                if not removed:
                    break
            
            if best_rank < 44:
                return {"U": best_U, "V": best_V, "W": best_W}
            
            return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial rank-48 decomposition
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
