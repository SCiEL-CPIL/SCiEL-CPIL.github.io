import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # We will construct a rank 50 decomposition
    # Split p=7 into p1=4 and p2=3
    # C = A B1 + A B2
    # B1 is 4x4, B2 is 4x3
    # A is 2x4
    
    # For A B1 (2x4 * 4x4):
    # Split A into AL (2x2, cols 0-1) and AR (2x2, cols 2-3)
    # Split B1 into BL1 (2x4, rows 0-1) and BR1 (2x4, rows 2-3)
    # C1 = AL BL1 + AR BR1
    # Each term is 2x2 * 2x4.
    # Split 2x4 into two 2x2 blocks.
    # AL BL1 = AL BL1a + AL BL1b
    # AR BR1 = AR BR1a + AR BR1b
    # Each 2x2 * 2x2 uses Strassen (Rank 7).
    # Total 4 * 7 = 28 ranks.
    
    # For A B2 (2x4 * 4x3):
    # Split A into AL and AR.
    # Split B2 into BL2 (2x3, rows 0-1) and BR2 (2x3, rows 2-3)
    # C2 = AL BL2 + AR BR2
    # Each term is 2x2 * 2x3.
    # Split 2x3 into 2x2 and 2x1.
    # AL BL2 = AL BL2a (2x2*2x2) + AL BL2b (2x2*2x1)
    # AR BR2 = AR BR2a (2x2*2x2) + AR BR2b (2x2*2x1)
    # 2x2*2x2 uses Strassen (Rank 7).
    # 2x2*2x1 uses Schoolbook (Rank 4).
    # Total 2 * (7 + 4) = 22 ranks.
    
    # Total Rank = 28 + 22 = 50.
    
    rank_total = 50
    U = np.zeros((dim_u, rank_total))
    V = np.zeros((dim_v, rank_total))
    W = np.zeros((dim_w, rank_total))
    
    col = 0
    
    # Helper to add Strassen block
    # u_idx: list of 4 indices for A (00, 01, 10, 11)
    # v_idx: list of 4 indices for B (00, 01, 10, 11)
    # w_idx: list of 4 indices for C (00, 01, 10, 11)
    def add_strassen(u_idx, v_idx, w_idx, col):
        # M1 = (A00+A11)(B00+B11)
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        W[w_idx[1]][col] -= 1
        W[w_idx[2]][col] -= 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        W[w_idx[3]][col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[u_idx[0]][col] += 1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[u_idx[3]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[0]][col] -= 1
        W[w_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] -= 1
        V[v_idx[0]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B11+B01)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] -= 1
        V[v_idx[3]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[1]][col] -= 1
        col += 1
        
        return col

    # Helper to add Schoolbook 2x2x1 block
    # u_idx: list of 4 indices for A (00, 01, 10, 11)
    # v_idx: list of 2 indices for B (0, 1)
    # w_idx: list of 2 indices for C (0, 1)
    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, col):
        # P1 = A00 B0
        U[u_idx[0]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # P2 = A01 B1
        U[u_idx[1]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # P3 = A10 B0
        U[u_idx[2]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # P4 = A11 B1
        U[u_idx[3]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        return col

    # Indices for U (2x4 matrix, row-major)
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    u_AL = [0, 1, 4, 5]
    u_AR = [2, 3, 6, 7]
    
    # Indices for V (4x7 matrix, row-major)
    # B[0,0]=0, B[0,1]=1, ..., B[0,6]=6
    # B[1,0]=7, ...
    # B[2,0]=14, ...
    # B[3,0]=21, ...
    
    # Indices for W (7x2 matrix, row-major)
    # C[0,0]=0, C[0,1]=1, ..., C[0,6]=12? No.
    # W[k*n+i, r]. n=2.
    # k=0: W[0,0]=0, W[1,0]=1
    # k=1: W[2,0]=2, W[3,0]=3
    # ...
    # k=6: W[12,0]=12, W[13,0]=13
    # Wait, the task says W[k*n+i, r].
    # So W index is k*n + i.
    # C is 2x7. C[i,k].
    # W index for C[i,k] is k*n + i = 2k + i.
    
    # Block 1: A_L B_L4a (2x2 * 2x2) -> C[0:2, 0:2]
    # A_L: u_AL
    # B_L4a: B[0:2, 0:2] -> v[0,1,7,8]
    # C: C[0:2, 0:2] -> w[0,1,2,3]
    v_B_L4a = [0, 1, 7, 8]
    w_C_0_0 = [0, 1, 2, 3]
    col = add_strassen(u_AL, v_B_L4a, w_C_0_0, col)
    
    # Block 2: A_L B_L4b (2x2 * 2x2) -> C[0:2, 2:4]
    # A_L: u_AL
    # B_L4b: B[0:2, 2:4] -> v[2,3,9,10]
    # C: C[0:2, 2:4] -> w[4,5,6,7]
    v_B_L4b = [2, 3, 9, 10]
    w_C_2_0 = [4, 5, 6, 7]
    col = add_strassen(u_AL, v_B_L4b, w_C_2_0, col)
    
    # Block 3: A_R B_R4a (2x2 * 2x2) -> C[0:2, 0:2]
    # A_R: u_AR
    # B_R4a: B[2:4, 0:2] -> v[14,15,21,22]
    # C: C[0:2, 0:2] -> w[0,1,2,3]
    v_B_R4a = [14, 15, 21, 22]
    col = add_strassen(u_AR, v_B_R4a, w_C_0_0, col)
    
    # Block 4: A_R B_R4b (2x2 * 2x2) -> C[0:2, 2:4]
    # A_R: u_AR
    # B_R4b: B[2:4, 2:4] -> v[16,17,23,24]
    # C: C[0:2, 2:4] -> w[4,5,6,7]
    v_B_R4b = [16, 17, 23, 24]
    col = add_strassen(u_AR, v_B_R4b, w_C_2_0, col)
    
    # Block 5: A_L B_L3a (2x2 * 2x2) -> C[0:2, 4:6]
    # A_L: u_AL
    # B_L3a: B[0:2, 4:6] -> v[4,5,11,12]
    # C: C[0:2, 4:6] -> w[8,9,10,11]
    v_B_L3a = [4, 5, 11, 12]
    w_C_4_0 = [8, 9, 10, 11]
    col = add_strassen(u_AL, v_B_L3a, w_C_4_0, col)
    
    # Block 6: A_L B_L3b (2x2 * 2x1) -> C[0:2, 6:7]
    # A_L: u_AL
    # B_L3b: B[0:2, 6:7] -> v[6,13]
    # C: C[0:2, 6:7] -> w[12,13]
    v_B_L3b = [6, 13]
    w_C_6_0 = [12, 13]
    col = add_schoolbook_2x2x1(u_AL, v_B_L3b, w_C_6_0, col)
    
    # Block 7: A_R B_R3a (2x2 * 2x2) -> C[0:2, 4:6]
    # A_R: u_AR
    # B_R3a: B[2:4, 4:6] -> v[18,19,25,26]
    # C: C[0:2, 4:6] -> w[8,9,10,11]
    v_B_R3a = [18, 19, 25, 26]
    col = add_strassen(u_AR, v_B_R3a, w_C_4_0, col)
    
    # Block 8: A_R B_R3b (2x2 * 2x1) -> C[0:2, 6:7]
    # A_R: u_AR
    # B_R3b: B[2:4, 6:7] -> v[20,27]
    # C: C[0:2, 6:7] -> w[12,13]
    v_B_R3b = [20, 27]
    col = add_schoolbook_2x2x1(u_AR, v_B_R3b, w_C_6_0, col)
    
    # Verify col == rank_total
    if col != rank_total:
        print(f"Warning: col {col} != rank_total {rank_total}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
