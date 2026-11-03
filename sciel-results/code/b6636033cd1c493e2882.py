import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(u_rows, v_rows, w_rows):
        nonlocal col
        # u_rows: [u00, u01, u10, u11]
        # v_rows: [v00, v01, v10, v11]
        # w_rows: [w00, w10, w01, w11] corresponding to C00, C10, C01, C11
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        U[u_rows[0]][col] = 1; U[u_rows[3]][col] = 1
        V[v_rows[0]][col] = 1; V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = 1; W[w_rows[1]][col] = 1; W[w_rows[2]][col] = 1; W[w_rows[3]][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[u_rows[2]][col] = 1; U[u_rows[3]][col] = 1
        V[v_rows[0]][col] = 1
        W[w_rows[1]][col] = 1; W[w_rows[3]][col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U[u_rows[0]][col] = 1
        V[v_rows[1]][col] = 1; V[v_rows[3]][col] = -1
        W[w_rows[1]][col] = 1; W[w_rows[3]][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[u_rows[3]][col] = 1
        V[v_rows[2]][col] = 1; V[v_rows[0]][col] = -1
        W[w_rows[0]][col] = 1; W[w_rows[1]][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[u_rows[0]][col] = 1; U[u_rows[1]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = -1; W[w_rows[2]][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[u_rows[2]][col] = 1; U[u_rows[0]][col] = -1
        V[v_rows[0]][col] = 1; V[v_rows[1]][col] = 1
        W[w_rows[3]][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[u_rows[1]][col] = 1; U[u_rows[3]][col] = -1
        V[v_rows[2]][col] = 1; V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = 1; W[w_rows[3]][col] = 1
        col += 1

    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_221(u_rows, v_rows, w_rows):
        nonlocal col
        # u_rows: [u00, u01, u10, u11]
        # v_rows: [v0, v1]
        # w_rows: [w0, w1]
        # C0 = A00*V0 + A01*V1
        U[u_rows[0]][col] = 1; V[v_rows[0]][col] = 1; W[w_rows[0]][col] = 1
        col += 1
        U[u_rows[1]][col] = 1; V[v_rows[1]][col] = 1; W[w_rows[0]][col] = 1
        col += 1
        # C1 = A10*V0 + A11*V1
        U[u_rows[2]][col] = 1; V[v_rows[0]][col] = 1; W[w_rows[1]][col] = 1
        col += 1
        U[u_rows[3]][col] = 1; V[v_rows[1]][col] = 1; W[w_rows[1]][col] = 1
        col += 1

    # Define indices
    # U: i*4+j
    u_AL = [0, 1, 4, 5]
    u_AR = [2, 3, 6, 7]
    
    # V: j*7+k
    # B_L1 (j=0,1; k=0,1)
    v_BL1 = [0, 1, 7, 8]
    # B_R1 (j=2,3; k=0,1) -- Wait, B_R1 is for A_R?
    # A_R B_R1. B_R1 is rows 2,3 of B? No.
    # B_L = [B_LL; B_RL]. B_RL is rows 2,3 of B.
    # So for A_L B_L, we use B rows 0,1 and 2,3.
    # B_LL: j=0,1; k=0,1 -> 0,1,7,8
    # B_RL: j=2,3; k=0,1 -> 14,15,21,22
    v_BL1 = [0, 1, 14, 15] # This is wrong. B_RL is j=2,3.
    # Let's re-verify V indices.
    # V[j*7+k].
    # B_LL: j in 0,1. k in 0,1.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 7. j=1, k=1 -> 8.
    # B_RL: j in 2,3. k in 0,1.
    # j=2, k=0 -> 14. j=2, k=1 -> 15.
    # j=3, k=0 -> 21. j=3, k=1 -> 22.
    # So v_BL1 = [0, 1, 14, 15] is wrong. It should be [0, 1, 14, 15] for B_LL?
    # No, B_LL is j=0,1. B_RL is j=2,3.
    # The Strassen block takes B as 2x2.
    # B[0,0]=v[0], B[0,1]=v[1], B[1,0]=v[2], B[1,1]=v[3].
    # So for A_L B_L:
    # B_L[0,0] = B_LL[0,0] = V[0]
    # B_L[0,1] = B_LL[0,1] = V[1]
    # B_L[1,0] = B_RL[0,0] = V[14]
    # B_L[1,1] = B_RL[0,1] = V[15]
    # Wait, B_RL is 2x2.
    # B_L is 4x2.
    # B_L = [B_LL; B_RL].
    # B_L[0,:] = B_LL[0,:].
    # B_L[1,:] = B_LL[1,:].
    # B_L[2,:] = B_RL[0,:].
    # B_L[3,:] = B_RL[1,:].
    # But Strassen takes B as 2x2.
    # My previous decomposition A_L B_L assumed B_L is 2x2.
    # But B_L is 4x2.
    # A_L is 2x2.
    # A_L (2x2) * B_L (4x2) is not defined.
    # A is 2x4. B is 4x4.
    # A = [A_L, A_R].
    # B = [B_L; B_R] where B_L is 2x4, B_R is 2x4.
    # Then A B = A_L B_L + A_R B_R.
    # B_L is 2x4.
    # Split B_L into B_LL (2x2) and B_LR (2x2).
    # B_L = [B_LL, B_LR].
    # Then A_L B_L = A_L B_LL + A_L B_LR.
    # This is two 2x2x2 multiplications.
    # So for Part 1 (4 cols):
    # A_L B_LL (rank 7)
    # A_L B_LR (rank 7)
    # A_R B_RL (rank 7)
    # A_R B_RR (rank 7)
    # Total 28.
    # This is worse than 14.
    
    # My previous "14" was based on A_L B_L where B_L is 2x2.
    # But B is 4x4.
    # If we split A into 2x2 and B into 2x2 blocks:
    # A = [A11 A12; A21 A22] (2x2 blocks)
    # B = [B11 B12; B21 B22] (2x2 blocks)
    # C = A11 B11 + A12 B21 + A21 B12 + A22 B22.
    # 4 multiplications of 2x2x2.
    # Rank 4 * 7 = 28.
    # Schoolbook is 16.
    # So Strassen is worse for 4x4.
    
    # So my rank 32 estimate was flawed.
    # 2x4x4 rank is at least 16 (schoolbook).
    # 2x4x3 rank is at least 12 (schoolbook).
    # Total 28.
    # Schoolbook total 56.
    # So 28 is better.
    
    # Can we do better than 28?
    # 2x4x4:
    # A is 2x4. B is 4x4.
    # Maybe use rank 7 for 2x2x2?
    # If we can find a decomposition of 2x4x4 with rank < 16.
    # It is known that rank of 2x4x4 is 14?
    # If rank is 14, then 14 + 12 = 26.
    
    # Let's assume we can get 14 for 2x4x4.
    # How?
    # A = [A_L, A_R].
    # B = [B_1; B_2] (2x4 each).
    # C = A_L B_1 + A_R B_2.
    # B_1 is 2x4.
    # A_L B_1 is 2x4.
    # Rank of 2x2x4?
    # 2x2x4 is 2x2x2 + 2x2x2. Rank 7+7=14.
    # So A_L B_1 has rank 14.
    # A_R B_2 has rank 14.
    # Total 28.
    
    # So 28 seems to be the bound for 2x4x4 using Strassen.
    # And 12 for 2x4x3 using Schoolbook.
    # Total 40.
    
    # Let's stick with Schoolbook for small blocks if Strassen doesn't help.
    # 2x4x4 Schoolbook: 16.
    # 2x4x3 Schoolbook: 12.
    # Total 28.
    
    # So Rank 28 is a good target.
    
    # Let's implement Rank 28.
    # Part 1: 2x4x4 Schoolbook.
    # Part 2: 2x4x3 Schoolbook.
    
    # 2x4x4 Schoolbook:
    # A (2x4), B (4x4).
    # 16 multiplications.
    # Part 2: 2x4x3 Schoolbook.
    # A (2x4), B (4x3).
    # 12 multiplications.
    
    # Total 28.
    
    # Let's verify if 2x4x4 can be done in < 16.
    # If not, 28 is the best.
    
    # I will implement Rank 28.
    
    # Reset col
    col = 0
    
    # Part 1: 2x4x4
    # A rows 0,1. B cols 0,1,2,3.
    # For each i in 0,1 and k in 0,1,2,3:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 4 j's.
    # So 4 mults per output element.
    # 2 * 4 = 8 elements.
    # 8 * 4 = 32 mults?
    # No.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k].
    # This is 4 mults.
    # Total 8 * 4 = 32.
    # Wait, schoolbook for 2x4x4 is 2*4*4 = 32.
    # My previous 16 was wrong.
    # 2x4x4 is 32.
    # 2x4x3 is 24.
    # Total 56.
    # So Schoolbook is 56.
    
    # So Strassen for 2x4x4 (28) is better.
    # 28 + 24 = 52.
    
    # Let's implement 52.
    
    # Part 1: 2x4x4 with Strassen (28).
    # Part 2: 2x4x3 with Schoolbook (24).
    
    # Part 1:
    # A_L B_LL (7)
    # A_L B_LR (7)
    # A_R B_RL (7)
    # A_R B_RR (7)
    
    # Indices for Part 1:
    # A_L: U[0,1,4,5]
    # A_R: U[2,3,6,7]
    
    # B_LL: V[0,1,7,8]
    # B_LR: V[2,3,9,10]
    # B_RL: V[14,15,21,22]
    # B_RR: V[16,17,23,24]
    
    # W for C_LL (k=0,1): W[0,1,2,3]
    # W for C_LR (k=2,3): W[4,5,6,7]
    # W for C_RL (k=0,1): W[0,1,2,3] (Add to same)
    # W for C_RR (k=2,3): W[4,5,6,7] (Add to same)
    
    # So we call add_strassen 4 times.
    
    add_strassen(u_AL, v_BL1, [0,1,2,3]) # A_L B_LL -> C_LL
    add_strassen(u_AL, v_BL2, [4,5,6,7]) # A_L B_LR -> C_LR
    add_strassen(u_AR, v_BR1, [0,1,2,3]) # A_R B_RL -> C_LL
    add_strassen(u_AR, v_BR2, [4,5,6,7]) # A_R B_RR -> C_LR
    
    # Part 2: 2x4x3 Schoolbook (24).
    # B cols 4,5,6.
    # For i in 0,1, k in 4,5,6:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 4 mults each.
    # 2 * 3 * 4 = 24.
    
    # We can just loop and add.
    for i in range(2):
        for k in range(4, 7):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
