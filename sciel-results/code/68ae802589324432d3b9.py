import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Strassen factors for 2x2x2
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # U (A factors)
    # M1: A11+A22 -> [1, 0, 0, 1]
    # M2: A21+A22 -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: A11+A12 -> [1, 1, 0, 0]
    # M6: A21-A11 -> [-1, 0, 1, 0]
    # M7: A12-A22 -> [0, 1, 0, -1]
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    # V (B factors)
    # M1: B11+B22 -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B12-B22 -> [0, 1, 0, -1]
    # M4: B21-B11 -> [-1, 0, 1, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: B11+B12 -> [1, 1, 0, 0]
    # M7: B21+B22 -> [0, 0, 1, 1]
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    # W (C factors)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W[i, k] where i is C index, k is Strassen factor index
    # C0 (C11): [1, 0, 0, 1, -1, 0, 1]
    # C1 (C12): [0, 0, 1, 0, 1, 0, 0]
    # C2 (C21): [0, 1, 0, 1, 0, 0, 0]
    # C3 (C22): [1, -1, 1, 0, 0, 1, 0]
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Construct 2x4x8 decomposition
    # A is 2x4. U shape (8, Rank)
    # B is 4x8. V shape (32, Rank)
    # C is 2x8. W shape (16, Rank)
    # Trace convention: W[k*n + i, r]
    
    # We use 8 blocks of 2x2x2 Strassen.
    # Split A into A_L (cols 0,1) and A_R (cols 2,3).
    # Split B into 4 column blocks of width 2? No, B is 4x8.
    # Let's split B into 4 blocks of 2x2?
    # B rows 0,1 and 2,3.
    # B cols 0,1; 2,3; 4,5; 6,7.
    
    # For each column block c (0..3):
    #   We need to compute contribution to C[:, 4*c : 4*c+2]
    #   C[:, 4*c : 4*c+2] = A_L * B_L_c + A_R * B_R_c
    #   where B_L_c is B[0:2, 4*c:4*c+2] (2x2)
    #   and B_R_c is B[2:4, 4*c:4*c+2] (2x2)
    
    # Each term A_L * B_L_c is a 2x2x2 multiplication (Rank 7).
    # Each term A_R * B_R_c is a 2x2x2 multiplication (Rank 7).
    # Total rank per column block: 14.
    # Total rank: 4 * 14 = 56.
    
    total_rank = 56
    U_full = np.zeros((8, total_rank))
    V_full = np.zeros((32, total_rank))
    W_full = np.zeros((16, total_rank))
    
    for c in range(4):
        # Column block c
        # B columns are 4*c, 4*c+1
        # C columns are 4*c, 4*c+1
        
        # Block 1: A_L * B_L_c
        # A_L is A[:, 0:2]. Indices in U: 0,1,2,3 (row-major: 0,1 for row 0; 2,3 for row 1? No, A is 2x4. U index is i*m+j. m=4.
        # A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5.
        # Wait, U shape is (n*m, rank) = (8, rank).
        # Index for A[i,j] is i*m + j = i*4 + j.
        # A_L (cols 0,1):
        # A[0,0]=0, A[0,1]=1
        # A[1,0]=4, A[1,1]=5
        
        # B_L_c is B[0:2, 4*c:4*c+2].
        # B[0, 4*c], B[0, 4*c+1]
        # B[1, 4*c], B[1, 4*c+1]
        # V index for B[j,k] is j*p + k = j*8 + k.
        # B[0, 4*c] -> 4*c
        # B[0, 4*c+1] -> 4*c+1
        # B[1, 4*c] -> 8 + 4*c
        # B[1, 4*c+1] -> 8 + 4*c+1
        
        # C contribution to C[:, 4*c : 4*c+2].
        # C[0, 4*c], C[0, 4*c+1]
        # C[1, 4*c], C[1, 4*c+1]
        # W index for C[i,k] is k*n + i = k*2 + i.
        # C[0, 4*c] -> (4*c)*2 + 0 = 8*c
        # C[0, 4*c+1] -> (4*c+1)*2 + 0 = 8*c + 2
        # C[1, 4*c] -> (4*c)*2 + 1 = 8*c + 1
        # C[1, 4*c+1] -> (4*c+1)*2 + 1 = 8*c + 3
        
        # Strassen for A_L * B_L_c
        # Local indices for A_L: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
        # Local indices for B_L_c: 0->B[0,4c], 1->B[0,4c+1], 2->B[1,4c], 3->B[1,4c+1]
        # Local indices for C_part: 0->C[0,4c], 1->C[0,4c+1], 2->C[1,4c], 3->C[1,4c+1]
        
        # Map U_strassen to U_full
        # U_strassen[i, s] where i is local A index.
        # Local A index 0 -> Global U index 0
        # Local A index 1 -> Global U index 1
        # Local A index 2 -> Global U index 4
        # Local A index 3 -> Global U index 5
        
        u_map_1 = [0, 1, 4, 5]
        v_map_1 = [4*c, 4*c+1, 8+4*c, 8+4*c+1]
        w_map_1 = [8*c, 8*c+2, 8*c+1, 8*c+3]
        
        for s in range(7):
            col_idx = c * 14 + s
            for i_local in range(4):
                u_val = U_strassen[i_local, s]
                if u_val != 0:
                    U_full[u_map_1[i_local], col_idx] += u_val
            
            for j_local in range(4):
                v_val = V_strassen[j_local, s]
                if v_val != 0:
                    V_full[v_map_1[j_local], col_idx] += v_val
            
            for k_local in range(4):
                w_val = W_strassen[k_local, s]
                if w_val != 0:
                    W_full[w_map_1[k_local], col_idx] += w_val
        
        # Block 2: A_R * B_R_c
        # A_R is A[:, 2:4].
        # A[0,2]=2, A[0,3]=3
        # A[1,2]=6, A[1,3]=7
        u_map_2 = [2, 3, 6, 7]
        
        # B_R_c is B[2:4, 4*c:4*c+2].
        # B[2, 4*c], B[2, 4*c+1]
        # B[3, 4*c], B[3, 4*c+1]
        # V index:
        # B[2, 4*c] -> 16 + 4*c
        # B[2, 4*c+1] -> 16 + 4*c + 1
        # B[3, 4*c] -> 24 + 4*c
        # B[3, 4*c+1] -> 24 + 4*c + 1
        v_map_2 = [16+4*c, 16+4*c+1, 24+4*c, 24+4*c+1]
        
        # C contribution is the same positions as Block 1.
        w_map_2 = [8*c, 8*c+2, 8*c+1, 8*c+3]
        
        for s in range(7):
            col_idx = c * 14 + 7 + s
            for i_local in range(4):
                u_val = U_strassen[i_local, s]
                if u_val != 0:
                    U_full[u_map_2[i_local], col_idx] += u_val
            
            for j_local in range(4):
                v_val = V_strassen[j_local, s]
                if v_val != 0:
                    V_full[v_map_2[j_local], col_idx] += v_val
            
            for k_local in range(4):
                w_val = W_strassen[k_local, s]
                if w_val != 0:
                    W_full[w_map_2[k_local], col_idx] += w_val
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
