import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # 1. Strassen 4x4x4 Decomposition (Rank 49)
    # Base case: 2x2x2 Strassen (Rank 7)
    # U, V, W are 4x7
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # Indices: A[0]=A11, A[1]=A12, A[2]=A21, A[3]=A22
    # B[0]=B11, B[1]=B12, B[2]=B21, B[3]=B22
    # C[0]=C11, C[1]=C12, C[2]=C21, C[3]=C22
    
    # M1 = (A11+A22)(B11+B22)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[2, 3] = 1; W_222[3, 3] = -1
    
    # M5 = (A11+A12)B22
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[3, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[3, 6] = 1

    # Build 4x4x4 Strassen (Rank 49)
    rank_444 = 49
    U_444 = np.zeros((16, rank_444), dtype=int)
    V_444 = np.zeros((16, rank_444), dtype=int) # 4x4 = 16
    W_444 = np.zeros((16, rank_444), dtype=int) # 4x4 = 16
    
    for k1 in range(7):
        for k2 in range(7):
            idx = k1 * 7 + k2
            for block_a in range(4):
                for elem_a in range(4):
                    a_idx = block_a * 4 + elem_a
                    U_444[a_idx, idx] = U_222[block_a, k1] * U_222[elem_a, k2]
            for block_b in range(4):
                for elem_b in range(4):
                    b_idx = block_b * 4 + elem_b
                    V_444[b_idx, idx] = V_222[block_b, k1] * V_222[elem_b, k2]
            for block_c in range(4):
                for elem_c in range(4):
                    c_idx = block_c * 4 + elem_c
                    W_444[c_idx, idx] = W_222[block_c, k1] * W_222[elem_c, k2]

    # 2. Schoolbook 4x4x2 Decomposition (Rank 32)
    rank_442 = 32
    U_442 = np.zeros((16, rank_442), dtype=int)
    V_442 = np.zeros((8, rank_442), dtype=int) # 4x2 = 8
    W_442 = np.zeros((8, rank_442), dtype=int) # 2x4 = 8
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i * 4 + j, col] = 1
                V_442[j * 2 + k, col] = 1
                W_442[k * 4 + i, col] = 1
                col += 1

    # 3. Combine into 4x4x6 (Rank 81)
    total_rank = rank_444 + rank_442
    
    U_final = np.zeros((16, total_rank), dtype=int)
    V_final = np.zeros((24, total_rank), dtype=int) # 4x6 = 24
    W_final = np.zeros((24, total_rank), dtype=int) # 6x4 = 24
    
    # Copy 4x4x4 part
    U_final[:, 0:rank_444] = U_444
    V_final[0:16, 0:rank_444] = V_444
    W_final[0:16, 0:rank_444] = W_444
    
    # Copy 4x4x2 part
    # B indices: 4x4 part is rows 0-15 of V_final. 4x2 part is rows 16-23.
    # V_442 has shape (8, 32). We need to place it in rows 16-23 of V_final.
    U_final[:, rank_444:total_rank] = U_442
    V_final[16:24, rank_444:total_rank] = V_442
    W_final[16:24, rank_444:total_rank] = W_442
    
    # Convert to lists of lists as required by the interface
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

if __name__ == "__main__":
    construct()
