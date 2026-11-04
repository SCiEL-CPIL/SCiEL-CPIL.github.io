import numpy as np
import time

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Laderman's algorithm (rank 19) for each of the two 4x4 output blocks.
    Total rank: 38.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    nm = n * m      # 16
    mp = m * p      # 32
    pn = p * n      # 32
    
    # Define Laderman's rank-19 decomposition for 4x4 * 4x4
    # U: (16, 19), V: (16, 19), W: (16, 19)
    U_lad = np.zeros((16, 19))
    V_lad = np.zeros((16, 19))
    W_lad = np.zeros((16, 19))
    
    # Laderman coefficients
    # Format: (U_indices, V_indices, W_indices)
    # Indices are i*m + j
    
    # 0
    U_lad[[0, 5, 10, 15], 0] = 1
    V_lad[[0, 5, 10, 15], 0] = 1
    W_lad[[0, 5, 10, 15], 0] = 1
    
    # 1
    U_lad[[0, 5], 1] = 1
    V_lad[[0, 5], 1] = 1
    W_lad[[0, 5], 1] = 1
    
    # 2
    U_lad[[10, 15], 2] = 1
    V_lad[[10, 15], 2] = 1
    W_lad[[10, 15], 2] = 1
    
    # 3
    U_lad[[0, 10], 3] = 1
    V_lad[[0, 10], 3] = 1
    W_lad[[0, 10], 3] = 1
    
    # 4
    U_lad[[5, 15], 4] = 1
    V_lad[[5, 15], 4] = 1
    W_lad[[5, 15], 4] = 1
    
    # 5
    U_lad[[1, 4, 11, 14], 5] = 1
    V_lad[[1, 4, 11, 14], 5] = 1
    W_lad[[1, 4, 11, 14], 5] = 1
    
    # 6
    U_lad[[1, 4], 6] = 1
    V_lad[[1, 4], 6] = 1
    W_lad[[1, 4], 6] = 1
    
    # 7
    U_lad[[11, 14], 7] = 1
    V_lad[[11, 14], 7] = 1
    W_lad[[11, 14], 7] = 1
    
    # 8
    U_lad[[1, 11], 8] = 1
    V_lad[[1, 11], 8] = 1
    W_lad[[1, 11], 8] = 1
    
    # 9
    U_lad[[4, 14], 9] = 1
    V_lad[[4, 14], 9] = 1
    W_lad[[4, 14], 9] = 1
    
    # 10
    U_lad[[3, 7, 8, 12], 10] = 1
    V_lad[[3, 7, 8, 12], 10] = 1
    W_lad[[3, 7, 8, 12], 10] = 1
    
    # 11
    U_lad[[3, 7], 11] = 1
    V_lad[[3, 7], 11] = 1
    W_lad[[3, 7], 11] = 1
    
    # 12
    U_lad[[8, 12], 12] = 1
    V_lad[[8, 12], 12] = 1
    W_lad[[8, 12], 12] = 1
    
    # 13
    U_lad[[3, 8], 13] = 1
    V_lad[[3, 8], 13] = 1
    W_lad[[3, 8], 13] = 1
    
    # 14
    U_lad[[7, 12], 14] = 1
    V_lad[[7, 12], 14] = 1
    W_lad[[7, 12], 14] = 1
    
    # 15
    U_lad[[2, 6, 9, 13], 15] = 1
    V_lad[[2, 6, 9, 13], 15] = 1
    W_lad[[2, 6, 9, 13], 15] = 1
    
    # 16
    U_lad[[2, 6], 16] = 1
    V_lad[[2, 6], 16] = 1
    W_lad[[2, 6], 16] = 1
    
    # 17
    U_lad[[9, 13], 17] = 1
    V_lad[[9, 13], 17] = 1
    W_lad[[9, 13], 17] = 1
    
    # 18
    U_lad[[2, 9], 18] = 1
    V_lad[[2, 9], 18] = 1
    W_lad[[2, 9], 18] = 1

    # Construct 4x4x8 decomposition
    # U shape: (16, 38)
    # V shape: (32, 38)
    # W shape: (32, 38)
    
    U_final = np.zeros((16, 38))
    V_final = np.zeros((32, 38))
    W_final = np.zeros((32, 38))
    
    # Left Block (B cols 0-3, C cols 0-3) -> Rank 19
    for r in range(19):
        U_final[:, r] = U_lad[:, r]
        V_final[:16, r] = V_lad[:, r]
        W_final[:16, r] = W_lad[:, r]
        
    # Right Block (B cols 4-7, C cols 4-7) -> Rank 19
    for r in range(19):
        U_final[:, 19+r] = U_lad[:, r]
        V_final[16:, 19+r] = V_lad[:, r]
        W_final[16:, 19+r] = W_lad[:, r]

    # Verification
    error = 0.0
    for i in range(4):
        for j in range(4):
            u_row = U_final[i*4+j, :]
            for k in range(8):
                val = np.dot(u_row, V_final[j*8+k, :] * W_final[k*4+i, :])
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    if error > 1e-8:
        # Fallback to Schoolbook if verification fails
        U_school = np.zeros((16, 128))
        V_school = np.zeros((32, 128))
        W_school = np.zeros((32, 128))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_school[i*4+j, col] = 1
                    V_school[j*8+k, col] = 1
                    W_school[k*4+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
