import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Part 1: 1x4x8 Schoolbook (Rank 32)
    # Part 2: 2x4x8 Strassen-Hybrid (Rank 56)
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 1x4x8 Schoolbook ---
    # A is 1x4, B is 4x8, C is 1x8
    # C_k = sum_j A_j B_jk
    # U index: i*m+j. i=0. So 0..3.
    # V index: j*p+k. 0..31.
    # W index: k*n+i. i=0. So 0, 3, 6, ... 21.
    
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # --- Part 2: 2x4x8 Strassen-Hybrid ---
    # A is 2x4, B is 4x8, C is 2x8
    # Partition A into A_L (2x2) and A_R (2x2)
    # Partition B into B_T (2x8) and B_B (2x8)
    # C = A_L B_T + A_R B_B
    
    # Strassen 2x2x2 decomposition vectors
    # U_local, V_local, W_local for 2x2x2
    # Indices 0..3 for U, V, W
    # U: a00, a01, a10, a11
    # V: b00, b01, b10, b11
    # W: c00, c10, c01, c11 (Order: k*n+i -> 0*2+0=0, 1*2+0=2? No.
    # W index convention: k*n+i. n=2.
    # c00: k=0, i=0 -> 0
    # c10: k=0, i=1 -> 1
    # c01: k=1, i=0 -> 2
    # c11: k=1, i=1 -> 3
    # Wait, my previous W mapping for Strassen used:
    # C00 (idx 0), C10 (idx 1), C01 (idx 2), C11 (idx 3).
    # Let's re-verify the Strassen W vectors with this indexing.
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # W1: [1, 0, 0, 1] (M1 in C00, C11)
    # W2: [0, 1, 0, -1] (M2 in C10, C11)
    # W3: [0, 0, 1, 1] (M3 in C01, C11)
    # W4: [1, 1, 0, 0] (M4 in C00, C10)
    # W5: [-1, 0, 1, 0] (M5 in C00, C01)
    # W6: [0, 0, 0, 1] (M6 in C11)
    # W7: [1, 0, 0, 0] (M7 in C00)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Loop over output blocks j_block (0..3)
    for jb in range(4):
        for t in range(7):
            u_vec = S_U[t]
            v_vec = S_V[t]
            w_vec = S_W[t]
            
            # --- Sub-part L (A_L * B_T) ---
            # A_L: Global U rows 4,5 (i=1, j=0,1)
            # B_T: Global V rows 0,1 (j=0,1)
            # C: Global W rows for i=1,2 (global i=1,2)
            
            # Map U
            # Local u[0] -> a00 -> Global U[4] (i=1, j=0)
            # Local u[1] -> a01 -> Global U[5] (i=1, j=1)
            # Local u[2] -> a10 -> Global U[8] (i=2, j=0)
            # Local u[3] -> a11 -> Global U[9] (i=2, j=1)
            
            U[4][col] += u_vec[0]
            U[5][col] += u_vec[1]
            U[8][col] += u_vec[2]
            U[9][col] += u_vec[3]
            
            # Map V
            # Local v[0] -> b00 -> Global V[0*8 + 2*jb] = V[2*jb]
            # Local v[1] -> b01 -> Global V[0*8 + 2*jb+1] = V[2*jb+1]
            # Local v[2] -> b10 -> Global V[1*8 + 2*jb] = V[8+2*jb]
            # Local v[3] -> b11 -> Global V[1*8 + 2*jb+1] = V[8+2*jb+1]
            
            V[2*jb][col] += v_vec[0]
            V[2*jb+1][col] += v_vec[1]
            V[8+2*jb][col] += v_vec[2]
            V[8+2*jb+1][col] += v_vec[3]
            
            # Map W
            # Local w[0] -> c00 (k=0, i=0) -> Global C(k=2*jb, i=1) -> W[(2*jb)*3 + 1]
            # Local w[1] -> c10 (k=0, i=1) -> Global C(k=2*jb, i=2) -> W[(2*jb)*3 + 2]
            # Local w[2] -> c01 (k=1, i=0) -> Global C(k=2*jb+1, i=1) -> W[(2*jb+1)*3 + 1]
            # Local w[3] -> c11 (k=1, i=1) -> Global C(k=2*jb+1, i=2) -> W[(2*jb+1)*3 + 2]
            
            W[(2*jb)*3 + 1][col] += w_vec[0]
            W[(2*jb)*3 + 2][col] += w_vec[1]
            W[(2*jb+1)*3 + 1][col] += w_vec[2]
            W[(2*jb+1)*3 + 2][col] += w_vec[3]
            
            col += 1
            
            # --- Sub-part R (A_R * B_B) ---
            # A_R: Global U rows 6,7 (i=1, j=2,3) and 10,11 (i=2, j=2,3)
            # B_B: Global V rows 2,3 (j=2,3)
            
            # Map U
            # Local u[0] -> a00 -> Global U[1*4 + 2] = U[6]
            # Local u[1] -> a01 -> Global U[1*4 + 3] = U[7]
            # Local u[2] -> a10 -> Global U[2*4 + 2] = U[10]
            # Local u[3] -> a11 -> Global U[2*4 + 3] = U[11]
            
            U[6][col] += u_vec[0]
            U[7][col] += u_vec[1]
            U[10][col] += u_vec[2]
            U[11][col] += u_vec[3]
            
            # Map V
            # Local v[0] -> b00 -> Global V[2*8 + 2*jb] = V[16+2*jb]
            # Local v[1] -> b01 -> Global V[2*8 + 2*jb+1] = V[16+2*jb+1]
            # Local v[2] -> b10 -> Global V[3*8 + 2*jb] = V[24+2*jb]
            # Local v[3] -> b11 -> Global V[3*8 + 2*jb+1] = V[24+2*jb+1]
            
            V[16+2*jb][col] += v_vec[0]
            V[16+2*jb+1][col] += v_vec[1]
            V[24+2*jb][col] += v_vec[2]
            V[24+2*jb+1][col] += v_vec[3]
            
            # Map W
            # Same as L, because C indices are the same.
            
            W[(2*jb)*3 + 1][col] += w_vec[0]
            W[(2*jb)*3 + 2][col] += w_vec[1]
            W[(2*jb+1)*3 + 1][col] += w_vec[2]
            W[(2*jb+1)*3 + 2][col] += w_vec[3]
            
            col += 1

    return {"U": U, "V": V, "W": W}
