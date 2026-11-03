import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1  # A00 (0,0), A11 (1,1)
    V2[0, 0] = 1; V2[3, 0] = 1  # B00 (0,0), B11 (1,1)
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 (0,0), C11 (1,1)
    
    # Term 2: M2 = (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1  # A10 (1,0), A11 (1,1)
    V2[0, 1] = 1                # B00 (0,0)
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 (1,0), C11 (1,1)
    
    # Term 3: M3 = A00(B01 - B11)
    U2[0, 2] = 1                # A00 (0,0)
    V2[1, 2] = 1; V2[3, 2] = -1 # B01 (0,1), B11 (1,1)
    W2[1, 2] = 1; W2[3, 2] = 1  # C01 (0,1), C11 (1,1)
    
    # Term 4: M4 = A11(B10 - B00)
    U2[3, 3] = 1                # A11 (1,1)
    V2[2, 3] = 1; V2[0, 3] = -1 # B10 (1,0), B00 (0,0)
    W2[0, 3] = 1; W2[2, 3] = 1  # C00 (0,0), C10 (1,0)
    
    # Term 5: M5 = (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1  # A00 (0,0), A01 (0,1)
    V2[3, 4] = 1                # B11 (1,1)
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 (0,0), C01 (0,1)
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1 # A10 (1,0), A00 (0,0)
    V2[0, 5] = 1; V2[1, 5] = 1  # B00 (0,0), B01 (0,1)
    W2[3, 5] = 1                # C11 (1,1)
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1 # A01 (0,1), A11 (1,1)
    V2[2, 6] = 1; V2[3, 6] = 1  # B10 (1,0), B11 (1,1)
    W2[0, 6] = 1                # C00 (0,0)

    # 2. Construct 4x4x4 decomposition using Kronecker product
    U4_kron = np.kron(U2, U2) # 16x49
    V4_kron = np.kron(V2, V2) # 16x49
    W4_kron = np.kron(W2, W2) # 16x49

    # Permute rows to match standard indexing
    # U4 row R = (2I+J)*4 + (2i+j)
    # Standard u = 4(2I+i) + (2J+j)
    # Map R -> u
    perm_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i in range(2):
                for j in range(2):
                    R = (2*I + J)*4 + (2*i + j)
                    u = 4*(2*I + i) + (2*J + j)
                    perm_U[R] = u
    U4 = U4_kron[perm_U]

    # V4 row S = (2J+K)*4 + (2j+k)
    # Standard v = 7j + k? No, for 4x4x4, v = 4j + k.
    # But here we are constructing the 4x4x4 part which will be embedded in 4x4x7.
    # The V4 part corresponds to B_right (4x4).
    # In the full 4x4x7 problem, B is 4x7.
    # B_right is columns 3..6.
    # So v_full = 7j + k, where k in 3..6.
    # Let's first construct V4 with standard 4x4 indexing v_4 = 4j + k.
    # Then map to v_full.
    perm_V = np.zeros(16, dtype=int)
    for J in range(2):
        for K in range(2):
            for j in range(2):
                for k in range(2):
                    S = (2*J + K)*4 + (2*j + k)
                    v_4 = 4*(2*J + j) + (2*K + k)
                    perm_V[S] = v_4
    V4 = V4_kron[perm_V]

    # W4 row T = (2K+I)*4 + (2k+i)
    # Standard w_4 = 4k + i (for 4x4x4, W is 4x4).
    perm_W = np.zeros(16, dtype=int)
    for K in range(2):
        for I in range(2):
            for k in range(2):
                for i in range(2):
                    T = (2*K + I)*4 + (2*k + i)
                    w_4 = 4*(2*K + k) + (2*I + i)
                    perm_W[T] = w_4
    W4 = W4_kron[perm_W]

    # 3. Construct 4x4x3 schoolbook decomposition
    rank_left = 4 * 4 * 3
    U_left = np.zeros((16, rank_left))
    V_left = np.zeros((12, rank_left)) # 4x3
    W_left = np.zeros((12, rank_left)) # 3x4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_left[i*4 + j, col] = 1
                V_left[j*3 + k, col] = 1
                W_left[k*4 + i, col] = 1
                col += 1

    # 4. Combine
    rank_total = 49 + rank_left
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((28, rank_total))
    W_full = np.zeros((28, rank_total))
    
    # Right part (49 cols)
    U_full[:, 0:49] = U4
    # V_full rows for B_right (k=3..6)
    # v_full = 7j + k
    # V4 rows are v_4 = 4j + k_4, where k_4 in 0..3.
    # k_full = 3 + k_4.
    for v_4 in range(16):
        j = v_4 // 4
        k_4 = v_4 % 4
        v_full = 7*j + (3 + k_4)
        V_full[v_full, 0:49] = V4[v_4]
        
    # W_full rows for C_right (k=3..6)
    # w_full = 4k + i
    # W4 rows are w_4 = 4k_4 + i.
    # k_full = 3 + k_4.
    for w_4 in range(16):
        k_4 = w_4 // 4
        i = w_4 % 4
        w_full = 4*(3 + k_4) + i
        W_full[w_full, 0:49] = W4[w_4]

    # Left part (48 cols)
    U_full[:, 49:97] = U_left
    # V_full rows for B_left (k=0..2)
    # v_full = 7j + k
    # V_left rows are v_l = 3j + k.
    for v_l in range(12):
        j = v_l // 3
        k = v_l % 3
        v_full = 7*j + k
        V_full[v_full, 49:97] = V_left[v_l]
        
    # W_full rows for C_left (k=0..2)
    # w_full = 4k + i
    # W_left rows are w_l = 4k + i.
    # They are in the same position!
    W_full[0:12, 49:97] = W_left

    return {"U": U_full, "V": V_full, "W": W_full}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
