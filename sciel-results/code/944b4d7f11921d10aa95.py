import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the rank-7 Strassen algorithm for 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen factors
    # U2[i*2 + j, r]
    U2 = np.zeros((4, 7), dtype=int)
    # r=0 (M1): A00, A11
    U2[0, 0] = 1; U2[3, 0] = 1
    # r=1 (M2): A10, A11
    U2[2, 1] = 1; U2[3, 1] = 1
    # r=2 (M3): A00
    U2[0, 2] = 1
    # r=3 (M4): A11
    U2[3, 3] = 1
    # r=4 (M5): A00, A01
    U2[0, 4] = 1; U2[1, 4] = 1
    # r=5 (M6): A10, A00
    U2[2, 5] = 1; U2[0, 5] = 1
    # r=6 (M7): A01, A11
    U2[1, 6] = 1; U2[3, 6] = 1

    # V2[j*2 + k, r]
    V2 = np.zeros((4, 7), dtype=int)
    # r=0 (M1): B00, B11
    V2[0, 0] = 1; V2[3, 0] = 1
    # r=1 (M2): B00
    V2[0, 1] = 1
    # r=2 (M3): B01, B11
    V2[1, 2] = 1; V2[3, 2] = 1
    # r=3 (M4): B10, B00
    V2[2, 3] = 1; V2[0, 3] = 1
    # r=4 (M5): B11
    V2[3, 4] = 1
    # r=5 (M6): B00, B01
    V2[0, 5] = 1; V2[1, 5] = 1
    # r=6 (M7): B10, B11
    V2[2, 6] = 1; V2[3, 6] = 1

    # W2[k*2 + i, r]
    W2 = np.zeros((4, 7), dtype=int)
    # C00 = M1 + M4 - M5 + M7 -> k=0, i=0 -> idx 0
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5 -> k=1, i=0 -> idx 1
    W2[1, 2] = 1; W2[1, 4] = 1
    # C10 = M2 + M4 -> k=0, i=1 -> idx 2
    W2[2, 1] = 1; W2[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6 -> k=1, i=1 -> idx 3
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1

    # Construct 4x4 factors
    # Shapes: U4 (16, 49), V4 (16, 49), W4 (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # Iterate over all combinations of 2x2 indices
    for i_a in range(2):
        for j_a in range(2):
            for k_a in range(2):
                for i_b in range(2):
                    for j_b in range(2):
                        for k_b in range(2):
                            # Compute 4x4 indices
                            i_4 = i_a * 2 + i_b
                            j_4 = j_a * 2 + j_b
                            k_4 = k_a * 2 + k_b
                            
                            idx_U = i_4 * 4 + j_4
                            idx_V = j_4 * 4 + k_4
                            idx_W = k_4 * 4 + i_4
                            
                            idx_U2_a = i_a * 2 + j_a
                            idx_V2_a = j_a * 2 + k_a
                            idx_W2_a = k_a * 2 + i_a
                            
                            idx_U2_b = i_b * 2 + j_b
                            idx_V2_b = j_b * 2 + k_b
                            idx_W2_b = k_b * 2 + i_b
                            
                            # Iterate over rank components
                            for r_a in range(7):
                                for r_b in range(7):
                                    r_4 = r_a * 7 + r_b
                                    
                                    u_val = U2[idx_U2_a, r_a] * U2[idx_U2_b, r_b]
                                    v_val = V2[idx_V2_a, r_a] * V2[idx_V2_b, r_b]
                                    w_val = W2[idx_W2_a, r_a] * W2[idx_W2_b, r_b]
                                    
                                    if u_val != 0:
                                        U4[idx_U, r_4] = u_val
                                    if v_val != 0:
                                        V4[idx_V, r_4] = v_val
                                    if w_val != 0:
                                        W4[idx_W, r_4] = w_val

    # Convert to lists as required by the interface
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
