import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U_loc: 4x7, V_loc: 4x7, W_loc: 4x7
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[1, 1] = 1
    
    # Term 2: M3 = A00(B01-B11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[2, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 8 blocks: 2 for j-split, 4 for k-split
    for bj in range(2):
        for bk in range(4):
            block_offset = 7 * (4 * bj + bk)
            for s in range(7):
                t = block_offset + s
                
                # Map local indices to global
                # Local a_loc = 2*i + j_loc
                # Global a = 4*i + (2*bj + j_loc)
                for i in range(2):
                    for j_loc in range(2):
                        a_loc = 2 * i + j_loc
                        a_glob = 4 * i + (2 * bj + j_loc)
                        U[a_glob, t] = U_loc[a_loc, s]
                        
                # Local b_loc = 2*j_loc + k_loc
                # Global b = 8*(2*bj + j_loc) + (4*bk + k_loc)
                for j_loc in range(2):
                    for k_loc in range(2):
                        b_loc = 2 * j_loc + k_loc
                        b_glob = 8 * (2 * bj + j_loc) + (4 * bk + k_loc)
                        V[b_glob, t] = V_loc[b_loc, s]
                        
                # Local c_loc = 2*k_loc + i
                # Global c = 8*(4*bk + k_loc) + i
                for k_loc in range(2):
                    for i in range(2):
                        c_loc = 2 * k_loc + i
                        c_glob = 8 * (4 * bk + k_loc) + i
                        W[c_glob, t] = W_loc[c_loc, s]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
