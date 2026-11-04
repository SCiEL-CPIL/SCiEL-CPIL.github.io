import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = time.time() + 140
_START = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Kronecker product of 2x2 Strassen (rank 7) and 3x3 Schoolbook (rank 27)
    # Total rank = 7 * 27 = 189
    
    # 2x2 Strassen decomposition
    # U_out, V_out, W_out shapes (4, 7)
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[1, 1] = 1; W_out[3, 1] = -1
    
    # M3
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[2, 2] = 1; W_out[3, 2] = 1
    
    # M4
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[2, 4] = 1
    
    # M6
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M7
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1
    
    # 3x3 Schoolbook decomposition
    # U_in, V_in, W_in shapes (9, 27)
    U_in = np.zeros((9, 27), dtype=int)
    V_in = np.zeros((9, 27), dtype=int)
    W_in = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_in[i * 3 + j, col] = 1
                V_in[j * 3 + k, col] = 1
                W_in[k * 3 + i, col] = 1
                col += 1
                
    rank = 7 * 27
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(27):
            if time.time() > DEADLINE:
                # Return best effort (partial) if deadline exceeded, though unlikely
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            c = c_out * 27 + c_in
            
            # U
            for r_out in range(4):
                if U_out[r_out, c_out] != 0:
                    i_out = r_out // 2
                    j_out = r_out % 2
                    val_out = U_out[r_out, c_out]
                    for r_in in range(9):
                        if U_in[r_in, c_in] != 0:
                            i_in = r_in // 3
                            j_in = r_in % 3
                            val_in = U_in[r_in, c_in]
                            r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
                            U[r, c] += val_out * val_in
                            
            # V
            for r_out in range(4):
                if V_out[r_out, c_out] != 0:
                    j_out = r_out // 2
                    k_out = r_out % 2
                    val_out = V_out[r_out, c_out]
                    for r_in in range(9):
                        if V_in[r_in, c_in] != 0:
                            j_in = r_in // 3
                            k_in = r_in % 3
                            val_in = V_in[r_in, c_in]
                            r = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in)
                            V[r, c] += val_out * val_in
                            
            # W
            for r_out in range(4):
                if W_out[r_out, c_out] != 0:
                    k_out = r_out // 2
                    i_out = r_out % 2
                    val_out = W_out[r_out, c_out]
                    for r_in in range(9):
                        if W_in[r_in, c_in] != 0:
                            k_in = r_in // 3
                            i_in = r_in % 3
                            val_in = W_in[r_in, c_in]
                            r = (k_out * 3 + k_in) * 6 + (i_out * 3 + i_in)
                            W[r, c] += val_out * val_in
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
