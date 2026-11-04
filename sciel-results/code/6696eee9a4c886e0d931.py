import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition (Rank 7)
    # U_out, V_out, W_out shapes (4, 7)
    # Indices 0..3 correspond to 2x2 blocks (i_out, j_out)
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2: (A10+A11)B00
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[1, 1] = 1; W_out[3, 1] = -1
    
    # M3: A00(B01-B11)
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[2, 2] = 1; W_out[3, 2] = 1
    
    # M4: A11(B10-B00)
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5: (A00+A01)B11
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1

    # 3x3 Schoolbook decomposition (Rank 27)
    # U_in, V_in, W_in shapes (9, 27)
    # Indices 0..8 correspond to 3x3 blocks (i_in, j_in)
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
    
    # Pre-compute index mappings for speed
    # U/V/W row indices in 6x6 matrix
    # U: (i_out, j_out) -> i, j. i = i_out*3 + i_in, j = j_out*3 + j_in
    # V: (j_out, k_out) -> j, k. j = j_out*3 + j_in, k = k_out*3 + k_in
    # W: (k_out, i_out) -> k, i. k = k_out*3 + k_in, i = i_out*3 + i_in
    
    for c_out in range(7):
        for c_in in range(27):
            if time.time() > DEADLINE:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            c = c_out * 27 + c_in
            
            # Process U
            u_out_vals = U_out[:, c_out]
            u_in_vals = U_in[:, c_in]
            
            nz_out = np.nonzero(u_out_vals)[0]
            nz_in = np.nonzero(u_in_vals)[0]
            
            for ro in nz_out:
                io = ro // 2
                jo = ro % 2
                val_o = u_out_vals[ro]
                
                for ri in nz_in:
                    ii = ri // 3
                    ji = ri % 3
                    val_i = u_in_vals[ri]
                    
                    i = io * 3 + ii
                    j = jo * 3 + ji
                    r = i * 6 + j
                    U[r, c] += val_o * val_i
                    
            # Process V
            v_out_vals = V_out[:, c_out]
            v_in_vals = V_in[:, c_in]
            
            nz_out = np.nonzero(v_out_vals)[0]
            nz_in = np.nonzero(v_in_vals)[0]
            
            for ro in nz_out:
                jo = ro // 2
                ko = ro % 2
                val_o = v_out_vals[ro]
                
                for ri in nz_in:
                    ji = ri // 3
                    ki = ri % 3
                    val_i = v_in_vals[ri]
                    
                    j = jo * 3 + ji
                    k = ko * 3 + ki
                    r = j * 6 + k
                    V[r, c] += val_o * val_i
                    
            # Process W
            w_out_vals = W_out[:, c_out]
            w_in_vals = W_in[:, c_in]
            
            nz_out = np.nonzero(w_out_vals)[0]
            nz_in = np.nonzero(w_in_vals)[0]
            
            for ro in nz_out:
                ko = ro // 2
                io = ro % 2
                val_o = w_out_vals[ro]
                
                for ri in nz_in:
                    ki = ri // 3
                    ii = ri % 3
                    val_i = w_in_vals[ri]
                    
                    k = ko * 3 + ki
                    i = io * 3 + ii
                    r = k * 6 + i
                    W[r, c] += val_o * val_i

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
