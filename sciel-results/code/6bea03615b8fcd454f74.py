import time
import numpy as np
import sys

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_vals = U_arr[i_idx*m + j_idx]
        v_vals = V_arr[j_idx*p + k_idx]
        w_vals = W_arr[k_idx*n + i_idx]
        
        tensor_vals = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        if np.any(np.abs(tensor_vals - 1.0) > tol):
            return False
        return True

    def build_strassen_2x2_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4

    def build_2x2x3_rank_11():
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                W[idx][r] = W2[idx][r]
        
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1
        
        return U, V, W

    def build_4x4x7_decomposition(U4, V4, W4, r4):
        U223, V223, W223 = build_2x2x3_rank_11()
        r223 = 11
        
        r_total = r4 + 4 * r223
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W

    U4_49, V4_49, W4_49 = kronecker_decompose_4x4x4()
    
    # Try to find rank 48 for 4x4x4
    U4_48, V4_48, W4_48 = None, None, None
    
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            # Remove one column from 49
            U_init = np.array(U4_49)[:, :48]
            V_init = np.array(V4_49)[:, :48]
            W_init = np.array(W4_49)[:, :48]
            
            # Gradient descent
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            lr = 0.001
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(5000):
                if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN - 5:
                    break
                
                # Compute error and gradients
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            s = np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :])
                            ds = 2 * (s - 1.0)
                            err += ds * ds / 4
                            grad_U[i*4+j, :] += ds * V[j*4+k, :] * W[k*4+i, :]
                            grad_V[j*4+k, :] += ds * U[i*4+j, :] * W[k*4+i, :]
                            grad_W[k*4+i, :] += ds * U[i*4+j, :] * V[j*4+k, :]
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    lr *= 1.01
                else:
                    lr *= 0.95
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if best_err < 1e-14:
                    break
            
            if best_err < 1e-14:
                U4_48 = [[best_U[i][r] for r in range(48)] for i in range(16)]
                V4_48 = [[best_V[i][r] for r in range(48)] for i in range(16)]
                W4_48 = [[best_W[i][r] for r in range(48)] for i in range(16)]
                
                # Verify
                if check_valid(U4_48, V4_48, W4_48):
                    print("Found rank 48 for 4x4x4")
                else:
                    U4_48, V4_48, W4_48 = None, None, None
        except Exception as e:
            print(e)
            U4_48, V4_48, W4_48 = None, None, None

    if U4_48 is not None:
        U_final, V_final, W_final = build_4x4x7_decomposition(U4_48, V4_48, W4_48, 48)
        rank = 48 + 44
    else:
        U_final, V_final, W_final = build_4x4x7_decomposition(U4_49, V4_49, W4_49, 49)
        rank = 49 + 44

    return {"U": U_final, "V": V_final, "W": W_final}
