import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] += 1
        U[3,0] += 1
        V[0,0] += 1
        V[3,0] += 1
        W[0,0] += 1  # C[0,0]
        W[3,0] += 1  # C[1,1]
        
        # M2 = (A10+A11)B00
        U[2,1] += 1
        U[3,1] += 1
        V[0,1] += 1
        W[2,1] += 1  # C[1,0]
        W[3,1] += 1  # C[1,1]
        
        # M3 = A00(B01-B11)
        U[0,2] += 1
        V[1,2] += 1
        V[3,2] -= 1
        W[1,2] += 1  # C[0,1]
        W[3,2] += 1  # C[1,1]
        
        # M4 = A11(B10-B00)
        U[3,3] += 1
        V[2,3] += 1
        V[0,3] -= 1
        W[0,3] += 1  # C[0,0]
        W[2,3] += 1  # C[1,0]
        
        # M5 = (A00+A01)B11
        U[0,4] += 1
        U[1,4] += 1
        V[3,4] += 1
        W[0,4] -= 1  # C[0,0]
        W[1,4] += 1  # C[0,1]
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5] += 1
        U[0,5] -= 1
        V[0,5] += 1
        V[1,5] += 1
        W[3,5] += 1  # C[1,1]
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6] += 1
        U[3,6] -= 1
        V[2,6] += 1
        V[3,6] += 1
        W[0,6] += 1  # C[0,0]
        W[1,6] += 1  # C[0,1]
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    outer_products = [
        [(1,0,0,1, 1,0,0,1, 0, 1),
         (1,0,0,1, 1,0,0,1, 3, 1)],
        [(0,0,1,1, 1,0,0,0, 2, 1),
         (0,0,1,1, 1,0,0,0, 3, -1)],
        [(1,0,0,0, 0,1,0,-1, 1, 1),
         (1,0,0,0, 0,1,0,-1, 3, 1)],
        [(0,0,0,1, -1,0,1,0, 0, 1),
         (0,0,0,1, -1,0,1,0, 2, 1)],
        [(1,1,0,0, 0,0,0,1, 0, -1),
         (1,1,0,0, 0,0,0,1, 1, 1)],
        [(0,0,1,-1, 1,1,0,0, 3, 1)],
        [(0,1,0,-1, 0,0,1,1, 0, 1),
         (0,1,0,-1, 0,0,1,1, 1, 1)],
    ]
    
    col = 0
    for prod_idx, entries in enumerate(outer_products):
        for entry in entries:
            a11, a12, a21, a22, b11, b12, b21, b22, c_block, sign = entry
            for inner_idx in range(7):
                if a11 != 0:
                    U_44[0:4, col] += a11 * U2[:, inner_idx]
                if a12 != 0:
                    U_44[4:8, col] += a12 * U2[:, inner_idx]
                if a21 != 0:
                    U_44[8:12, col] += a21 * U2[:, inner_idx]
                if a22 != 0:
                    U_44[12:16, col] += a22 * U2[:, inner_idx]
                
                if b11 != 0:
                    V_44[0:4, col] += b11 * V2[:, inner_idx]
                if b12 != 0:
                    V_44[4:8, col] += b12 * V2[:, inner_idx]
                if b21 != 0:
                    V_44[8:12, col] += b21 * V2[:, inner_idx]
                if b22 != 0:
                    V_44[12:16, col] += b22 * V2[:, inner_idx]
                
                for k in range(2):
                    for i in range(2):
                        w2_idx = k * 2 + i
                        if c_block == 0:
                            w44_idx = k * 4 + i
                        elif c_block == 1:
                            w44_idx = (2 + k) * 4 + i
                        elif c_block == 2:
                            w44_idx = k * 4 + (2 + i)
                        else:
                            w44_idx = (2 + k) * 4 + (2 + i)
                        W_44[w44_idx, col] += sign * W2[w2_idx, inner_idx]
                col += 1
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
