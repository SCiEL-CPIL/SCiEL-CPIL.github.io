import numpy as np

def strassen_2x2():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1; W[0, 5] = 1
    
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    
    U1, V1, W1 = strassen_2x2()
    
    def build_444(p_offset):
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((36, 49), dtype=int)
        W = np.zeros((36, 49), dtype=int)
        
        for alpha in range(7):
            for beta in range(7):
                col = alpha * 7 + beta
                for I in range(2):
                    for J in range(2):
                        for i in range(2):
                            for j in range(2):
                                u_val = U1[I*2 + J, alpha] * U1[i*2 + j, beta]
                                U[(I*2 + i)*4 + (J*2 + j), col] = u_val
                                
                                for K in range(2):
                                    for k in range(2):
                                        v_val = V1[J*2 + K, alpha] * V1[j*2 + k, beta]
                                        V[(J*2 + j)*9 + (K*2 + k + p_offset), col] = v_val
                                        
                                        w_val = W1[K*2 + I, alpha] * W1[k*2 + i, beta]
                                        W[(K*2 + k + p_offset)*4 + (I*2 + i), col] = w_val
                                        
        return U, V, W
        
    U_part1, V_part1, W_part1 = build_444(0)
    U_part2, V_part2, W_part2 = build_444(4)
    
    U_part3 = np.zeros((16, 16), dtype=int)
    V_part3 = np.zeros((36, 16), dtype=int)
    W_part3 = np.zeros((36, 16), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_part3[i*4 + j, col] = 1
            V_part3[j*9 + 8, col] = 1
            W_part3[8*4 + i, col] = 1
            col += 1
            
    U = np.hstack([U_part1, U_part2, U_part3])
    V = np.hstack([V_part1, V_part2, V_part3])
    W = np.hstack([W_part1, W_part2, W_part3])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
