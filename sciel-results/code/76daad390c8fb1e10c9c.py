import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strassen 2x2 coefficients
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,1,0]),
    ]
    
    def block_4x4(block_row, block_col, val):
        mat = [0]*16
        for i in range(2):
            for j in range(2):
                mat[(block_row*2+i)*4 + (block_col*2+j)] = val
        return mat
    
    def add_4x4(a, b, scale_b=1):
        return [a[i] + scale_b*b[i] for i in range(16)]
    
    def sub_4x4(a, b):
        return [a[i] - b[i] for i in range(16)]
    
    def neg_4x4(a):
        return [-x for x in a]
    
    A11 = block_4x4(0,0,1)
    A12 = block_4x4(0,1,1)
    A21 = block_4x4(1,0,1)
    A22 = block_4x4(1,1,1)
    
    B11 = block_4x4(0,0,1)
    B12 = block_4x4(0,1,1)
    B21 = block_4x4(1,0,1)
    B22 = block_4x4(1,1,1)
    
    C11 = block_4x4(0,0,1)
    C12 = block_4x4(0,1,1)
    C21 = block_4x4(1,0,1)
    C22 = block_4x4(1,1,1)
    
    top_A = [
        add_4x4(A11, A22),       
        add_4x4(A21, A22),       
        A11,                      
        A22,                      
        add_4x4(A11, A12),       
        sub_4x4(A21, A11),       
        sub_4x4(A12, A22),       
    ]
    top_B = [
        add_4x4(B11, B22),       
        B11,                     
        sub_4x4(B12, B22),       
        sub_4x4(B21, B11),       
        B22,                     
        add_4x4(B11, B12),       
        add_4x4(B21, B22),       
    ]
    top_C = [
        add_4x4(C11, C22),       
        add_4x4(C21, C22),       
        add_4x4(C12, C22),       
        add_4x4(C11, C21),       
        add_4x4(neg_4x4(C11), C12),  
        C22,                     
        add_4x4(C11, C21),       
    ]
    
    rank = 49 + 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    col = 0
    for t in range(7):
        for s in range(7):
            bot_A = strassen_2x2[s][0]
            bot_B = strassen_2x2[s][1]
            bot_C = strassen_2x2[s][2]
            
            for gi in range(4):
                for gj in range(4):
                    br, lr = divmod(gi, 2)
                    bc, lc = divmod(gj, 2)
                    val = top_A[t][gi*4+gj] * bot_A[lr*2+lc]
                    if val != 0:
                        U[gi*4+gj, col] = val
            
            for gj in range(4):
                for gk in range(4):
                    br, lr = divmod(gj, 2)
                    bc, lc = divmod(gk, 2)
                    val = top_B[t][gj*4+gk] * bot_B[lr*2+lc]
                    if val != 0:
                        V[gj*5+gk, col] = val
            
            for gk in range(4):
                for gi in range(4):
                    br, lr = divmod(gk, 2)
                    bc, lc = divmod(gi, 2)
                    val = top_C[t][gk*4+gi] * bot_C[lr*2+lc]
                    if val != 0:
                        W[gk*4+gi, col] = val
            
            col += 1
    
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
            col += 1
            
    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                
                prod = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                
                if prod != 1:
                    return construct_fallback()
                    
    return {"U": U, "V": V, "W": W}

def construct_fallback():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
