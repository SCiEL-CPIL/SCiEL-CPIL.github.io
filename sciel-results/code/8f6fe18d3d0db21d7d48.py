"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U index: i*2 + j
        V index: j*2 + k
        W index: k*2 + i
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # a=A00, d=A11, e=B00, h=B11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C11 += M1
        W[0, 0] = 1; W[3, 0] = 1 # W[k*2+i]
        # C00 is k=0,i=0 -> idx 0. C11 is k=1,i=1 -> idx 3.
        
        # M2 = (c+d)h
        # c=A10, d=A11, h=B11
        U[2, 1] = 1; U[3, 1] = 1
        V[3, 1] = 1
        # C10 += M2, C11 -= M2
        # C10 is k=0,i=1 -> idx 1. C11 is k=1,i=1 -> idx 3.
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        # a=A00, f=B10, h=B11
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C01 += M3, C11 += M3
        # C01 is k=1,i=0 -> idx 2. C11 is k=1,i=1 -> idx 3.
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        # d=A11, g=B01, e=B00
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C00 += M4, C10 += M4
        # C00 idx 0, C10 idx 1.
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+b)e
        # a=A00, b=A01, e=B00
        U[0, 4] = 1; U[1, 4] = 1
        V[0, 4] = 1
        # C00 -= M5, C01 += M5
        # C00 idx 0, C01 idx 2.
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (c+d)(h-g)
        # c=A10, d=A11, h=B11, g=B01
        U[2, 5] = 1; U[3, 5] = 1
        V[3, 5] = 1; V[1, 5] = -1
        # C11 += M6
        # C11 idx 3.
        W[3, 5] = 1
        
        # M7 = (a-b)(e+f)
        # a=A00, b=A01, e=B00, f=B10
        U[0, 6] = 1; U[1, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        # C00 += M7
        # C00 idx 0.
        W[0, 6] = 1
        
        return U, V, W

    def get_2x1x2():
        """
        Returns U, V, W for 2x1x2 decomposition.
        U shape (2, 4), V shape (2, 4), W shape (4, 4)
        U index: i*1 + 0 = i
        V index: 0*2 + k = k
        W index: k*2 + i
        """
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # C00 = A00 B00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # k=0,i=0
        
        # C01 = A00 B01
        U[0, 1] = 1
        V[1, 1] = 1
        W[2, 1] = 1 # k=1,i=0
        
        # C10 = A10 B00
        U[1, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1 # k=0,i=1
        
        # C11 = A10 B01
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1 # k=1,i=1
        
        return U, V, W

    def get_2x2x6():
        """
        Returns U, V, W for 2x2x6 decomposition.
        Composed of 3 Strassen 2x2x2 blocks.
        U shape (4, 21), V shape (12, 21), W shape (12, 21)
        U index: i*2 + j
        V index: j*6 + k
        W index: k*2 + i
        """
        U = np.zeros((4, 21))
        V = np.zeros((12, 21))
        W = np.zeros((12, 21))
        
        U_sub, V_sub, W_sub = get_strassen_2x2x2()
        
        for block in range(3):
            start_r = block * 7
            U[:, start_r:start_r+7] = U_sub
            
            # V_sub is 4x7 (j*2+k). j in {0,1}, k in {0,1}
            # We need to map to V (j*6+k). j in {0,1}, k in {block*2, block*2+1}
            for j in range(2):
                for k in range(2):
                    v_idx_sub = j * 2 + k
                    v_idx_global = j * 6 + (block * 2 + k)
                    V[v_idx_global, start_r:start_r+7] = V_sub[v_idx_sub, :]
            
            # W_sub is 4x7 (k*2+i). k in {0,1}, i in {0,1}
            # We need to map to W (k*2+i). k in {block*2, block*2+1}, i in {0,1}
            for k in range(2):
                for i in range(2):
                    w_idx_sub = k * 2 + i
                    w_idx_global = (block * 2 + k) * 2 + i
                    W[w_idx_global, start_r:start_r+7] = W_sub[w_idx_sub, :]
                    
        return U, V, W

    def get_2x3x2():
        """
        Returns U, V, W for 2x3x2 decomposition.
        Composed of one 2x2x2 Strassen and one 2x1x2.
        U shape (6, 11), V shape (6, 11), W shape (4, 11)
        U index: i*3 + j
        V index: j*2 + k
        W index: k*2 + i
        """
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_1, V_1, W_1 = get_2x1x2()
        
        # Part 1: 2x2x2 (j=0,1)
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + j
                U[u_idx, 0:7] = U_s[i * 2 + j, :]
        
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + k
                V[v_idx, 0:7] = V_s[j * 2 + k, :]
        
        W[0:4, 0:7] = W_s
        
        # Part 2: 2x1x2 (j=2)
        for i in range(2):
            u_idx = i * 3 + 2
            U[u_idx, 7:11] = U_1[i, :]
        
        for k in range(2):
            v_idx = 2 * 2 + k
            V[v_idx, 7:11] = V_1[k, :]
        
        W[0:4, 7:11] = W_1
        
        return U, V, W

    def get_2x3x6():
        """
        Returns U, V, W for 2x3x6 decomposition.
        Composed of 3 2x3x2 blocks.
        U shape (6, 33), V shape (18, 33), W shape (12, 33)
        U index: i*3 + j
        V index: j*6 + k
        W index: k*2 + i
        """
        U = np.zeros((6, 33))
        V = np.zeros((18, 33))
        W = np.zeros((12, 33))
        
        U_sub, V_sub, W_sub = get_2x3x2()
        
        for block in range(3):
            start_r = block * 11
            U[:, start_r:start_r+11] = U_sub
            
            # V_sub is 6x11 (j*2+k). j in {0,1,2}, k in {0,1}
            # Map to V (j*6+k). j in {0,1,2}, k in {block*2, block*2+1}
            for j in range(3):
                for k in range(2):
                    v_idx_sub = j * 2 + k
                    v_idx_global = j * 6 + (block * 2 + k)
                    V[v_idx_global, start_r:start_r+11] = V_sub[v_idx_sub, :]
            
            # W_sub is 4x11 (k*2+i). k in {0,1}, i in {0,1}
            # Map to W (k*2+i). k in {block*2, block*2+1}, i in {0,1}
            for k in range(2):
                for i in range(2):
                    w_idx_sub = k * 2 + i
                    w_idx_global = (block * 2 + k) * 2 + i
                    W[w_idx_global, start_r:start_r+11] = W_sub[w_idx_sub, :]
                    
        return U, V, W

    # Combine 2x2x6 and 2x3x6 into 2x5x6
    U_22, V_22, W_22 = get_2x2x6()
    U_23, V_23, W_23 = get_2x3x6()
    
    U_final = np.zeros((10, 54))
    V_final = np.zeros((30, 54))
    W_final = np.zeros((12, 54))
    
    # 2x2x6 part (j=0,1)
    for i in range(2):
        for j in range(2):
            u_idx = i * 5 + j
            U_final[u_idx, 0:21] = U_22[i * 2 + j, :]
            
    for j in range(2):
        for k in range(6):
            v_idx = j * 6 + k
            V_final[v_idx, 0:21] = V_22[j * 6 + k, :]
            
    W_final[0:12, 0:21] = W_22
    
    # 2x3x6 part (j=2,3,4)
    for i in range(2):
        for j in range(3):
            u_idx = i * 5 + (2 + j)
            U_final[u_idx, 21:54] = U_23[i * 3 + j, :]
            
    for j in range(3):
        for k in range(6):
            v_idx = (2 + j) * 6 + k
            V_final[v_idx, 21:54] = V_23[j * 6 + k, :]
            
    W_final[0:12, 21:54] = W_23
    
    # Convert to list of lists of floats
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
