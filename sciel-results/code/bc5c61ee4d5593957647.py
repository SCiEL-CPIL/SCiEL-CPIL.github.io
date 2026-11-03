import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 25
    U = np.zeros((n*m, rank))  # 8 x 25
    V = np.zeros((m*p, rank))  # 24 x 25
    W = np.zeros((p*n, rank))  # 12 x 25
    
    col_idx = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[:, 0] indices in U: k*4 + 0
    # B[0, :] indices in V: 0*6 + j
    # C[k, j] index in W: j*n + k = j*2 + k
    
    for k in range(2):
        for j in range(6):
            U[k*4 + 0, col_idx] += 1
            V[0*6 + j, col_idx] += 1
            W[j*2 + k, col_idx] += 1
            
    col_idx += 1
    
    # Part 2: A[:, 1:4] * B[1:4, :]
    # A' is 2x3. U indices: k*4 + (1+pj')
    # B' is 3x6. V indices: (1+qi')*6 + j
    
    for c in range(3):
        # Block c: B[1:4, 2c:2c+2]
        # j in {2c, 2c+1}
        
        # Sub-part 2a: A[:, 1] * B[1, 2c:2c+2]
        # U: k*4 + 1
        # V: 1*6 + 2c + j' (j' in 0,1)
        # W: (2c+j')*2 + k
        
        for k in range(2):
            for j_prime in range(2):
                j = 2*c + j_prime
                U[k*4 + 1, col_idx] += 1
                V[1*6 + j, col_idx] += 1
                W[j*2 + k, col_idx] += 1
                
        col_idx += 1
        
        # Sub-part 2b: A[:, 2:4] * B[2:4, 2c:2c+2]
        # U: k*4 + 2, k*4 + 3
        # V: (2+qi'')*6 + 2c + j'
        # W: (2c+j')*2 + k
        
        # Strassen
        # P00=U[0*4+2], P01=U[0*4+3], P10=U[1*4+2], P11=U[1*4+3]
        # Q00=V[2*6+2c], Q01=V[2*6+2c+1], Q10=V[3*6+2c], Q11=V[3*6+2c+1]
        # R00 -> C[0, 2c] -> W[2c*2+0]
        # R01 -> C[0, 2c+1] -> W[(2c+1)*2+0]
        # R10 -> C[1, 2c] -> W[2c*2+1]
        # R11 -> C[1, 2c+1] -> W[(2c+1)*2+1]
        
        # M1
        U[0*4 + 2, col_idx] += 1
        U[1*4 + 3, col_idx] += 1
        V[2*6 + 2*c, col_idx] += 1
        V[3*6 + 2*c+1, col_idx] += 1
        W[2*c*2 + 0, col_idx] += 1
        W[(2*c+1)*2 + 1, col_idx] += 1
        col_idx += 1
        
        # M2
        U[1*4 + 2, col_idx] += 1
        U[1*4 + 3, col_idx] += 1
        V[2*6 + 2*c, col_idx] += 1
        W[2*c*2 + 1, col_idx] += 1
        W[(2*c+1)*2 + 1, col_idx] += -1
        col_idx += 1
        
        # M3
        U[0*4 + 2, col_idx] += 1
        V[2*6 + 2*c+1, col_idx] += 1
        V[3*6 + 2*c+1, col_idx] += -1
        W[(2*c+1)*2 + 0, col_idx] += 1
        W[(2*c+1)*2 + 1, col_idx] += 1
        col_idx += 1
        
        # M4
        U[1*4 + 3, col_idx] += 1
        V[3*6 + 2*c, col_idx] += 1
        V[2*6 + 2*c, col_idx] += -1
        W[2*c*2 + 0, col_idx] += 1
        W[2*c*2 + 1, col_idx] += 1
        col_idx += 1
        
        # M5
        U[0*4 + 2, col_idx] += 1
        U[0*4 + 3, col_idx] += 1
        V[3*6 + 2*c+1, col_idx] += 1
        W[2*c*2 + 0, col_idx] += -1
        W[(2*c+1)*2 + 0, col_idx] += 1
        col_idx += 1
        
        # M6
        U[1*4 + 2, col_idx] += 1
        U[0*4 + 2, col_idx] += -1
        V[2*6 + 2*c, col_idx] += 1
        V[2*6 + 2*c+1, col_idx] += 1
        W[(2*c+1)*2 + 1, col_idx] += 1
        col_idx += 1
        
        # M7
        U[0*4 + 3, col_idx] += 1
        U[1*4 + 3, col_idx] += -1
        V[3*6 + 2*c, col_idx] += 1
        V[3*6 + 2*c+1, col_idx] += 1
        W[2*c*2 + 0, col_idx] += 1
        col_idx += 1
        
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
