"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 33 decomposition using Strassen for 2x2 blocks + residual
    # A is 2x3, B is 3x6, C is 2x6
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x6) and B_bottom (1x6)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # A_left @ B_top:
    # B_top is 2x6. Split into 3 blocks of 2x2.
    # Each block uses Strassen (rank 7). Total 21.
    
    # A_right @ B_bottom:
    # A_right is 2x1, B_bottom is 1x6.
    # Outer product. Rank 2*6 = 12.
    
    total_rank = 21 + 12
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a Strassen block
    # a_indices: indices in U for A_{00}, A_{01}, A_{10}, A_{11}
    # b_indices: indices in V for B_{00}, B_{01}, B_{10}, B_{11}
    # c_indices: indices in W for C_{00}, C_{01}, C_{10}, C_{11}
    def add_strassen_block(a_idx, b_idx, c_idx, start_col):
        c = start_col
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0], c] += 1; U[a_idx[3], c] += 1
        V[b_idx[0], c] += 1; V[b_idx[3], c] += 1
        W[c_idx[0], c] += 1; W[c_idx[1], c] += 1; W[c_idx[2], c] += 1; W[c_idx[3], c] += 1
        c += 1
        
        # M2 = (A10+A11)B00
        U[a_idx[2], c] += 1; U[a_idx[3], c] += 1
        V[b_idx[0], c] += 1
        W[c_idx[2], c] += 1; W[c_idx[3], c] += 1
        c += 1
        
        # M3 = A00(B10-B11)
        U[a_idx[0], c] += 1
        V[b_idx[2], c] += 1; V[b_idx[3], c] -= 1
        W[c_idx[1], c] += 1; W[c_idx[3], c] += 1
        c += 1
        
        # M4 = A11(B01-B00)
        U[a_idx[3], c] += 1
        V[b_idx[1], c] += 1; V[b_idx[0], c] -= 1
        W[c_idx[0], c] += 1; W[c_idx[2], c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[a_idx[0], c] += 1; U[a_idx[1], c] += 1
        V[b_idx[3], c] += 1
        W[c_idx[0], c] -= 1; W[c_idx[1], c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_idx[2], c] += 1; U[a_idx[0], c] -= 1
        V[b_idx[0], c] += 1; V[b_idx[1], c] += 1
        W[c_idx[3], c] += 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_idx[1], c] += 1; U[a_idx[3], c] -= 1
        V[b_idx[2], c] += 1; V[b_idx[3], c] += 1
        W[c_idx[0], c] += 1; W[c_idx[3], c] -= 1
        c += 1
        
        return c

    # Process 3 Strassen blocks for A_left (cols 0,1) and B_top (rows 0,1)
    # A_left indices in U:
    # A00: 0*3+0=0, A01: 0*3+1=1, A10: 1*3+0=3, A11: 1*3+1=4
    a_idx = [0, 1, 3, 4]
    
    for block in range(3):
        k1 = block * 2
        k2 = block * 2 + 1
        
        # B_top block indices in V:
        # B00: 0*6+k1 = k1
        # B01: 0*6+k2 = k2
        # B10: 1*6+k1 = 6+k1
        # B11: 1*6+k2 = 6+k2
        b_idx = [k1, k2, 6+k1, 6+k2]
        
        # C block indices in W:
        # C00: k1*2+0 = 2*k1
        # C01: k2*2+0 = 2*k2  (Wait, W index is k*n+i. C_{0,k1} -> k1*2+0. C_{0,k2} -> k2*2+0)
        # C10: k1*2+1 = 2*k1+1
        # C11: k2*2+1 = 2*k2+1
        c_idx = [2*k1, 2*k2, 2*k1+1, 2*k2+1]
        
        col = add_strassen_block(a_idx, b_idx, c_idx, col)
        
    # Process residual A_right (col 2) and B_bottom (row 2)
    # A_right indices in U:
    # A02: 0*3+2=2
    # A12: 1*3+2=5
    # B_bottom indices in V:
    # B20: 2*6+0=12
    # ...
    # B25: 2*6+5=17
    
    for i in range(n): # i=0,1
        for k in range(p): # k=0..5
            u_idx = i * m + 2 # A_{i2}
            v_idx = 2 * p + k # B_{2k}
            w_idx = k * n + i # C_{ik}
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
