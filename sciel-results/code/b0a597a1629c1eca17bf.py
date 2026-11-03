"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n + i, r] corresponds to output C[i, k]
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # We will construct a decomposition of rank 31 using block structure.
    # M_{2,5,5} split into M_{2,2,5} + M_{2,2,5} + M_{2,1,5}
    # M_{2,2,5} split into M_{2,2,2} + M_{2,2,2} + M_{2,2,1}
    # M_{2,2,2} uses Strassen (rank 7)
    # M_{2,2,1} uses rank 1
    # M_{2,1,5} uses rank 1
    
    # Total rank: 15 + 15 + 1 = 31
    
    rank = 31
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    current_rank = 0
    
    def add_strassen_222(offset_i, offset_j, offset_k, rank_offset):
        # Adds a Strassen decomposition for M_{2,2,2}
        # A is 2x2 at A[offset_i:offset_i+2, offset_j:offset_j+2]
        # B is 2x2 at B[offset_j:offset_j+2, offset_k:offset_k+2]
        # C is 2x2 at C[offset_i:offset_i+2, offset_k:offset_k+2]
        
        # Strassen formulas for M_{2,2,2}
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B10-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A indices)
        # A00 is at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
        # Indices in U: (i*m + j)
        
        u_coefs = [
            [1, 0, 1, 0], # M1: A00, A11
            [0, 0, 1, 1], # M2: A10, A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00, A01
            [1, 0, -1, 0],# M6: A10, A00
            [0, 1, 0, -1] # M7: A01, A11
        ]
        
        v_coefs = [
            [1, 1, 0, 0], # M1: B00, B11
            [1, 0, 0, 0], # M2: B00
            [0, 0, 1, -1],# M3: B11, B00
            [0, 1, -1, 0],# M4: B10, B01
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00, B01
            [0, 1, 1, 1]  # M7: B10, B11
        ]
        
        w_coefs = [
            [1, 0, 0, -1],# C00, C01, C10, C11
            [-1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # Note: The W coefficients correspond to the linear combinations to recover C.
        # C00 = M1 - M3 - M4 + M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Correct W coefs:
        w_coefs = [
            [1, 0, 0, -1], # C00
            [0, 1, 0, 1],  # C01
            [0, 0, 1, 0],  # C10
            [1, -1, 1, 1]  # C11
        ]
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    idx_a = (offset_i + i) * m + (offset_j + j)
                    U[idx_a, rank_offset + r] += u_coefs[r][i*2+j]
            
            # V
            for j in range(2):
                for k in range(2):
                    idx_b = (offset_j + j) * p + (offset_k + k)
                    V[idx_b, rank_offset + r] += v_coefs[r][j*2+k]
            
            # W
            for i in range(2):
                for k in range(2):
                    idx_c = (offset_k + k) * n + (offset_i + i)
                    W[idx_c, rank_offset + r] += w_coefs[i*2+k][r]
        
        return rank_offset + 7

    def add_rank1_221(offset_i, offset_j, offset_k, rank_offset):
        # M_{2,2,1} rank 1
        # C_i0 = A_i0 B_00 + A_i1 B_10
        # T_{i,j,0} = 1
        # U_{i,0} = 1, V_{j,0} = 1, W_{0,0} = 1
        for i in range(2):
            idx_a = (offset_i + i) * m + offset_j
            U[idx_a, rank_offset] = 1
            idx_a2 = (offset_i + i) * m + (offset_j + 1)
            U[idx_a2, rank_offset] = 1
        
        for j in range(2):
            idx_b = (offset_j + j) * p + offset_k
            V[idx_b, rank_offset] = 1
        
        for i in range(2):
            idx_c = offset_k * n + (offset_i + i)
            W[idx_c, rank_offset] = 1
            
        return rank_offset + 1

    def add_rank1_215(offset_i, offset_j, offset_k, rank_offset):
        # M_{2,1,5} rank 1
        # C_i,k = A_i0 B_0k
        # T_{i,0,k} = 1
        for i in range(2):
            idx_a = (offset_i + i) * m + offset_j
            U[idx_a, rank_offset] = 1
        
        idx_b = offset_j * p + offset_k
        V[idx_b, rank_offset] = 1
        
        for k in range(5):
            idx_c = (offset_k + k) * n + offset_i
            W[idx_c, rank_offset] = 1
            
        return rank_offset + 1

    # Term 1: A1 (2x2) * B1 (2x5)
    # Split B1 into B11 (2x2), B12 (2x2), B13 (2x1)
    # Offsets: i=0, j=0, k=0
    current_rank = add_strassen_222(0, 0, 0, current_rank)
    current_rank = add_strassen_222(0, 0, 2, current_rank)
    current_rank = add_rank1_221(0, 0, 4, current_rank)
    
    # Term 2: A2 (2x2) * B2 (2x5)
    # Offsets: i=0, j=2, k=0
    current_rank = add_strassen_222(0, 2, 0, current_rank)
    current_rank = add_strassen_222(0, 2, 2, current_rank)
    current_rank = add_rank1_221(0, 2, 4, current_rank)
    
    # Term 3: A3 (2x1) * B3 (1x5)
    # Offsets: i=0, j=4, k=0
    current_rank = add_rank1_215(0, 4, 0, current_rank)
    
    # Verify
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
