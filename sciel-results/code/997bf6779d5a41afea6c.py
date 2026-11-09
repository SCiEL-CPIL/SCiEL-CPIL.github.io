import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    R = 28
    
    # Initialize matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Local Strassen matrices (4x7)
    # U_loc[i, r], V_loc[j, r], W_loc[k, r]
    # Indices 0..3 for 2x2 blocks
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 4 products
    # P1: A1 B11 -> C11
    # P2: A2 B21 -> C21
    # P3: A1 B12 -> C12
    # P4: A2 B22 -> C22
    
    # Mappings:
    # A index: i (0,1), j (0,1,2,3)
    # B index: k (0,1,2,3), l (0,1,2,3)
    # C index: i (0,1), l (0,1,2,3)
    
    # P1: A1 (j 0,1), B11 (k 0,1, l 0,1), C11 (l 0,1)
    # P2: A2 (j 2,3), B21 (k 2,3, l 0,1), C21 (l 0,1)
    # P3: A1 (j 0,1), B12 (k 0,1, l 2,3), C12 (l 2,3)
    # P4: A2 (j 2,3), B22 (k 2,3, l 2,3), C22 (l 2,3)
    
    products = [
        {
            'A_offset_j': 0, # j = 0 + j_loc
            'B_offset_k': 0, # k = 0 + k_loc
            'B_offset_l': 0, # l = 0 + l_loc
            'C_offset_l': 0, # l = 0 + l_loc
            'col_start': 0
        },
        {
            'A_offset_j': 2, # j = 2 + j_loc
            'B_offset_k': 2, # k = 2 + k_loc
            'B_offset_l': 0, # l = 0 + l_loc
            'C_offset_l': 0, # l = 0 + l_loc
            'col_start': 7
        },
        {
            'A_offset_j': 0, # j = 0 + j_loc
            'B_offset_k': 0, # k = 0 + k_loc
            'B_offset_l': 2, # l = 2 + l_loc
            'C_offset_l': 2, # l = 2 + l_loc
            'col_start': 14
        },
        {
            'A_offset_j': 2, # j = 2 + j_loc
            'B_offset_k': 2, # k = 2 + k_loc
            'B_offset_l': 2, # l = 2 + l_loc
            'C_offset_l': 2, # l = 2 + l_loc
            'col_start': 21
        }
    ]
    
    for prod in products:
        a_off_j = prod['A_offset_j']
        b_off_k = prod['B_offset_k']
        b_off_l = prod['B_offset_l']
        c_off_l = prod['C_offset_l']
        c_start = prod['col_start']
        
        for r_loc in range(7):
            r_glob = c_start + r_loc
            
            for i in range(2):
                for j_loc in range(2):
                    j_glob = a_off_j + j_loc
                    U[i * 4 + j_glob][r_glob] = U_loc[i * 2 + j_loc][r_loc]
                    
            for k_loc in range(2):
                for l_loc in range(2):
                    k_glob = b_off_k + k_loc
                    l_glob = b_off_l + l_loc
                    V[k_glob * 4 + l_glob][r_glob] = V_loc[k_loc * 2 + l_loc][r_loc]
                    
            for i in range(2):
                for l_loc in range(2):
                    l_glob = c_off_l + l_loc
                    W[l_glob * 2 + i][r_glob] = W_loc[l_loc * 2 + i][r_loc]
                    
    return {"U": U, "V": V, "W": W}
