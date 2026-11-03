import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = 4, 4, 9
    
    # 1. Define Strassen 2x2x2 terms
    # u, v, w are lists of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for U/V
    # W Indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -> k*2+i
    # k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3
    
    strassen_2x2 = [
        # M1
        ([1,0,0,0], [0,1,0,-1], [1,0,0,1]),
        # M2
        ([1,0,1,0], [0,0,0,1], [0,1,0,-1]),
        # M3
        ([0,1,0,-1], [0,0,1,1], [0,0,1,1]),
        # M4
        ([0,0,0,1], [1,-1,0,0], [1,1,0,0]),
        # M5
        ([1,1,0,0], [1,0,0,0], [-1,0,1,0]),
        # M6
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7
        ([0,0,-1,1], [0,0,1,-1], [1,0,0,0]),
    ]
    
    # 2. Define Outer Strassen Patterns for 4x4x4
    # Each pattern is a dict with 'A', 'B', 'C' keys
    # Values are dicts: (block_i, block_j): coeff
    outer_patterns = [
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(0,0): 1, (1,0): 1}, 'B': {(1,1): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(0,0): 1, (0,1): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(0,0): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(1,1): 1, (1,0): -1}, 'B': {(1,0): 1, (1,1): -1}, 'C': {(0,0): 1}},
    ]
    
    def generate_4x4x4_terms(k_offset):
        terms = []
        for pattern in outer_patterns:
            A_map = pattern['A']
            B_map = pattern['B']
            C_map = pattern['C']
            for u_inner, v_inner, w_inner in strassen_2x2:
                u_term = [0.0] * 16
                v_term = [0.0] * 16
                w_term = [0.0] * 16
                
                # Expand U
                for (bi, bj), c_a in A_map.items():
                    for i_prime in range(2):
                        for j_prime in range(2):
                            c_u = u_inner[i_prime * 2 + j_prime]
                            if c_u != 0:
                                i = 2 * bi + i_prime
                                j = 2 * bj + j_prime
                                idx = i * 4 + j
                                u_term[idx] += c_a * c_u
                                
                # Expand V
                for (bj, bk), c_b in B_map.items():
                    for j_prime in range(2):
                        for k_prime in range(2):
                            c_v = v_inner[j_prime * 2 + k_prime]
                            if c_v != 0:
                                j = 2 * bj + j_prime
                                k = 2 * bk + k_prime + k_offset
                                idx = j * 9 + k
                                v_term[idx] += c_b * c_v
                                
                # Expand W
                for (bk, bi), c_c in C_map.items():
                    for k_prime in range(2):
                        for i_prime in range(2):
                            c_w = w_inner[k_prime * 2 + i_prime]
                            if c_w != 0:
                                k = 2 * bk + k_prime + k_offset
                                i = 2 * bi + i_prime
                                idx = k * 4 + i
                                w_term[idx] += c_c * c_w
                                
                terms.append((u_term, v_term, w_term))
        return terms

    # Generate terms for the two 4x4x4 blocks
    terms_1 = generate_4x4x4_terms(0)
    terms_2 = generate_4x4x4_terms(4)
    
    # Generate terms for the 4x4x1 block (schoolbook)
    terms_3 = []
    for i in range(4):
        for j in range(4):
            for k in range(1): # k is 0 locally, global 8
                u_term = [0.0] * 16
                v_term = [0.0] * 16
                w_term = [0.0] * 16
                u_term[i * 4 + j] = 1.0
                v_term[j * 9 + 8] = 1.0
                w_term[8 * 4 + i] = 1.0
                terms_3.append((u_term, v_term, w_term))
                
    all_terms = terms_1 + terms_2 + terms_3
    rank = len(all_terms)
    
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for r, (u, v, w) in enumerate(all_terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
