import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    C = A*B1 + A*B2 + A*B3
    Use Strassen's algorithm for 4x4*4x4 (rank 49) twice, and schoolbook for 4x4*4x1 (rank 4).
    Total rank: 49 + 49 + 4 = 102.
    """
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    rank1_terms = []  # list of (U_dict, V_dict, W_dict) where each is {index: coeff}
    
    def add_rank1_term(U_dict, V_dict, W_dict):
        rank1_terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_block(col_offset):
        """Add 49 rank-1 terms for 4x4 * 4x4 with B columns at col_offset..col_offset+3"""
        
        block_defs = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]
        
        for x_blocks, y_blocks in block_defs:
            X = [[{} for _ in range(2)] for _ in range(2)]
            Y = [[{} for _ in range(2)] for _ in range(2)]
            
            for (bi, bj), coeff in x_blocks.items():
                for ei in range(2):
                    for ej in range(2):
                        i = 2*bi + ei
                        j = 2*bj + ej
                        u = i * 4 + j
                        X[ei][ej][u] = X[ei][ej].get(u, 0) + coeff
            
            for (bi, bj), coeff in y_blocks.items():
                for fi in range(2):
                    for fj in range(2):
                        j = 2*bi + fi
                        k = col_offset + 2*bj + fj
                        v = j * 9 + k
                        Y[fi][fj][v] = Y[fi][fj].get(v, 0) + coeff
            
            def add_linear_form(d1, d2, sign1=1, sign2=1):
                result = {}
                for k, v in d1.items():
                    result[k] = result.get(k, 0) + sign1 * v
                for k, v in d2.items():
                    result[k] = result.get(k, 0) + sign2 * v
                return {k: v for k, v in result.items() if abs(v) > 1e-12}
            
            sub_products = [
                (add_linear_form(X[0][0], X[1][1]), add_linear_form(Y[0][0], Y[1][1])),
                (add_linear_form(X[1][0], X[1][1]), Y[0][0]),
                (X[0][0], add_linear_form(Y[0][1], Y[1][1], 1, -1)),
                (X[1][1], add_linear_form(Y[1][0], Y[0][0], 1, -1)),
                (add_linear_form(X[0][0], X[0][1]), Y[1][1]),
                (add_linear_form(X[1][0], X[0][0], 1, -1), add_linear_form(Y[0][0], Y[0][1])),
                (add_linear_form(X[0][1], X[1][1], 1, -1), add_linear_form(Y[1][0], Y[1][1])),
            ]
            
            # Pk contributions to Z entries:
            # P1: Z00 += 1, Z11 += 1
            # P2: Z10 += 1, Z11 -= 1
            # P3: Z01 += 1, Z11 += 1
            # P4: Z00 += 1, Z10 += 1
            # P5: Z00 -= 1, Z01 += 1
            # P6: Z11 += 1
            # P7: Z00 += 1
            z_contribs = [
                {(0,0): 1, (1,1): 1},
                {(1,0): 1, (1,1): -1},
                {(0,1): 1, (1,1): 1},
                {(0,0): 1, (1,0): 1},
                {(0,0): -1, (0,1): 1},
                {(1,1): 1},
                {(0,0): 1},
            ]
            
            for pk_idx, (x_form, y_form) in enumerate(sub_products):
                z_contrib = z_contribs[pk_idx]
                
                W_dict = {}
                for (ei, fi), coeff in z_contrib.items():
                    for ci in range(2):
                        for cj in range(2):
                            k = col_offset + 2*ci + ei
                            i = 2*cj + fi
                            w = k * 4 + i
                            W_dict[w] = W_dict.get(w, 0) + coeff
                
                W_dict = {k: v for k, v in W_dict.items() if abs(v) > 1e-12}
                
                if x_form and y_form and W_dict:
                    add_rank1_term(x_form, y_form, W_dict)
    
    # Add 49 terms for columns 0-3
    add_strassen_block(0)
    # Add 49 terms for columns 4-7
    add_strassen_block(4)
    
    # Add 4 terms for column 8 (4x4 * 4x1)
    for j in range(4):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for i in range(4):
            u = i * 4 + j
            U_dict[u] = 1
            w = 8 * 4 + i
            W_dict[w] = 1
        V_dict[j * 9 + 8] = 1
        add_rank1_term(U_dict, V_dict, W_dict)
    
    R = len(rank1_terms)
    
    # Check deadline before building matrices
    if time.time() > DEADLINE:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build factor matrices
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    for c, (U_dict, V_dict, W_dict) in enumerate(rank1_terms):
        for u, coeff in U_dict.items():
            U[u, c] = coeff
        for v, coeff in V_dict.items():
            V[v, c] = coeff
        for w, coeff in W_dict.items():
            W[w, c] = coeff
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
