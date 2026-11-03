import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 decomposition (rank 49)
    
    def get_A_block(br, bc):
        indices = []
        for r in range(2):
            row = []
            for c in range(2):
                i = br * 2 + r
                j = bc * 2 + c
                row.append(i * m + j)
            indices.append(row)
        return indices

    def get_B_block(br, bc):
        indices = []
        for r in range(2):
            row = []
            for c in range(2):
                j = br * 2 + r
                k = bc * 2 + c
                row.append(j * p + k)
            indices.append(row)
        return indices

    M_defs = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),          # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),         # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),         # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),          # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),# M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]) # M7
    ]
    
    T_coeffs = {
        'z00': [1, 0, 0, 1, -1, 0, 1],
        'z01': [0, 0, 1, 0, 1, 0, 0],
        'z10': [0, 1, 0, 1, 0, 0, 0],
        'z11': [1, -1, 1, 0, 0, 1, 0]
    }
    
    M_to_C = {
        0: [('C11', 'z00'), ('C22', 'z11')], # M1
        1: [('C21', 'z10')],                   # M2
        2: [('C12', 'z01'), ('C22', 'z11')],   # M3
        3: [('C11', 'z00'), ('C21', 'z10')],   # M4
        4: [('C12', 'z01'), ('C11', 'z00')],   # M5
        5: [('C22', 'z11')],                   # M6
        6: [('C11', 'z00')]                    # M7
    }
    
    C_block_indices = {
        'C11': [(i, k) for i in range(2) for k in range(2)],
        'C12': [(i, k) for i in range(2) for k in range(2, 4)],
        'C21': [(i, k) for i in range(2, 4) for k in range(2)],
        'C22': [(i, k) for i in range(2, 4) for k in range(2, 4)]
    }
    
    col = 0
    for m_idx, (x_blocks, y_blocks) in enumerate(M_defs):
        X = [[{} for _ in range(2)] for _ in range(2)]
        for br, bc, coeff in x_blocks:
            a_indices = get_A_block(br, bc)
            for r in range(2):
                for c in range(2):
                    a_idx = a_indices[r][c]
                    if a_idx not in X[r][c]:
                        X[r][c][a_idx] = 0
                    X[r][c][a_idx] += coeff
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for br, bc, coeff in y_blocks:
            b_indices = get_B_block(br, bc)
            for r in range(2):
                for c in range(2):
                    b_idx = b_indices[r][c]
                    if b_idx not in Y[r][c]:
                        Y[r][c][b_idx] = 0
                    Y[r][c][b_idx] += coeff
        
        for t_idx in range(7):
            L_A = {}
            L_B = {}
            
            if t_idx == 0:
                for r in [0, 1]:
                    for c in [0, 1]:
                        if (r,c) == (0,0) or (r,c) == (1,1):
                            for a_idx, coeff in X[r][c].items():
                                L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                            for b_idx, coeff in Y[r][c].items():
                                L_B[b_idx] = L_B.get(b_idx, 0) + coeff
            elif t_idx == 1:
                for r in [1]:
                    for c in [0, 1]:
                        for a_idx, coeff in X[r][c].items():
                            L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for b_idx, coeff in Y[0][0].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) + coeff
            elif t_idx == 2:
                for a_idx, coeff in X[0][0].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for b_idx, coeff in Y[0][1].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) + coeff
                for b_idx, coeff in Y[1][1].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) - coeff
            elif t_idx == 3:
                for a_idx, coeff in X[1][1].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for b_idx, coeff in Y[1][0].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) + coeff
                for b_idx, coeff in Y[0][0].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) - coeff
            elif t_idx == 4:
                for c in [0, 1]:
                    for a_idx, coeff in X[0][c].items():
                        L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for b_idx, coeff in Y[1][1].items():
                    L_B[b_idx] = L_B.get(b_idx, 0) + coeff
            elif t_idx == 5:
                for a_idx, coeff in X[1][0].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for a_idx, coeff in X[0][0].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) - coeff
                for c in [0, 1]:
                    for b_idx, coeff in Y[0][c].items():
                        L_B[b_idx] = L_B.get(b_idx, 0) + coeff
            elif t_idx == 6:
                for a_idx, coeff in X[0][1].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) + coeff
                for a_idx, coeff in X[1][1].items():
                    L_A[a_idx] = L_A.get(a_idx, 0) - coeff
                for c in [0, 1]:
                    for b_idx, coeff in Y[1][c].items():
                        L_B[b_idx] = L_B.get(b_idx, 0) + coeff
            
            if not L_A or not L_B:
                continue
                
            W_vec = np.zeros(p * n)
            valid = False
            for c_block_name, z_pos in M_to_C[m_idx]:
                coeff_t = T_coeffs[z_pos][t_idx]
                if coeff_t == 0:
                    continue
                valid = True
                for i, k in C_block_indices[c_block_name]:
                    w_idx = k * n + i
                    W_vec[w_idx] += coeff_t
            
            if not valid:
                continue
                
            for a_idx, coeff_a in L_A.items():
                U[a_idx, col] = coeff_a
            for b_idx, coeff_b in L_B.items():
                V[b_idx, col] = coeff_b
            W[:, col] = W_vec
            
            col += 1
            
    # Part 2: 4x4x1 decomposition (rank 16)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
