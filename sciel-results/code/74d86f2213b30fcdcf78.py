import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 with rank 49
    # 2x2x2 Strassen terms
    u2 = [
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0]
    ]
    v2 = [
        [1, 0, 0, 1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, -1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    w2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [1, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # 4x4x4 Strassen terms
    # (A_blocks, B_blocks, C_blocks)
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    s4 = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(3, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
        ([(0, 1)], [(2, 1), (0, -1)], [(0, 1)]),
        ([(2, 1), (0, 1)], [(0, 1)], [(0, -1), (2, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)]),
        ([(0, 1), (2, -1)], [(0, 1), (1, 1)], [(0, 1)])
    ]
    
    t = 0
    for k in range(7):
        a_blocks, b_blocks, c_blocks = s4[k]
        for l in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            for b_idx, coeff in a_blocks:
                # A block indices
                if b_idx == 0: idxs = [0, 1, 4, 5]
                elif b_idx == 1: idxs = [2, 3, 6, 7]
                elif b_idx == 2: idxs = [8, 9, 12, 13]
                else: idxs = [10, 11, 14, 15]
                for i in range(4):
                    u_vec[idxs[i]] += coeff * u2[l][i]
            
            for b_idx, coeff in b_blocks:
                if b_idx == 0: idxs = [0, 1, 4, 5]
                elif b_idx == 1: idxs = [2, 3, 6, 7]
                elif b_idx == 2: idxs = [8, 9, 12, 13]
                else: idxs = [10, 11, 14, 15]
                for i in range(4):
                    v_vec[idxs[i]] += coeff * v2[l][i]
            
            for b_idx, coeff in c_blocks:
                if b_idx == 0: idxs = [0, 1, 4, 5]
                elif b_idx == 1: idxs = [2, 3, 6, 7]
                elif b_idx == 2: idxs = [8, 9, 12, 13]
                else: idxs = [10, 11, 14, 15]
                for i in range(4):
                    w_vec[idxs[i]] += coeff * w2[l][i]
            
            for i in range(16):
                U[i][t] = u_vec[i]
                V[i][t] = v_vec[i]
                W[i][t] = w_vec[i]
            t += 1
            
    # Part 2: 4x4x3 with rank 44
    # 2x2x3 decomposition: 7 Strassen (cols 0,1) + 4 Trivial (col 2)
    u23 = []
    v23 = []
    w23 = []
    
    # Strassen part (cols 0,1)
    for l in range(7):
        u_vec = u2[l]
        v_vec = [0] * 6
        v_vec[0] = v2[l][0]
        v_vec[1] = v2[l][1]
        v_vec[3] = v2[l][2]
        v_vec[4] = v2[l][3]
        w_vec = [0] * 6
        w_vec[0] = w2[l][0]
        w_vec[1] = w2[l][1]
        w_vec[2] = w2[l][2]
        w_vec[3] = w2[l][3]
        u23.append(u_vec)
        v23.append(v_vec)
        w23.append(w_vec)
        
    # Trivial part (col 2)
    # 1. A00 B02 -> C02
    u23.append([1, 0, 0, 0])
    v23.append([0, 0, 1, 0, 0, 0])
    w23.append([0, 0, 0, 0, 1, 0])
    # 2. A01 B12 -> C02
    u23.append([0, 1, 0, 0])
    v23.append([0, 0, 0, 0, 0, 1])
    w23.append([0, 0, 0, 0, 1, 0])
    # 3. A10 B02 -> C12
    u23.append([0, 0, 1, 0])
    v23.append([0, 0, 1, 0, 0, 0])
    w23.append([0, 0, 0, 0, 0, 1])
    # 4. A11 B12 -> C12
    u23.append([0, 0, 0, 1])
    v23.append([0, 0, 0, 0, 0, 1])
    w23.append([0, 0, 0, 0, 0, 1])
    
    # 4x4x3 terms
    # (A_block, B_block, C_block_type)
    # C_block_type: 0 for top, 1 for bottom
    s3 = [
        (0, 0, 0), # A11 B11 -> C_top
        (1, 1, 0), # A12 B21 -> C_top
        (2, 0, 1), # A21 B11 -> C_bot
        (3, 1, 1)  # A22 B21 -> C_bot
    ]
    
    for a_block, b_block, c_type in s3:
        for l in range(11):
            u_vec = [0] * 16
            v_vec = [0] * 28
            w_vec = [0] * 12
            
            # A block
            if a_block == 0: idxs = [0, 1, 4, 5]
            elif a_block == 1: idxs = [2, 3, 6, 7]
            elif a_block == 2: idxs = [8, 9, 12, 13]
            else: idxs = [10, 11, 14, 15]
            for i in range(4):
                u_vec[idxs[i]] = u23[l][i]
                
            # B block
            if b_block == 0: offset = 0
            else: offset = 6
            for i in range(6):
                v_vec[offset + i] = v23[l][i]
                
            # C block
            if c_type == 0:
                # Top: 0->0, 1->1, 2->4, 3->5, 4->8, 5->9
                w_vec[0] = w23[l][0]
                w_vec[1] = w23[l][1]
                w_vec[4] = w23[l][2]
                w_vec[5] = w23[l][3]
                w_vec[8] = w23[l][4]
                w_vec[9] = w23[l][5]
            else:
                # Bottom: 0->2, 1->3, 2->6, 3->7, 4->10, 5->11
                w_vec[2] = w23[l][0]
                w_vec[3] = w23[l][1]
                w_vec[6] = w23[l][2]
                w_vec[7] = w23[l][3]
                w_vec[10] = w23[l][4]
                w_vec[11] = w23[l][5]
                
            for i in range(16):
                U[i][t] = u_vec[i]
            for i in range(28):
                V[i][t] = v_vec[i]
            for i in range(12):
                W[i][t] = w_vec[i]
            t += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
