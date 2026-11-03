import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Try to construct a rank-30 decomposition using tensor product structure
    # 2x4x6 = 2 * (1x4x6) where 1x4x6 has rank 15
    
    # For 1x4x6: C[k] = sum_{j=0}^{3} A[j]*B[j,k]
    # Split into two 1x2x6 problems:
    # C1[k] = A[0]*B[0,k] + A[1]*B[1,k]  (rank 6)
    # C2[k] = A[2]*B[2,k] + A[3]*B[3,k]  (rank 6)
    # Total: 12 terms for 1x4x6, so 24 for 2x4x6.
    # But we can do better with Strassen-like tricks on the 1x2x2 sub-blocks.
    
    # Let me try a rank-36 construction first (6*6) and see if I can reduce.
    # Actually, let me just use the known working rank-42 solution.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):
        for c in range(3):
            for t in range(7):
                p_factors, q_factors, r_factors = strassen_terms[t]
                for (pi, pj), coeff in p_factors:
                    u_row = pi * m + (pj + 2 * source)
                    U[u_row, col_idx] += coeff
                for (qi, qj), coeff in q_factors:
                    v_row = (qi + 2 * source) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                for (ri, rj), coeff in r_factors:
                    w_row = (2 * c + rj) * n + ri
                    W[w_row, col_idx] += coeff
                col_idx += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-6:
        # Fallback to hardcoded
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        steps = [
            ({(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
             {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
             [(0, 1), (3, 1)]),
            ({(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
             {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
             [(2, 1), (3, -1)]),
            ({(0,0): 1, (0,1): 1},
             {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
             [(1, 1), (3, 1)]),
            ({(1,2): 1, (1,3): 1},
             {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
             [(0, 1), (2, 1)]),
            ({(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
             {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
             [(0, -1), (1, 1)]),
            ({(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
             {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
             [(3, 1)]),
            ({(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
             {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
             [(0, 1)]),
        ]
        
        col = 0
        for X_dict, Y_dict, C_targets in steps:
            x0_coeffs = {}
            x1_coeffs = {}
            for (i, j), c in X_dict.items():
                if j % 2 == 0:
                    x0_coeffs[(i, j)] = c
                else:
                    x1_coeffs[(i, j)] = c
            y_coeffs = [[], [], [], [], [], []]
            for (j, k), c in Y_dict.items():
                r_block = j % 2
                c_block = k % 3
                idx = r_block * 3 + c_block
                y_coeffs[idx].append(((j, k), c))
            sub_terms = [
                (x0_coeffs, y_coeffs[0], 0),
                (x1_coeffs, y_coeffs[3], 0),
                (x0_coeffs, y_coeffs[1], 1),
                (x1_coeffs, y_coeffs[4], 1),
                (x0_coeffs, y_coeffs[2], 2),
                (x1_coeffs, y_coeffs[5], 2),
            ]
            for X_c, Y_c, z_idx in sub_terms:
                for (i, j), c in X_c.items():
                    U[i * m + j, col] += c
                for (j, k), c in Y_c:
                    V[j * p + k, col] += c
                for block_idx, sign in C_targets:
                    if block_idx == 0:
                        i_C, k_C = 0, z_idx
                    elif block_idx == 1:
                        i_C, k_C = 0, 3 + z_idx
                    elif block_idx == 2:
                        i_C, k_C = 1, z_idx
                    elif block_idx == 3:
                        i_C, k_C = 1, 3 + z_idx
                    W[k_C * n + i_C, col] += sign
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
