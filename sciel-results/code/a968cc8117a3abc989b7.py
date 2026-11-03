import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen 2x2 x 2x2 coefficients
    # M1 = (P00+P11)*(Q00+Q11) -> R00 + R11
    # M2 = (P10+P11)*Q00 -> R10 - R11
    # M3 = P00*(Q01-Q11) -> R01 + R11
    # M4 = P11*(Q10-Q00) -> R00 + R10
    # M5 = (P00+P01)*Q11 -> -R00 + R01
    # M6 = (P10-P00)*(Q00+Q01) -> R11
    # M7 = (P01-P11)*(Q10+Q11) -> R00
    
    strassen_terms = [
        # (P_factors, Q_factors, R_factors)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    # Try to find a lower rank decomposition
    # The key insight: we can potentially share terms across the 3 column blocks
    # by combining the Strassen computations more cleverly
    
    # Approach: Try rank 36 by using 5 Strassen-like terms per 2x4*4x2 subproblem
    # instead of 14 (7+7), or find a global structure
    
    # For now, use the known rank-42 solution as baseline
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_idx = 0
    for source in range(2):  # 0=top (A[0:2,:]), 1=bot (A[2:4,:])
        for c in range(3):  # column block 0,1,2
            for t in range(7):  # 7 Strassen terms
                p_factors, q_factors, r_factors = strassen_terms[t]
                
                # Map P indices to U indices:
                # P[pi, pj] = A[pi, pj + 2*source]
                # U index: pi * m + (pj + 2*source)
                for (pi, pj), coeff in p_factors:
                    u_row = pi * m + (pj + 2 * source)
                    U[u_row, col_idx] += coeff
                
                # Map Q indices to V indices:
                # Q[qi, qj] = B[qi + 2*source, 2*c + qj]
                # V index: (qi + 2*source) * p + (2*c + qj)
                for (qi, qj), coeff in q_factors:
                    v_row = (qi + 2 * source) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                
                # Map R indices to W indices:
                # R[ri, rj] contributes to C[ri, 2*c + rj]
                # W index: (2*c + rj) * n + ri
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
        # Fallback: try the original hardcoded approach
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        steps = [
            (
                {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1), (3, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
                [(2, 1), (3, -1)]
            ),
            (
                {(0,0): 1, (0,1): 1},
                {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
                [(1, 1), (3, 1)]
            ),
            (
                {(1,2): 1, (1,3): 1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
                [(0, 1), (2, 1)]
            ),
            (
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
                {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, -1), (1, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
                [(3, 1)]
            ),
            (
                {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1)]
            ),
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
                        i_C = 0
                        k_C = z_idx
                    elif block_idx == 1:
                        i_C = 0
                        k_C = 3 + z_idx
                    elif block_idx == 2:
                        i_C = 1
                        k_C = z_idx
                    elif block_idx == 3:
                        i_C = 1
                        k_C = 3 + z_idx
                    
                    W[k_C * n + i_C, col] += sign
                
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
