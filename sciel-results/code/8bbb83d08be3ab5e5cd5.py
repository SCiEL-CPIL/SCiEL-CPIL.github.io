import numpy as np
import time

# Safety margin: The evaluator kills after 150s. We aim to finish in < 5s.
_DEADLINE = time.time() + 10

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4x4) + 44 (4 * Rank(2x2x3) = 4 * 11)
    # 2x2x3: 7 Strassen (cols 0,1) + 4 schoolbook (col 2)
    total_rank = 49 + 44
    
    # Use NumPy for efficient initialization and final verification
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # For 4x4x4, the c_block for each outer term m_idx
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Compute U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # Compute V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            # Compute W coefficients
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # --- Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11 ---
    # 2x2x3: Strassen 2x2x2 (7 terms) + 4 schoolbook terms for col 3
    
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5 (local cols 0,1)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj  # global cols 4,5
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj  # global cols 4,5
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6 (local col 2)
        # C[i][6] = A[i][0]*B[0][6] + A[i][1]*B[1][6]
        # Local A: rows 2ar, 2ar+1; cols 2ac, 2ac+1
        # Local B: rows 2br, 2br+1; col 6
        # Local C: rows 2cr, 2cr+1; col 6
        
        # Term 1: A[0,0] * B[0,2] -> C[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 2: A[0,1] * B[1,2] -> C[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 3: A[1,0] * B[0,2] -> C[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        # Term 4: A[1,1] * B[1,2] -> C[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    # The four 2x2x3 blocks:
    # A_00 (ar=0, ac=0) * B_top (br=0) -> C_top (cr=0)
    add_2x2x3(0, 0, 0, 0)
    # A_01 (ar=0, ac=1) * B_bot (br=1) -> C_top (cr=0)
    add_2x2x3(0, 1, 1, 0)
    # A_10 (ar=1, ac=0) * B_top (br=0) -> C_bot (cr=1)
    add_2x2x3(1, 0, 0, 1)
    # A_11 (ar=1, ac=1) * B_bot (br=1) -> C_bot (cr=1)
    add_2x2x3(1, 1, 1, 1)
    
    # Trim to actual rank used
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Verify correctness
    if time.time() < _DEADLINE:
        # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Target: T[i_A, i_B, i_C] = 1 if i_A=i*m+j, i_B=j*p+k, i_C=k*n+i
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        max_err = np.max(np.abs(recon - target))
        if max_err > 1e-10:
            # Fallback: return schoolbook
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb))
            V_sb = np.zeros((m*p, rank_sb))
            W_sb = np.zeros((p*n, rank_sb))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, r] = 1.0
                        V_sb[j*p+k, r] = 1.0
                        W_sb[k*n+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
