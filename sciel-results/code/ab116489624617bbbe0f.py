import numpy as np
import time

DEADLINE_SECONDS = 115.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Block contributions
    # M0 (M1): C00:+1, C11:+1
    # M1 (M2): C10:+1, C11:-1
    # M2 (M3): C01:+1, C11:+1
    # M3 (M4): C00:+1, C10:+1
    # M4 (M5): C00:-1, C01:+1
    # M5 (M6): C11:+1
    # M6 (M7): C00:+1
    
    block_contrib = np.array([
        [1, 0, 0, 1],   
        [0, 0, 1, -1],  
        [0, 1, 0, 1],   
        [1, 0, 1, 0],   
        [-1, 1, 0, 0],  
        [0, 0, 0, 1],   
        [1, 0, 0, 0],   
    ], dtype=int)
    
    # Scalar contributions within M_b
    # s=0: [0,0]:+1, [1,1]:+1
    # s=1: [1,0]:+1, [1,1]:-1
    # s=2: [0,1]:+1, [1,1]:+1
    # s=3: [0,0]:+1, [1,0]:+1
    # s=4: [0,0]:-1, [0,1]:+1
    # s=5: [1,1]:+1
    # s=6: [0,0]:+1
    
    # pos: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    scalar_contrib = np.array([
        [1, 0, 0, 1],   
        [0, 0, 1, -1],  
        [0, 1, 0, 1],   
        [1, 0, 1, 0],   
        [-1, 1, 0, 0],  
        [0, 0, 0, 1],   
        [1, 0, 0, 0],   
    ], dtype=int)
    
    c_block_offsets = np.array([(0, 0), (0, 2), (2, 0), (2, 2)])
    
    # Build W
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            for pos in range(4):
                sc_coeff = scalar_contrib[s, pos]
                if sc_coeff == 0:
                    continue
                
                if pos == 0:
                    local_r, local_c = 0, 0
                elif pos == 1:
                    local_r, local_c = 0, 1
                elif pos == 2:
                    local_r, local_c = 1, 0
                else:
                    local_r, local_c = 1, 1
                
                for cb in range(4):
                    bc_coeff = block_contrib[b, cb]
                    if bc_coeff == 0:
                        continue
                    row_off, col_off = c_block_offsets[cb]
                    i = row_off + local_r
                    k = col_off + local_c
                    W[k * n + i, r] += bc_coeff * sc_coeff

    # Define L_defs and R_defs as lists of dicts for flexibility, then convert
    # Or just hardcode the U and V filling logic with numpy.
    
    L_defs = [
        [ {(0,0): 1, (2,2): 1}, {(0,1): 1, (2,3): 1}, {(1,0): 1, (3,2): 1}, {(1,1): 1, (3,3): 1} ],
        [ {(2,0): 1, (2,2): 1}, {(2,1): 1, (2,3): 1}, {(3,0): 1, (3,2): 1}, {(3,1): 1, (3,3): 1} ],
        [ {(0,0): 1}, {(0,1): 1}, {(1,0): 1}, {(1,1): 1} ],
        [ {(2,2): 1}, {(2,3): 1}, {(3,2): 1}, {(3,3): 1} ],
        [ {(0,0): 1, (0,2): 1}, {(0,1): 1, (0,3): 1}, {(1,0): 1, (1,2): 1}, {(1,1): 1, (1,3): 1} ],
        [ {(2,0): 1, (0,0): -1}, {(2,1): 1, (0,1): -1}, {(3,0): 1, (1,0): -1}, {(3,1): 1, (1,1): -1} ],
        [ {(0,2): 1, (2,2): -1}, {(0,3): 1, (2,3): -1}, {(1,2): 1, (3,2): -1}, {(1,3): 1, (3,3): -1} ],
    ]
    
    R_defs = [
        [ {(0,0): 1, (2,2): 1}, {(0,1): 1, (2,3): 1}, {(1,0): 1, (3,2): 1}, {(1,1): 1, (3,3): 1} ],
        [ {(0,0): 1}, {(0,1): 1}, {(1,0): 1}, {(1,1): 1} ],
        [ {(0,2): 1, (2,2): -1}, {(0,3): 1, (2,3): -1}, {(1,2): 1, (3,2): -1}, {(1,3): 1, (3,3): -1} ],
        [ {(2,0): 1, (0,0): -1}, {(2,1): 1, (0,1): -1}, {(3,0): 1, (1,0): -1}, {(3,1): 1, (1,1): -1} ],
        [ {(2,2): 1}, {(2,3): 1}, {(3,2): 1}, {(3,3): 1} ],
        [ {(0,0): 1, (0,2): 1}, {(0,1): 1, (0,3): 1}, {(1,0): 1, (1,2): 1}, {(1,1): 1, (1,3): 1} ],
        [ {(2,0): 1, (2,2): 1}, {(2,1): 1, (2,3): 1}, {(3,0): 1, (3,2): 1}, {(3,1): 1, (3,3): 1} ],
    ]
    
    s_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            l_coeffs, r_coeffs = s_defs[s]
            
            left_map = {}
            for pos, coeff in l_coeffs:
                for (ia, ja), c in L_defs[b][pos].items():
                    left_map[(ia, ja)] = left_map.get((ia, ja), 0) + coeff * c
            
            right_map = {}
            for pos, coeff in r_coeffs:
                for (jb, kb), c in R_defs[b][pos].items():
                    right_map[(jb, kb)] = right_map.get((jb, kb), 0) + coeff * c
            
            for (ia, ja), coeff in left_map.items():
                if coeff != 0:
                    U[ia * m + ja, r] = coeff
            
            for (jb, kb), coeff in right_map.items():
                if coeff != 0:
                    V[jb * p + kb, r] = coeff
    
    # Verification using numpy for speed
    # Check T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_col = U[i*m+j, :]
                v_col = V[j*p+k, :]
                w_col = W[k*n+i, :]
                s = np.dot(u_col, v_col * w_col)
                if s != 1:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # Fallback: return schoolbook rank-64 decomposition
        rank_fallback = 64
        U_fb = np.zeros((n * m, rank_fallback), dtype=int)
        V_fb = np.zeros((m * p, rank_fallback), dtype=int)
        W_fb = np.zeros((p * n, rank_fallback), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i*m+j, r] = 1
                    V_fb[j*p+k, r] = 1
                    W_fb[k*n+i, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
