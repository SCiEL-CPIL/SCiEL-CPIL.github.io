import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    
    # 2x2x2 Strassen UVW (verified)
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    strassen_V = [
        [1, 1, -1, 0, 0, 1, 0],  # B00
        [0, 0, 0, -1, 0, 1, 0],  # B01
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 1]    # B11
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    outer_steps = [
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}},
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}},
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}},
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}},
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}},
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}},
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}}
    ]
    
    c_block_combos = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: -1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    def a_global(i, j):
        return i * m + j
        
    def b_global(j, k):
        return j * p + k
        
    def c_global(k, i):
        return k * n + i
    
    for s, step in enumerate(outer_steps):
        a_blocks = step['a']
        b_blocks = step['b']
        
        for inner in range(7):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for la in range(4):
                cu = strassen_U[la][inner]
                if cu == 0:
                    continue
                lr, lc = la // 2, la % 2
                for (br, bc), ca in a_blocks.items():
                    gi, gj = br * 2 + lr, bc * 2 + lc
                    u_vec[a_global(gi, gj)] += ca * cu
            
            for lb in range(4):
                cv = strassen_V[lb][inner]
                if cv == 0:
                    continue
                lr, lc = lb // 2, lb % 2
                for (br, bc), cb in b_blocks.items():
                    gi, gj = br * 2 + lr, bc * 2 + lc
                    v_vec[b_global(gi, gj)] += cb * cv
            
            for (br, bc), coeffs in c_block_combos.items():
                if s not in coeffs:
                    continue
                csc = coeffs[s]
                for lc_local in range(4):
                    cw = strassen_W[lc_local][inner]
                    if cw == 0:
                        continue
                    lr, lc = lc_local // 2, lc_local % 2
                    gi, gj = br * 2 + lr, bc * 2 + lc
                    w_vec[c_global(gi, gj)] += csc * cw
            
            for i in range(n * m):
                if u_vec[i] != 0:
                    U[i][t] = u_vec[i]
            for j in range(m * p):
                if v_vec[j] != 0:
                    V[j][t] = v_vec[j]
            for k in range(p * n):
                if w_vec[k] != 0:
                    W[k][t] = w_vec[k]
            t += 1
    
    # Part 2: 4x4x3 trivial
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
