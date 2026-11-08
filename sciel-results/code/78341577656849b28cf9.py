# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 9
    DEADLINE = start_time + 100

    # Use Strassen for 4x4x4 (rank 49)
    # And schoolbook for 4x4x1 (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    # We can potentially improve by using a better decomposition for 4x4x8
    # But for now, we stick to the known 114.
    
    # Generate Strassen terms for 4x4x4
    def get_strassen_44_terms():
        terms = []
        # 7 block products
        # Each block product is a 2x2x2 Strassen (7 scalar mults)
        # Total 49 terms
        
        # We hardcode the 49 terms for efficiency
        # This is a known decomposition
        
        # Instead of generating, we can just use the standard Strassen indices
        # But generating is safer to avoid errors
        
        block_products = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
             {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)],
              2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}),
            ({(1,0):1, (1,1):1}, {(0,0):1},
             {0: [(2,0,1),(2,2,-1)], 1: [(2,1,1),(2,3,-1)],
              2: [(3,0,1),(3,2,-1)], 3: [(3,1,1),(3,3,-1)]}),
            ({(0,0):1}, {(1,0):1, (0,0):-1},
             {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)],
              2: [(1,0,-1),(1,2,1)], 3: [(1,1,-1),(1,3,1)]}),
            ({(1,1):1}, {(0,1):1, (1,0):-1},
             {0: [(0,0,-1),(2,0,1)], 1: [(0,1,-1),(2,1,1)],
              2: [(1,0,-1),(3,0,1)], 3: [(1,1,-1),(3,1,1)]}),
            ({(0,0):1, (0,1):1}, {(1,1):1},
             {0: [(0,2,1),(2,2,1)], 1: [(0,3,1),(2,3,1)],
              2: [(1,2,1),(3,2,1)], 3: [(1,3,1),(3,3,1)]}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
             {0: [(2,2,-1)], 1: [(2,3,-1)],
              2: [(3,2,-1)], 3: [(3,3,-1)]}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
             {0: [(0,0,1)], 1: [(0,1,1)],
              2: [(1,0,1)], 3: [(1,1,1)]}),
        ]
        
        c_contribs = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {0: -1, 1: 1},
            {0: -1, 2: 1},
            {1: 1, 3: 1},
            {3: -1},
            {0: 1},
        ]
        
        def merge(*dicts):
            result = {}
            for d in dicts:
                for k, v in d.items():
                    result[k] = result.get(k, 0) + v
            return {k: v for k, v in result.items() if v != 0}
            
        def negate(d):
            return {k: -v for k, v in d.items()}
            
        for a_blocks, b_blocks, c_map in block_products:
            X = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), sign in a_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        X[di][dj][idx] = sign
            
            Y = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), sign in b_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        Y[di][dj][idx] = sign
            
            scalar_mults = [
                (merge(X[0][0], X[1][1]), merge(Y[0][0], Y[1][1])),
                (merge(X[1][0], X[1][1]), merge(Y[0][0])),
                (merge(X[0][0]), merge(Y[1][0], negate(Y[0][0]))),
                (merge(X[1][1]), merge(Y[0][1], negate(Y[1][0]))),
                (merge(X[0][0], X[0][1]), merge(Y[1][1])),
                (merge(X[1][0], negate(X[0][0])), merge(Y[0][0], Y[0][1])),
                (merge(X[0][1], negate(X[1][1])), merge(Y[1][0], Y[1][1])),
            ]
            
            for k, (a_coefs, b_coefs) in enumerate(scalar_mults):
                if not a_coefs or not b_coefs:
                    continue
                
                c_coefs = {}
                for p_pos, s_sign in c_contribs[k].items():
                    pr, pc = divmod(p_pos, 2)
                    for (c_row, c_col, c_sign) in c_map[p_pos]:
                        c_idx = (c_col, c_row)
                        total_sign = s_sign * c_sign
                        c_coefs[c_idx] = c_coefs.get(c_idx, 0) + total_sign
                
                c_coefs = {k: v for k, v in c_coefs.items() if v != 0}
                if c_coefs:
                    terms.append((a_coefs, b_coefs, c_coefs))
                    
        return terms

    strassen_terms = get_strassen_44_terms()
    
    if len(strassen_terms) != 49:
        # Fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=int)
        V_fb = np.zeros((m*p, rank_fb), dtype=int)
        W_fb = np.zeros((p*n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col] = 1
                    V_fb[j*p + k, col] = 1
                    W_fb[k*n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: A * B1 (cols 0-3)
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + k, c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, c] += val
            
    # Part 2: A * B2 (cols 4-7)
    offset = 49
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, offset + c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + (k + 4), offset + c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, offset + c] += val
            
    # Part 3: A * B3 (col 8)
    offset2 = 98
    for i in range(n):
        for j in range(m):
            col = offset2 + i*m + j
            U[i*m + j, col] = 1
            V[j*p + 8, col] = 1
            W[8*n + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
