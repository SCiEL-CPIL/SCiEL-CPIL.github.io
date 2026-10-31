import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    
    # Rank 49 decomposition using recursive Strassen.
    # This is the best known upper bound for 4x4 matrix multiplication.
    # Score = 64 / 49 = 1.3061...
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    def a_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def w_idx(br, bc, ir, ic):
        # C row = br*2+ir, C col = bc*2+ic
        # W index = col*4 + row
        return (bc*2 + ic)*4 + (br*2 + ir)

    def mk_dict(pairs):
        d = {}
        for k, v in pairs:
            d[k] = d.get(k, 0) + v
        return d

    strassen_2x2_terms = [
        (mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)])),
        (mk_dict([((1,0),1), ((1,1),1)]), mk_dict([((0,0),1)]),              mk_dict([((1,0),1), ((1,1),-1)])),
        (mk_dict([((0,0),1)]),              mk_dict([((0,1),1), ((1,1),-1)]), mk_dict([((0,1),1), ((1,1),1)])),
        (mk_dict([((1,1),1)]),              mk_dict([((1,0),1), ((0,0),-1)]), mk_dict([((0,0),1), ((1,0),1)])),
        (mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)]),              mk_dict([((0,0),-1), ((0,1),1)])),
        (mk_dict([((1,0),1), ((0,0),-1)]),  mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)])),
        (mk_dict([((0,1),1), ((1,1),-1)]),  mk_dict([((1,0),1), ((1,1),1)]),   mk_dict([((0,0),1)])),
    ]
    
    outer_strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]),
        ([(0,0,1)],           [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)],           [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)],  [(1,1,1)],           [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],  [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],  [(0,0,1)]),
    ]
    
    col_idx = 0
    for outer_term in outer_strassen_terms:
        a_blocks, b_blocks, c_blocks = outer_term
        for inner_term in strassen_2x2_terms:
            a_comb, b_comb, c_comb = inner_term
            
            for (br, bc, coeff) in a_blocks:
                for (lr, lc), a_val in a_comb.items():
                    if a_val == 0: continue
                    u_i = a_idx(br, bc, lr, lc)
                    U[u_i, col_idx] += coeff * a_val
            
            for (br, bc, coeff) in b_blocks:
                for (lr, lc), b_val in b_comb.items():
                    if b_val == 0: continue
                    v_i = b_idx(br, bc, lr, lc)
                    V[v_i, col_idx] += coeff * b_val
            
            for (cbr, cbc, ccoeff_outer) in c_blocks:
                for (ir, ic), c_val_inner in c_comb.items():
                    if c_val_inner == 0: continue
                    w_i = w_idx(cbr, cbc, ir, ic)
                    W[w_i, col_idx] += ccoeff_outer * c_val_inner
            
            col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
