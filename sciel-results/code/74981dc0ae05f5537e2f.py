import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Dimensions
    N = 4
    R = 49
    
    # Initialize matrices
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((16, R), dtype=int)
    W = np.zeros((16, R), dtype=int)
    
    # Helper to create coefficient dicts
    def coeff(pairs):
        d = {}
        for pos, val in pairs:
            d[pos] = d.get(pos, 0) + val
        return d

    # Strassen 2x2 terms (Inner level)
    # Format: (A_dict, B_dict, C_dict)
    S2 = [
        (coeff([((0,0),1), ((1,1),1)]), coeff([((0,0),1), ((1,1),1)]), coeff([((0,0),1), ((1,1),1)])),
        (coeff([((1,0),1), ((1,1),1)]), coeff([((0,0),1)]),              coeff([((1,0),1), ((1,1),-1)])),
        (coeff([((0,0),1)]),              coeff([((0,1),1), ((1,1),-1)]), coeff([((0,1),1), ((1,1),1)])),
        (coeff([((1,1),1)]),              coeff([((1,0),1), ((0,0),-1)]), coeff([((0,0),1), ((1,0),1)])),
        (coeff([((0,0),1), ((0,1),1)]),   coeff([((1,1),1)]),              coeff([((0,0),-1), ((0,1),1)])),
        (coeff([((1,0),1), ((0,0),-1)]),  coeff([((0,0),1), ((0,1),1)]),   coeff([((1,1),1)])),
        (coeff([((0,1),1), ((1,1),-1)]),  coeff([((1,0),1), ((1,1),1)]),  coeff([((0,0),1)])),
    ]
    
    # Outer Strassen terms (4x4 level)
    # Format: (A_blocks, B_blocks, C_blocks)
    # Blocks: list of (block_row, block_col, coeff)
    S4 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]),
        ([(0,0,1)],           [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)],           [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)],  [(1,1,1)],           [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],  [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],  [(0,0,1)]),
    ]
    
    # Index mapping functions
    def idx_a(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def idx_b(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def idx_c(cbr, cbc, ir, ic):
        # Trace convention: C[i,k] -> W[k*4+i]
        # C row = cbr*2+ir, C col = cbc*2+ic
        return (cbc*2 + ic)*4 + (cbr*2 + ir)

    # Fill matrices
    col = 0
    for o in S4:
        a_blocks, b_blocks, c_blocks = o
        for i in S2:
            a_d, b_d, c_d = i
            
            # U
            for (br, bc, cf) in a_blocks:
                for (lr, lc), val in a_d.items():
                    if val == 0: continue
                    U[idx_a(br, bc, lr, lc), col] += cf * val
            
            # V
            for (br, bc, cf) in b_blocks:
                for (lr, lc), val in b_d.items():
                    if val == 0: continue
                    V[idx_b(br, bc, lr, lc), col] += cf * val
            
            # W
            for (cbr, cbc, cf) in c_blocks:
                for (ir, ic), val in c_d.items():
                    if val == 0: continue
                    W[idx_c(cbr, cbc, ir, ic), col] += cf * val
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
