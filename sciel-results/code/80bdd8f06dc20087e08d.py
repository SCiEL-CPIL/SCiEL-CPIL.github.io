"""Matrix multiplication 2,2,13 — rank-46 tensor decomposition"""

import numpy as np

def construct():
    n, m, p = 2, 2, 13
    total_rank = 6 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    
    for pair_idx in range(6):
        ka, kb = 2 * pair_idx, 2 * pair_idx + 1
        
        # Strassen rank-7 block for columns (ka, kb)
        # Each sub-term: (U_part, V_part, W_part)
        blocks = [
            # (U coeffs, V coeffs, W coeffs) as (idx, coeff) lists
            ([(0,1),(3,1)],   [(ka,1),(13+kb,1)],   [(2*ka,1),(2*ka+1,1),(2*kb,1),(2*kb+1,1)]),
            ([(2,1),(3,1)],   [(ka,1)],              [(2*ka+1,1),(2*kb+1,1)]),
            ([(0,1)],         [(13+ka,1),(ka,-1)],   [(2*ka,1),(2*kb,1)]),
            ([(3,1)],         [(13+kb,1),(13+ka,-1)],[(2*kb,1),(2*kb+1,1)]),
            ([(0,1),(1,1)],   [(13+kb,1)],           [(2*ka,1),(2*ka+1,1)]),
            ([(2,1),(0,-1)],  [(13+ka,1),(ka,1)],    [(2*kb+1,1)]),
            ([(1,1),(3,-1)],  [(13+kb,1),(13+ka,1)], [(2*kb,1)]),
        ]
        
        for u_list, v_list, w_list in blocks:
            for ui, uc in u_list:
                U[ui, offset] += uc
            for vi, vc in v_list:
                V[vi, offset] += vc
            for wi, wc in w_list:
                W[wi, offset] += wc
            offset += 1
    
    # Final column k=12: schoolbook rank-4
    k = 12
    for (ui, vi, wi) in [(0, k, 2*k), (1, 13+k, 2*k), (2, k, 2*k+1), (3, 13+k, 2*k+1)]:
        U[ui, offset] = 1
        V[vi, offset] = 1
        W[wi, offset] = 1
        offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
