import time
import numpy as np

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses the recursive Strassen algorithm (7x7 = 49 rank terms).
    
    The decomposition is: sum_r U[:,r] * V[:,r] * W[:,r] == tensor_{i,j,k} A[i,j] B[j,k] C[i,k]
    where U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    W uses trace convention: W[k*4+i, r] corresponds to C[i,k].
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 4  # Matrix dimension
    R = 49  # Rank = 7 * 7 (two levels of Strassen)
    
    # Initialize factor matrices
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((16, R), dtype=int)
    W = np.zeros((16, R), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Each term: (A_coeff_dict, B_coeff_dict, C_coeff_dict)
    # where dicts map (row, col) -> coefficient
    def mk_dict(pairs):
        d = {}
        for (pos, val) in pairs:
            d[pos] = d.get(pos, 0) + val
        # Remove zeros
        return {k: v for k, v in d.items() if v != 0}
    
    S2 = [
        (mk_dict([((0,0),1),((1,1),1)]), mk_dict([((0,0),1),((1,1),1)]), mk_dict([((0,0),1),((1,1),1)])),
        (mk_dict([((1,0),1),((1,1),1)]), mk_dict([((0,0),1)]),             mk_dict([((1,0),1),((1,1),-1)])),
        (mk_dict([((0,0),1)]),             mk_dict([((0,1),1),((1,1),-1)]), mk_dict([((0,1),1),((1,1),1)])),
        (mk_dict([((1,1),1)]),             mk_dict([((1,0),1),((0,0),-1)]), mk_dict([((0,0),1),((1,0),1)])),
        (mk_dict([((0,0),1),((0,1),1)]),   mk_dict([((1,1),1)]),             mk_dict([((0,0),-1),((0,1),1)])),
        (mk_dict([((1,0),1),((0,0),-1)]),  mk_dict([((0,0),1),((0,1),1)]),   mk_dict([((1,1),1)])),
        (mk_dict([((0,1),1),((1,1),-1)]),  mk_dict([((1,0),1),((1,1),1)]),   mk_dict([((0,0),1)])),
    ]
    
    # Outer 4x4 level: 7 terms, each specifying which 2x2 blocks to use
    # Format: (list of (block_row, block_col, coeff), ...) for A, B, C
    S4 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],          [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
        ([(1,1,1)],          [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]
    
    # Index mapping: convert (block_row, block_col, local_row, local_col) to linear index
    def a_idx(br, bc, lr, lc):
        # A[i,j]: i = br*2+lr, j = bc*2+lc; linear = i*4+j
        return (br*2 + lr)*4 + (bc*2 + lc)
    
    def b_idx(br, bc, lr, lc):
        # B[j,k]: j = br*2+lr, k = bc*2+lc; linear = j*4+k
        return (br*2 + lr)*4 + (bc*2 + lc)
    
    def w_idx(cbr, cbc, ir, ic):
        # C[i,k]: i = cbr*2+ir, k = cbc*2+ic
        # Trace convention: W[k*4+i]
        return (cbc*2 + ic)*4 + (cbr*2 + ir)
    
    # Fill the factor matrices
    col = 0
    for o in S4:
        a_blocks, b_blocks, c_blocks = o
        for i in S2:
            a_d, b_d, c_d = i
            
            # Fill U column
            for (br, bc, cf) in a_blocks:
                for (lr, lc), val in a_d.items():
                    U[a_idx(br, bc, lr, lc), col] += cf * val
            
            # Fill V column
            for (br, bc, cf) in b_blocks:
                for (lr, lc), val in b_d.items():
                    V[b_idx(br, bc, lr, lc), col] += cf * val
            
            # Fill W column
            for (cbr, cbc, cf) in c_blocks:
                for (ir, ic), val in c_d.items():
                    W[w_idx(cbr, cbc, ir, ic), col] += cf * val
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
