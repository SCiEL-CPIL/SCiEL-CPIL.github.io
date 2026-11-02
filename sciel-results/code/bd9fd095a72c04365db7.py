import numpy as np

def construct():
    n, m, p = 2, 2, 8
    
    # Rank is 28 (4 blocks of 2x2x2, each rank 7)
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00      -> C10, C11
    # M3 = A00(B01-B11)      -> C01, C11
    # M4 = A11(B10-B00)      -> C00, C10
    # M5 = (A00+A01)B11      -> C00, C01
    # M6 = (A10-A00)(B00+B01)-> C11
    # M7 = (A01-A11)(B10+B11)-> C00
    
    # Defined as lists of (index, coeff)
    # A indices: i*m + j
    # B indices: j*p + k_local (k_local is 0 or 1 within the block)
    # W indices: k_global*n + i
    
    strassen = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Note: 
    # A indices in Strassen: 
    # A00=0, A01=1, A10=2, A11=3
    # B indices in Strassen (local k):
    # B00=0, B01=1, B10=2, B11=3
    # C indices in Strassen (local k):
    # C00=0, C01=1, C10=2, C11=3
    
    # My tuple format: (list_of_A_indices, list_of_B_indices, list_of_C_indices)
    # But I need coefficients.
    # Let's rewrite Strassen more explicitly with coefficients.
    
    S = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A: {i*m+j: c}, B: {j*p+k_loc: c}, C: {k_loc*n+i: c}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),           # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),           # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),           # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),           # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),           # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),           # M7
    ]
    
    for q in range(4):
        k_offset = 2 * q
        for r, (A_c, B_c, C_c) in enumerate(S):
            t = q * 7 + r
            
            for idx, val in A_c.items():
                U[idx, t] = val
                
            for idx, val in B_c.items():
                # idx is j*p + k_loc
                # j = idx // p (but p=2, so j=0 or 1? No, p is global p=8. 
                # Wait, B index is j*p + k. 
                # In the 2x2 block, k is local 0,1.
                # Global k = k_offset + k_local.
                # j is 0 or 1.
                # So B index = j*8 + (k_offset + k_local).
                # My B_c keys are j*2 + k_local? 
                # No, in the 2x2x2 tensor, the B dimension is size 2x2=4.
                # Index is j*2 + k_local.
                # Let's map j*2 + k_local -> j*8 + (k_offset + k_local).
                j = idx // 2
                k_loc = idx % 2
                global_idx = j * p + (k_offset + k_loc)
                V[global_idx, t] = val
                
            for idx, val in C_c.items():
                # idx is k_loc*n + i? 
                # In 2x2x2, C dimension is 2x2=4.
                # Index is k_loc*2 + i.
                # Global W index is k_global*n + i.
                k_loc = idx // 2
                i = idx % 2
                global_idx = (k_offset + k_loc) * n + i
                W[global_idx, t] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
