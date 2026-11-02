import numpy as np

def construct():
    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7 blocks
    
    # Total rank is 7 terms per block * 7 blocks
    rank = 7 * num_blocks
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (4, 49)
    # V: (m*p, rank) -> (28, 49)
    # W: (p*n, rank) -> (28, 49)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 multiplication
    # Local indices for A (U): 00:0, 01:1, 10:2, 11:3
    # Local indices for B (V): 00:0, 01:1, 10:2, 11:3
    # Local indices for C (W): 00:0, 01:1, 10:2, 11:3
    
    # Terms: list of (U_coeff_dict, V_coeff_dict, W_coeff_dict)
    # Coefficients are integers.
    
    # M1: (A00 + A11) * (B00 + B11)
    t0 = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    
    # M2: (A10 + A11) * B00
    # C10 += M2, C11 -= M2
    t1 = ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    
    # M3: A00 * (B01 - B11)
    # C01 += M3, C11 += M3
    t2 = ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    
    # M4: A11 * (B10 - B00)
    # C00 += M4, C10 += M4
    t3 = ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    
    # M5: (A00 + A01) * B11
    # C00 -= M5, C01 += M5
    t4 = ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    
    # M6: (A10 - A00) * (B00 + B01)
    # C11 += M6
    t5 = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    
    # M7: (A01 - A11) * (B10 + B11)
    # C00 += M7
    t6 = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    terms = [t0, t1, t2, t3, t4, t5, t6]
    
    # Fill the matrices
    for b in range(num_blocks):
        # Global offsets for V and W indices for block b
        # V local l -> global index
        # l=0 (B00): row 0, col 2b -> 0*14 + 2b
        # l=1 (B01): row 0, col 2b+1 -> 0*14 + 2b+1
        # l=2 (B10): row 1, col 2b -> 1*14 + 2b
        # l=3 (B11): row 1, col 2b+1 -> 1*14 + 2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # W local l -> global index
        # l=0 (C00): row 2b, col 0 -> 2b*2 + 0
        # l=1 (C01): row 2b+1, col 0 -> (2b+1)*2 + 0
        # l=2 (C10): row 2b, col 1 -> 2b*2 + 1
        # l=3 (C11): row 2b+1, col 1 -> (2b+1)*2 + 1
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = b * 7 + t_idx
            
            # Update U
            for idx, val in u_dict.items():
                U[idx, col] += val
            
            # Update V
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx], col] += val
            
            # Update W
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx], col] += val
                
    # Convert to lists as required by the interface if it expects lists, 
    # but usually numpy arrays or lists of lists are accepted. 
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The seed used lists. I will return lists to be safe and consistent.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
