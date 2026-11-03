import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict {(i,j): coeff}
        # v_vals: dict {(j,k): coeff}
        # w_vals: dict {(k,i): coeff}
        pass # Implemented in main loop

    # Helper to index U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    term = 0
    
    # Strassen helper
    def add_strassen(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_k0, c_k1, c_k2, c_k3):
        # a_i0, a_i1 are row 0 of A block (cols i0, i1)
        # a_j0, a_j1 are row 1 of A block (cols j0, j1)
        # b_j0, b_j1 are row 0 of B block (rows j0, j1)
        # b_k0, b_k1 are row 1 of B block (rows k0, k1)
        # c_k0, c_k1 are cols of C block (cols k0, k1)
        # c_k2, c_k3 are cols of C block (cols k2, k3)
        # This is getting messy with variable names. 
        # Let's just use the explicit indices.
        pass

    # I'll just write out the 36 terms explicitly in a loop or list.
    
    # Define the 7 Strassen patterns
    # Pattern 1: M1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    
    # Let's create a list of (U_dict, V_dict, W_dict)
    
    terms = []
    
    # Group 1: A_L (cols 0,1) * B_t1 (rows 0,1, cols 0,1) -> C cols 0,1
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W indices: 0, 2, 1, 3
    
    # M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    # Group 2: A_L (cols 0,1) * B_t2 (rows 0,1, cols 2,3) -> C cols 2,3
    # B indices: (0,2), (0,3), (1,2), (1,3)
    # C indices: (0,2), (0,3), (1,2), (1,3) -> W indices: 4, 6, 5, 7
    
    # M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,2): 1, (1,3): 1},
        'W': {(0,2): 1, (1,3): 1}
    })
    # M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,2): 1},
        'W': {(1,2): 1, (1,3): -1}
    })
    # M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,2): 1, (1,3): -1},
        'W': {(0,3): 1, (1,3): 1}
    })
    # M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,3): 1, (0,2): -1},
        'W': {(0,2): 1, (1,2): 1}
    })
    # M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,3): 1},
        'W': {(0,2): -1, (0,3): 1}
    })
    # M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,2): 1, (0,3): 1},
        'W': {(1,3): 1}
    })
    # M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,2): 1, (1,3): 1},
        'W': {(0,2): 1}
    })
    
    # Group 3: A_R (cols 2,3) * B_b1 (rows 2,3, cols 0,1) -> C cols 0,1
    # A indices: (0,2), (0,3), (1,2), (1,3)
    # B indices: (2,0), (2,1), (3,0), (3,1)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W indices: 0, 2, 1, 3
    
    # M1
    terms.append({
        'U': {(0,2): 1, (1,3): 1},
        'V': {(2,0): 1, (3,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2
    terms.append({
        'U': {(1,2): 1, (1,3): 1},
        'V': {(2,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3
    terms.append({
        'U': {(0,2): 1},
        'V': {(3,0): 1, (3,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4
    terms.append({
        'U': {(1,3): 1},
        'V': {(2,1): 1, (2,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5
    terms.append({
        'U': {(0,2): 1, (0,3): 1},
        'V': {(3,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6
    terms.append({
        'U': {(1,2): 1, (0,2): -1},
        'V': {(2,0): 1, (2,1): 1},
        'W': {(1,1): 1}
    })
    # M7
    terms.append({
        'U': {(0,3): 1, (1,3): -1},
        'V': {(3,0): 1, (3,1): 1},
        'W': {(0,0): 1}
    })
    
    # Group 4: A_R (cols 2,3) * B_b2 (rows 2,3, cols 2,3) -> C cols 2,3
    # B indices: (2,2), (2,3), (3,2), (3,3)
    # C indices: (0,2), (0,3), (1,2), (1,3) -> W indices: 4, 6, 5, 7
    
    # M1
    terms.append({
        'U': {(0,2): 1, (1,3): 1},
        'V': {(2,2): 1, (3,3): 1},
        'W': {(0,2): 1, (1,3): 1}
    })
    # M2
    terms.append({
        'U': {(1,2): 1, (1,3): 1},
        'V': {(2,2): 1},
        'W': {(1,2): 1, (1,3): -1}
    })
    # M3
    terms.append({
        'U': {(0,2): 1},
        'V': {(3,2): 1, (3,3): -1},
        'W': {(0,3): 1, (1,3): 1}
    })
    # M4
    terms.append({
        'U': {(1,3): 1},
        'V': {(2,3): 1, (2,2): -1},
        'W': {(0,2): 1, (1,2): 1}
    })
    # M5
    terms.append({
        'U': {(0,2): 1, (0,3): 1},
        'V': {(3,3): 1},
        'W': {(0,2): -1, (0,3): 1}
    })
    # M6
    terms.append({
        'U': {(1,2): 1, (0,2): -1},
        'V': {(2,2): 1, (2,3): 1},
        'W': {(1,3): 1}
    })
    # M7
    terms.append({
        'U': {(0,3): 1, (1,3): -1},
        'V': {(3,2): 1, (3,3): 1},
        'W': {(0,2): 1}
    })
    
    # Group 5: A * b (8 terms)
    # b is col 4 of B.
    # C col 4.
    # W indices: (0,4)=8, (1,4)=9
    
    # Row 0 of C
    terms.append({'U': {(0,0): 1}, 'V': {(0,4): 1}, 'W': {(0,4): 1}})
    terms.append({'U': {(0,1): 1}, 'V': {(1,4): 1}, 'W': {(0,4): 1}})
    terms.append({'U': {(0,2): 1}, 'V': {(2,4): 1}, 'W': {(0,4): 1}})
    terms.append({'U': {(0,3): 1}, 'V': {(3,4): 1}, 'W': {(0,4): 1}})
    
    # Row 1 of C
    terms.append({'U': {(1,0): 1}, 'V': {(0,4): 1}, 'W': {(1,4): 1}})
    terms.append({'U': {(1,1): 1}, 'V': {(1,4): 1}, 'W': {(1,4): 1}})
    terms.append({'U': {(1,2): 1}, 'V': {(2,4): 1}, 'W': {(1,4): 1}})
    terms.append({'U': {(1,3): 1}, 'V': {(3,4): 1}, 'W': {(1,4): 1}})
    
    # Fill U, V, W
    for c, t in enumerate(terms):
        for (i, j), val in t['U'].items():
            U[i * m + j, c] = val
        for (j, k), val in t['V'].items():
            V[j * p + k, c] = val
        for (k, i), val in t['W'].items():
            W[k * n + i, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
