import numpy as np

def construct():
    n, m, p = 2, 2, 14
    U = [[0] * 49 for _ in range(4)]
    V = [[0] * 49 for _ in range(28)]
    W = [[0] * 49 for _ in range(28)]
    
    # Strassen terms for 2x2 x 2x2
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    for b in range(7):
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
