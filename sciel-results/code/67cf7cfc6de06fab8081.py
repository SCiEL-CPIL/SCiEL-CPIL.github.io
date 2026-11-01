import numpy as np

def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2 # 5
    
    # Strassen 7 terms for 2x2
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # But for a single block, local indices are i,j in 0..1, k in 0..1.
    # Global indices:
    # U[i*m + j, col]
    # V[j*p + (2*r + k), col]
    # W[(2*r + k)*n + i, col]
    
    # Define the 7 Strassen terms as (u_dict, v_dict, w_dict)
    # u_dict: (i,j) -> coeff
    # v_dict: (j,k) -> coeff
    # w_dict: (k,i) -> coeff
    
    terms = [
        # M1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        # M3 = a(f-h)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ),
        # M4 = d(g-e)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        # M5 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ),
        # M6 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    rank = num_blocks * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_dict, v_dict, w_dict = terms[t]
            
            for (i, j), val in u_dict.items():
                U[i*m + j, col] = val
                
            for (j, k), val in v_dict.items():
                V[j*p + (2*r + k), col] = val
                
            for (k, i), val in w_dict.items():
                W[(2*r + k)*n + i, col] = val
                
    # Convert to lists to match seed interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
