import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    # Initialize matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each term is a tuple of three dicts: {local_index: coeff}
    # U local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: W local index maps to (i, k_local) where i is row, k_local is col in 2x2 block.
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00) -> C10+=M4, C00+=M4
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: 1}),
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    # Loop over the 3 k-blocks (columns of B/C)
    for k_block in range(3):
        # Loop over the 2 j-blocks (rows of B / cols of A)
        for j_block in range(2):
            # Loop over the 7 Strassen terms
            for term in strassen_terms:
                u_dict, v_dict, w_dict = term
                
                # Update U
                # Global U index: i * m + j
                # i is from local_u // 2
                # j is j_block * 2 + (local_u % 2)
                for local_u, coeff in u_dict.items():
                    i = local_u // 2
                    j_loc = local_u % 2
                    j_global = j_block * 2 + j_loc
                    u_idx = i * m + j_global
                    U[u_idx][col] += coeff
                
                # Update V
                # Global V index: j * p + k
                # j is j_block * 2 + (local_v // 2)
                # k is k_block * 2 + (local_v % 2)
                for local_v, coeff in v_dict.items():
                    j_loc = local_v // 2
                    k_loc = local_v % 2
                    j_global = j_block * 2 + j_loc
                    k_global = k_block * 2 + k_loc
                    v_idx = j_global * p + k_global
                    V[v_idx][col] += coeff
                
                # Update W
                # Global W index: k * n + i
                # k is k_block * 2 + (local_w % 2)
                # i is local_w // 2
                for local_w, coeff in w_dict.items():
                    i = local_w // 2
                    k_loc = local_w % 2
                    k_global = k_block * 2 + k_loc
                    w_idx = k_global * n + i
                    W[w_idx][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
