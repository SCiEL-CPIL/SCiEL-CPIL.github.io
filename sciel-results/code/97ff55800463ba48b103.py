import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    block_size = 3
    rank = 108 # Calculated: 3 * (8+4+4+4+4+4+8) = 3 * 36 = 108
    
    # Strassen coefficients
    # A_coeffs[s]: list of (block_row, block_col, coeff)
    # B_coeffs[s]: list of (block_row, block_col, coeff)
    # W_coeffs[s]: list of (block_row, block_col, coeff)
    
    A_coeffs = [
        [ (0,0,1), (1,1,1) ], # M1
        [ (1,0,1), (1,1,1) ], # M2
        [ (0,0,1) ],          # M3
        [ (1,1,1) ],          # M4
        [ (0,0,1), (0,1,1) ], # M5
        [ (1,0,1), (0,0,-1) ],# M6
        [ (0,1,1), (1,1,-1) ] # M7
    ]
    
    B_coeffs = [
        [ (0,0,1), (1,1,1) ], # M1
        [ (0,0,1) ],          # M2
        [ (0,1,1), (1,1,-1) ],# M3
        [ (1,0,1), (0,0,-1) ],# M4
        [ (1,1,1) ],          # M5
        [ (0,0,1), (0,1,1) ], # M6
        [ (1,0,1), (1,1,1) ]  # M7
    ]
    
    W_coeffs = [
        [ (0,0,1), (1,1,1) ], # M1
        [ (1,0,1), (1,1,-1) ],# M2
        [ (0,1,1), (1,1,1) ], # M3
        [ (0,0,1), (1,0,1) ], # M4
        [ (0,1,-1), (1,1,-1) ],# M5
        [ (1,1,1) ],          # M6
        [ (0,0,-1), (1,1,1) ] # M7
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        a_list = A_coeffs[s]
        b_list = B_coeffs[s]
        w_list = W_coeffs[s]
        
        for k in range(block_size):
            for a_idx, b_idx, w_idx in zip(a_list, b_list, w_list): # Wait, lengths might differ?
                # In Strassen, the lists can have different lengths.
                # We need to iterate over ALL combinations of a in A_s, b in B_s, w in W_s.
                pass
                
        # Correct Iteration:
        for a_item in a_list:
            a_br, a_bc, a_c = a_item
            for b_item in b_list:
                b_br, b_bc, b_c = b_item
                for w_item in w_list:
                    w_br, w_bc, w_c = w_item
                    
                    # For this combination, we have a term.
                    # We need to fill the column 'col' for all r, c.
                    for r in range(block_size):
                        # U index
                        u_row = a_br * block_size + r
                        u_col = a_bc * block_size + k
                        U[u_row * 6 + u_col][col] += a_c
                        
                        for c in range(block_size):
                            # V index
                            v_row = b_br * block_size + k
                            v_col = b_bc * block_size + c
                            V[v_row * 6 + v_col][col] += b_c
                            
                            # W index
                            w_row = w_br * block_size + r
                            w_col = w_bc * block_size + c
                            W[w_row * 6 + w_col][col] += w_c
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
