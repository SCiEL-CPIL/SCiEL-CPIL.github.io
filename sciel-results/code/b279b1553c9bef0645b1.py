import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition for each 2x2 block of the second matrix.
    # p=12, so 6 blocks.
    # Rank per block is 7. Total rank = 6 * 7 = 42.
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen factors for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+c)(e+g)
    # P2 = (c+d)(e+g)
    # P3 = (a+b)(f+h)
    # P4 = (c+d)(f+h) -- Wait, standard P4 is (c+d)(f+h)?
    # Let's use the standard 7 multiplications:
    # M1 = (a+c)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = (a+b)(f+h)
    # M4 = (c+d)(f+h)
    # M5 = (a-d)(g+h)
    # M6 = (b-c)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M2 - M6
    # C12 = M3 + M7
    # C21 = M2 + M5
    # C22 = M1 + M3 - M5 - M4
    
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices for block s: 4s:e, 4s+1:f, 4s+2:g, 4s+3:h
    # W indices for block s: 4s:C11, 4s+1:C12, 4s+2:C21, 4s+3:C22 (assuming row-major C)
    # But W convention is W[k*n+i, r].
    # C is 2x2. C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22.
    # W index for C[i,k] is k*n + i.
    # For block s, k=2s, 2s+1.
    # C[0, 2s] -> W[2s*2 + 0] = W[4s]
    # C[0, 2s+1] -> W[(2s+1)*2 + 0] = W[4s+2]
    # C[1, 2s] -> W[2s*2 + 1] = W[4s+1]
    # C[1, 2s+1] -> W[(2s+1)*2 + 1] = W[4s+3]
    
    # Let's define the Strassen coefficients
    # M1: (a+c)(e+g)
    # U: a:1, c:1. V: e:1, g:1.
    # M2: (c+d)(e+g)
    # U: c:1, d:1. V: e:1, g:1.
    # M3: (a+b)(f+h)
    # U: a:1, b:1. V: f:1, h:1.
    # M4: (c+d)(f+h)
    # U: c:1, d:1. V: f:1, h:1.
    # M5: (a-d)(g+h)
    # U: a:1, d:-1. V: g:1, h:1.
    # M6: (b-c)(e+f)
    # U: b:1, c:-1. V: e:1, f:1.
    # M7: (b-d)(g+h)
    # U: b:1, d:-1. V: g:1, h:1.
    
    # Outputs:
    # C11 = M1 + M4 - M2 - M6
    # C12 = M3 + M7
    # C21 = M2 + M5
    # C22 = M1 + M3 - M5 - M4
    
    # We will fill U, V, W for each block.
    col_offset = 0
    for s in range(num_blocks):
        # V indices for this block
        v_e = 4 * s + 0
        v_f = 4 * s + 1
        v_g = 4 * s + 2
        v_h = 4 * s + 3
        
        # W indices for this block
        # C[0, 2s] is W[4s]
        # C[1, 2s] is W[4s+1]
        # C[0, 2s+1] is W[4s+2]
        # C[1, 2s+1] is W[4s+3]
        w_c11 = 4 * s + 0
        w_c21 = 4 * s + 1
        w_c12 = 4 * s + 2
        w_c22 = 4 * s + 3
        
        # M1
        U[0, col_offset] = 1 # a
        U[2, col_offset] = 1 # c
        V[v_e, col_offset] = 1
        V[v_g, col_offset] = 1
        W[w_c11, col_offset] = 1
        W[w_c22, col_offset] = 1
        col_offset += 1
        
        # M2
        U[2, col_offset] = 1 # c
        U[3, col_offset] = 1 # d
        V[v_e, col_offset] = 1
        V[v_g, col_offset] = 1
        W[w_c21, col_offset] = 1
        W[w_c11, col_offset] = -1
        col_offset += 1
        
        # M3
        U[0, col_offset] = 1 # a
        U[1, col_offset] = 1 # b
        V[v_f, col_offset] = 1
        V[v_h, col_offset] = 1
        W[w_c12, col_offset] = 1
        W[w_c22, col_offset] = 1
        col_offset += 1
        
        # M4
        U[2, col_offset] = 1 # c
        U[3, col_offset] = 1 # d
        V[v_f, col_offset] = 1
        V[v_h, col_offset] = 1
        W[w_c11, col_offset] = 1
        W[w_c22, col_offset] = -1
        col_offset += 1
        
        # M5
        U[0, col_offset] = 1 # a
        U[3, col_offset] = -1 # d
        V[v_g, col_offset] = 1
        V[v_h, col_offset] = 1
        W[w_c21, col_offset] = 1
        W[w_c22, col_offset] = -1
        col_offset += 1
        
        # M6
        U[1, col_offset] = 1 # b
        U[2, col_offset] = -1 # c
        V[v_e, col_offset] = 1
        V[v_f, col_offset] = 1
        W[w_c11, col_offset] = -1
        col_offset += 1
        
        # M7
        U[1, col_offset] = 1 # b
        U[3, col_offset] = -1 # d
        V[v_g, col_offset] = 1
        V[v_h, col_offset] = 1
        W[w_c12, col_offset] = 1
        col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
