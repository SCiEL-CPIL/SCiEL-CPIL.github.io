import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # We will use a block-wise approach.
    # The multiplication C = A * B where A is 2x2 and B is 2x14.
    # We can decompose B into 7 blocks of 2x2 matrices: B_r for r=0..6.
    # C_r = A * B_r.
    # Since the blocks are independent (no cross-terms in standard multiplication),
    # we can use a rank-7 decomposition for each 2x2x2 block.
    # Total rank = 7 * 7 = 49.
    
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications.
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]].
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    
    # Strassen's 7 terms:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - e)
    # M5 = (a + b) * h
    # M6 = (c - a) * (e + f)
    # M7 = (b - d) * (g + h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We map these to the tensor factors.
    # For each block r (0..6), we have 7 terms.
    # Total rank = 49.
    
    rank = 49
    U = np.zeros((4, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Indices for A (U):
    # a -> 0, b -> 1, c -> 2, d -> 3
    
    # Indices for B (V):
    # For block r, B has columns 2r and 2r+1.
    # e = B[0, 2r] -> index 0*14 + 2r = 2r
    # f = B[0, 2r+1] -> index 0*14 + 2r+1 = 2r+1
    # g = B[1, 2r] -> index 1*14 + 2r = 14+2r
    # h = B[1, 2r+1] -> index 1*14 + 2r+1 = 14+2r+1
    
    # Indices for C (W):
    # C11 = C[0, 2r] -> index 2r*2 + 0 = 4r
    # C12 = C[0, 2r+1] -> index (2r+1)*2 + 0 = 4r+2
    # C21 = C[1, 2r] -> index 2r*2 + 1 = 4r+1
    # C22 = C[1, 2r+1] -> index (2r+1)*2 + 1 = 4r+3
    
    for r in range(7):
        # V indices for this block
        idx_e = 2*r
        idx_f = 2*r + 1
        idx_g = 14 + 2*r
        idx_h = 14 + 2*r + 1
        
        # W indices for this block
        idx_c11 = 4*r
        idx_c12 = 4*r + 2
        idx_c21 = 4*r + 1
        idx_c22 = 4*r + 3
        
        # Term 1: M1 = (a + d) * (e + h)
        t = 7*r
        U[0, t] += 1  # a
        U[3, t] += 1  # d
        V[idx_e, t] += 1  # e
        V[idx_h, t] += 1  # h
        W[idx_c11, t] += 1
        W[idx_c22, t] += 1
        
        # Term 2: M2 = (c + d) * e
        t = 7*r + 1
        U[2, t] += 1  # c
        U[3, t] += 1  # d
        V[idx_e, t] += 1  # e
        W[idx_c21, t] += 1
        W[idx_c22, t] -= 1
        
        # Term 3: M3 = a * (f - h)
        t = 7*r + 2
        U[0, t] += 1  # a
        V[idx_f, t] += 1  # f
        V[idx_h, t] -= 1  # -h
        W[idx_c12, t] += 1
        W[idx_c22, t] += 1
        
        # Term 4: M4 = d * (g - e)
        t = 7*r + 3
        U[3, t] += 1  # d
        V[idx_g, t] += 1  # g
        V[idx_e, t] -= 1  # -e
        W[idx_c11, t] += 1
        W[idx_c21, t] += 1
        
        # Term 5: M5 = (a + b) * h
        t = 7*r + 4
        U[0, t] += 1  # a
        U[1, t] += 1  # b
        V[idx_h, t] += 1  # h
        W[idx_c11, t] -= 1
        W[idx_c12, t] += 1
        
        # Term 6: M6 = (c - a) * (e + f)
        t = 7*r + 5
        U[2, t] += 1  # c
        U[0, t] -= 1  # -a
        V[idx_e, t] += 1  # e
        V[idx_f, t] += 1  # f
        W[idx_c22, t] += 1
        
        # Term 7: M7 = (b - d) * (g + h)
        t = 7*r + 6
        U[1, t] += 1  # b
        U[3, t] -= 1  # -d
        V[idx_g, t] += 1  # g
        V[idx_h, t] += 1  # h
        W[idx_c11, t] += 1
    
    # Convert to lists as per interface requirement (though numpy arrays might be accepted, 
    # the seed used lists. The contract says "factor matrices of shapes... with integer/half-integer entries".
    # Usually numpy arrays are fine, but to be safe and match seed style, let's return lists 
    # or ensure numpy is handled. The seed returned lists. I will return lists to be safe.)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
