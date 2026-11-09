import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a product term
    # u: vector of length n*m (flattened A)
    # v: vector of length m*p (flattened B)
    # w: vector of length p*n (flattened C coefficients, W[k*n+i])
    # col: index
    def add_term(u, v, w, col):
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w

    # Strassen 2x2x2 products
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # Indices in A: 0:a, 1:b, 2:c, 3:d
    # Indices in B: 0:e, 1:f, 2:g, 3:h (for 2x2)
    # But B is 4x5.
    
    # We need to define the linear forms for each block.
    
    # Block 1: A1 (cols 0,1), B1a (rows 0,1, cols 0,1)
    # A1 = [[a00, a01], [a10, a11]]
    # B1a = [[b00, b01], [b10, b11]]
    
    def strassen_2x2x2(A_idx, B_idx, base_col):
        # A_idx: indices for a,b,c,d in A
        # B_idx: indices for e,f,g,h in B
        # Returns 7 terms
        
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        
        # P1 = (a+d)(e+h)
        u1 = np.zeros(n*m); u1[a] = 1; u1[d] = 1
        v1 = np.zeros(m*p); v1[e] = 1; v1[h] = 1
        w1 = np.zeros(p*n)
        # C11 = P1 + ...
        # C12 = ...
        # C21 = ...
        # C22 = P1 - ...
        # W indices: W[k*n+i]
        # C11 (i=0, k=0) -> W[0] += 1
        # C12 (i=0, k=1) -> W[1] += 0 (for this term)
        # C21 (i=1, k=0) -> W[2] += 0
        # C22 (i=1, k=1) -> W[3] += 1
        # Wait, the output of this block is a 2x2 matrix.
        # The global C is 2x5.
        # We need to map local C indices to global C indices.
        # This function should return the U, V, W vectors for the 7 products.
        
        pass

    # This is getting complex to hardcode.
    # Let's generate the matrices directly.
    
    # We will define the 36 products explicitly.
    
    # Product 1-7: A1 * B1a (2x2x2)
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5 in U?
    # U index: i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    
    # B1a: B[0,0], B[0,1], B[1,0], B[1,1]
    # V index: j*p + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    
    # W index: k*n + i.
    # Output C[0,0] -> 0
    # Output C[0,1] -> 1
    # Output C[1,0] -> 2
    # Output C[1,1] -> 3
    
    # Strassen P1: (A00+A11)(B00+B11)
    # u: [1,0,0,0,0,1,0,0]
    # v: [1,0,0,0,0,0,1,0...]
    # w: C00 += 1, C11 += 1?
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We will write a helper to add Strassen terms for a specific block.
    
    def add_strassen_2x2x2(A_offsets, B_offsets, W_offsets, base_col):
        # A_offsets: [a, b, c, d] indices in U
        # B_offsets: [e, f, g, h] indices in V
        # W_offsets: [c00, c01, c10, c11] indices in W
        # base_col: starting column index
        
        a, b, c, d = A_offsets
        e, f, g, h = B_offsets
        c00, c01, c10, c11 = W_offsets
        
        # P1 = (a+d)(e+h)
        u = np.zeros(n*m); u[a]=1; u[d]=1
        v = np.zeros(m*p); v[e]=1; v[h]=1
        w = np.zeros(p*n); w[c00]=1; w[c11]=1
        add_term(u, v, w, base_col + 0)
        
        # P2 = (c+d)e
        u = np.zeros(n*m); u[c]=1; u[d]=1
        v = np.zeros(m*p); v[e]=1
        w = np.zeros(p*n); w[c10]=1; w[c11]=-1
        add_term(u, v, w, base_col + 1)
        
        # P3 = a(f-h)
        u = np.zeros(n*m); u[a]=1
        v = np.zeros(m*p); v[f]=1; v[h]=-1
        w = np.zeros(p*n); w[c01]=1; w[c11]=1
        add_term(u, v, w, base_col + 2)
        
        # P4 = d(g-e)
        u = np.zeros(n*m); u[d]=1
        v = np.zeros(m*p); v[g]=1; v[e]=-1
        w = np.zeros(p*n); w[c00]=1; w[c10]=1
        add_term(u, v, w, base_col + 3)
        
        # P5 = (a+b)h
        u = np.zeros(n*m); u[a]=1; u[b]=1
        v = np.zeros(m*p); v[h]=1
        w = np.zeros(p*n); w[c00]=-1; w[c01]=1
        add_term(u, v, w, base_col + 4)
        
        # P6 = (c+a)(e-f)
        u = np.zeros(n*m); u[c]=1; u[a]=1
        v = np.zeros(m*p); v[e]=1; v[f]=-1
        w = np.zeros(p*n); w[c11]=1
        add_term(u, v, w, base_col + 5)
        
        # P7 = (b+d)(g-h)
        u = np.zeros(n*m); u[b]=1; u[d]=1
        v = np.zeros(m*p); v[g]=1; v[h]=-1
        w = np.zeros(p*n); w[c00]=1
        add_term(u, v, w, base_col + 6)

    # Helper for 2x2x3 (Strassen 2x2x2 + 4 new)
    def add_strassen_2x2x3(A_offsets, B_offsets, W_offsets, base_col):
        # A_offsets: [a, b, c, d]
        # B_offsets: [e, f, i, g, h, j] (cols 0,1,2)
        # W_offsets: [c00, c01, c02, c10, c11, c12]
        
        a, b, c, d = A_offsets
        e, f, i, g, h, j = B_offsets
        c00, c01, c02, c10, c11, c12 = W_offsets
        
        # First 7 are Strassen for e,f,g,h -> c00, c01, c10, c11
        # Note: W_offsets for Strassen part should be [c00, c01, c10, c11]
        add_strassen_2x2x2(A_offsets, [e, f, g, h], [c00, c01, c10, c11], base_col)
        
        # New 4 for i, j -> c02, c12
        # P8 = a i
        u = np.zeros(n*m); u[a]=1
        v = np.zeros(m*p); v[i]=1
        w = np.zeros(p*n); w[c02]=1
        add_term(u, v, w, base_col + 7)
        
        # P9 = b j
        u = np.zeros(n*m); u[b]=1
        v = np.zeros(m*p); v[j]=1
        w = np.zeros(p*n); w[c02]=1
        add_term(u, v, w, base_col + 8)
        
        # P10 = c i
        u = np.zeros(n*m); u[c]=1
        v = np.zeros(m*p); v[i]=1
        w = np.zeros(p*n); w[c12]=1
        add_term(u, v, w, base_col + 9)
        
        # P11 = d j
        u = np.zeros(n*m); u[d]=1
        v = np.zeros(m*p); v[j]=1
        w = np.zeros(p*n); w[c12]=1
        add_term(u, v, w, base_col + 10)

    # Define indices
    # A: 2x4
    # A1 (cols 0,1): A00=0, A01=1, A10=4, A11=5
    # A2 (cols 2,3): A02=2, A03=3, A12=6, A13=7
    
    # B: 4x5
    # B1 (rows 0,1)
    # B1a (cols 0,1): B00=0, B01=1, B10=5, B11=6
    # B1b (cols 2,3,4): B02=2, B03=3, B04=4, B12=7, B13=8, B14=9
    # B2 (rows 2,3)
    # B2a (cols 0,1): B20=10, B21=11, B30=15, B31=16
    # B2b (cols 2,3,4): B22=12, B23=13, B24=14, B32=17, B33=18, B34=19
    
    # C: 2x5
    # C00=0, C01=1, C02=2, C03=3, C04=4
    # C10=5, C11=6, C12=7, C13=8, C14=9
    
    # Block 1: A1 * B1a (2x2x2)
    # A1: [0, 1, 4, 5]
    # B1a: [0, 1, 5, 6]
    # W: [0, 1, 5, 6] (C00, C01, C10, C11)
    add_strassen_2x2x2([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 5, 6], 0)
    
    # Block 2: A1 * B1b (2x2x3)
    # A1: [0, 1, 4, 5]
    # B1b: [2, 3, 4, 7, 8, 9] (e=2, f=3, i=4, g=7, h=8, j=9)
    # W: [2, 3, 4, 7, 8, 9] (C02, C03, C04, C12, C13, C14)
    add_strassen_2x2x3([0, 1, 4, 5], [2, 3, 4, 7, 8, 9], [2, 3, 4, 7, 8, 9], 7)
    
    # Block 3: A2 * B2a (2x2x2)
    # A2: [2, 3, 6, 7]
    # B2a: [10, 11, 15, 16]
    # W: [0, 1, 5, 6]
    add_strassen_2x2x2([2, 3, 6, 7], [10, 11, 15, 16], [0, 1, 5, 6], 18)
    
    # Block 4: A2 * B2b (2x2x3)
    # A2: [2, 3, 6, 7]
    # B2b: [12, 13, 14, 17, 18, 19]
    # W: [2, 3, 4, 7, 8, 9]
    add_strassen_2x2x3([2, 3, 6, 7], [12, 13, 14, 17, 18, 19], [2, 3, 4, 7, 8, 9], 25)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
