import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (a+d)(e+h)
    # A: a(0), d(3) -> idx 0, 3
    # B: e(0), h(3) -> idx 0, 3
    # C: c11(0,0), c22(1,1) -> W idx k*2+i
    # c11: k=0, i=0 -> idx 0. Coeff 1
    # c22: k=1, i=1 -> idx 3. Coeff 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M2 = (c+d)e
    # A: c(2), d(3) -> idx 2, 3
    # B: e(0) -> idx 0
    # C: c21(1,0), c22(1,1)
    # c21: k=0, i=1 -> idx 2. Coeff 1
    # c22: k=1, i=1 -> idx 3. Coeff -1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3 = a(f-h)
    # A: a(0) -> idx 0
    # B: f(1), h(3) -> idx 1, 3
    # C: c12(0,1), c22(1,1)
    # c12: k=1, i=0 -> idx 1. Coeff 1
    # c22: k=1, i=1 -> idx 3. Coeff 1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4 = d(g-e)
    # A: d(3) -> idx 3
    # B: g(2), e(0) -> idx 2, 0
    # C: c11(0,0), c21(1,0)
    # c11: k=0, i=0 -> idx 0. Coeff 1
    # c21: k=0, i=1 -> idx 2. Coeff 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: M5 = (a+b)h
    # A: a(0), b(1) -> idx 0, 1
    # B: h(3) -> idx 3
    # C: c11(0,0), c12(0,1)
    # c11: k=0, i=0 -> idx 0. Coeff -1
    # c12: k=1, i=0 -> idx 1. Coeff 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: M6 = (c-a)(e+f)
    # A: c(2), a(0) -> idx 2, 0
    # B: e(0), f(1) -> idx 0, 1
    # C: c22(1,1)
    # c22: k=1, i=1 -> idx 3. Coeff 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: M7 = (b-d)(g+h)
    # A: b(1), d(3) -> idx 1, 3
    # B: g(2), h(3) -> idx 2, 3
    # C: c11(0,0)
    # c11: k=0, i=0 -> idx 0. Coeff 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Kronecker Product
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permute rows to match i*4+j indexing
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for idx in range(16):
        # U4 index i*4+j
        i = idx // 4
        j = idx % 4
        i1 = i // 2
        i0 = i % 2
        j1 = j // 2
        j0 = j % 2
        # R = (2*i1 + j1)*4 + (2*i0 + j0)
        R_U = (2 * i1 + j1) * 4 + (2 * i0 + j0)
        U4[idx, :] = U4_kron[R_U, :]
        
        # V4 index j*4+k
        # Here idx is j*4+k
        # j is block row? No, j is row of B.
        # B is 4x4. j is row, k is col.
        # j = 2*u + x ? No.
        # In V2, index is 2*row + col.
        # For B, row j, col k.
        # j = 2*u + x, k = 2*v + y.
        # u = j//2, x = j%2.
        # v = k//2, y = k%2.
        # R_V = (2*u + v)*4 + (2*x + y)
        j_val = idx // 4
        k_val = idx % 4
        u = j_val // 2
        x = j_val % 2
        v = k_val // 2
        y = k_val % 2
        R_V = (2 * u + v) * 4 + (2 * x + y)
        V4[idx, :] = V4_kron[R_V, :]
        
        # W4 index k*4+i
        # k is col of C, i is row of C.
        # k = 2*v + y, i = 2*u + x.
        # v = k//2, y = k%2.
        # u = i//2, x = i%2.
        # R_W = (2*v + u)*4 + (2*y + x)
        k_val = idx // 4
        i_val = idx % 4
        v = k_val // 2
        y = k_val % 2
        u = i_val // 2
        x = i_val % 2
        R_W = (2 * v + u) * 4 + (2 * y + x)
        W4[idx, :] = W4_kron[R_W, :]
        
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
