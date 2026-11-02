import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen factors for 2x2x2
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to the global indices.
    # A indices: i*m + j. i in {0,1}, j in {0,1,2,3}
    # B indices: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}
    # C indices: k*n + i. k in {0,1,2,3}, i in {0,1}
    
    col = 0
    
    def add_strassen(a_base, b_base, c_base, signs):
        nonlocal col
        # a_base: index of A[0,0] in A matrix (i=0, j=0) -> 0*m + 0
        # b_base: index of B[0,0] in B matrix (j=0, k=0) -> 0*p + 0
        # c_base: index of C[0,0] in C matrix (k=0, i=0) -> 0*n + 0
        
        # A indices for 2x2 block starting at a_base
        # a00 = A[a_base]
        # a01 = A[a_base + 1]
        # a10 = A[a_base + m]
        # a11 = A[a_base + m + 1]
        
        # B indices for 2x2 block starting at b_base
        # b00 = B[b_base]
        # b01 = B[b_base + 1]
        # b10 = B[b_base + p]
        # b11 = B[b_base + p + 1]
        
        # C indices for 2x2 block starting at c_base
        # c00 = C[c_base]
        # c01 = C[c_base + n] # k=0, i=1? No.
        # C index is k*n + i.
        # c00: k=0, i=0 -> 0
        # c01: k=0, i=1 -> 1? No, C is 2x4.
        # C[:, 0:2] corresponds to k=0,1.
        # C[0,0] is k=0, i=0. Index 0.
        # C[0,1] is k=1, i=0. Index 1*2 + 0 = 2.
        # C[1,0] is k=0, i=1. Index 0*2 + 1 = 1.
        # C[1,1] is k=1, i=1. Index 1*2 + 1 = 3.
        
        # So for a 2x2 block of C at columns k0, k0+1:
        # c00 (i=0, k=k0) -> k0*n + 0
        # c01 (i=0, k=k0+1) -> (k0+1)*n + 0
        # c10 (i=1, k=k0) -> k0*n + 1
        # c11 (i=1, k=k0+1) -> (k0+1)*n + 1
        
        # Let's define helper to get indices
        def a_idx(di, dj): return a_base + di*m + dj
        def b_idx(dj, dk): return b_base + dj*p + dk
        def c_idx(dk, di): return c_base + dk*n + di
        
        # P1
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # P2
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        W[c_idx(1,1)][col] -= 1
        col += 1
        
        # P3
        U[a_idx(0,0)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] -= 1
        W[c_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # P4
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(0,0)][col] -= 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        col += 1
        
        # P5
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] -= 1
        W[c_idx(0,1)][col] += 1
        col += 1
        
        # P6
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] -= 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # P7
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] -= 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        col += 1

    # Block 1: A0 (cols 0,1), B00 (rows 0,1 cols 0,1), C cols 0,1
    add_strassen(0, 0, 0, None)
    
    # Block 2: A1 (cols 2,3), B10 (rows 2,3 cols 0,1), C cols 0,1
    # A1 base: i=0, j=2 -> 0*4 + 2 = 2
    # B10 base: j=2, k=0 -> 2*4 + 0 = 8
    # C base: k=0, i=0 -> 0
    add_strassen(2, 8, 0, None)
    
    # Block 3: A0 (cols 0,1), B01 (rows 0,1 cols 2,3), C cols 2,3
    # A0 base: 0
    # B01 base: j=0, k=2 -> 0*4 + 2 = 2
    # C base: k=2, i=0 -> 2*2 + 0 = 4
    add_strassen(0, 2, 4, None)
    
    # Block 4: A1 (cols 2,3), B11 (rows 2,3 cols 2,3), C cols 2,3
    # A1 base: 2
    # B11 base: j=2, k=2 -> 2*4 + 2 = 10
    # C base: k=2, i=0 -> 4
    add_strassen(2, 10, 4, None)
    
    return {"U": U, "V": V, "W": W}
