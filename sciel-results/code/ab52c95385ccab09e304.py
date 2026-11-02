import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 7 blocks of 2x2 (cols 0-13) using Strassen (rank 7 each) -> 49
    # 1 block of 2x1 (col 14) using Schoolbook (rank 4) -> 4
    # Total Rank = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 53
    V = [[0] * rank for _ in range(m * p)] # 30 x 53
    W = [[0] * rank for _ in range(p * n)] # 30 x 53
    
    # Helper to set values
    def set_U(idx, val):
        U[idx][r] = val
    def set_V(idx, val):
        V[idx][r] = val
    def set_W(idx, val):
        W[idx][r] = val

    # Indices for A (n*m)
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # Indices for B (m*p)
    # B[j, k] = j*p + k
    # Block l (0..6) uses k = 2*l, 2*l+1
    # Block 7 uses k = 14
    
    # Indices for C (p*n)
    # C[k, i] = k*n + i  (Trace convention W[k*n+i])
    
    # Strassen coefficients
    # M1: A(00+11), B(00+11)
    # M2: A(10+11), B(00)
    # M3: A(00), B(01-11)
    # M4: A(11), B(10-00)
    # M5: A(00+01), B(11)
    # M6: A(10-00), B(00+01)
    # M7: A(01-11), B(10+11)
    
    # Reconstruction
    # C00: M1 - M4 - M6 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M3 + M6 - M7
    
    # We need to add these contributions to W for each block.
    # Since W is a sum of rank-1 terms, each term r adds its W vector.
    
    def add_strassen_block(l):
        # l is 0..6
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # B indices
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        
        # C indices (for this block's output columns k0, k1)
        # C[k, i]
        c00 = k0 * n + 0
        c01 = k0 * n + 1
        c10 = k1 * n + 0
        c11 = k1 * n + 1
        
        base_r = l * 7
        
        # M1
        r = base_r
        set_U(0, 1); set_U(3, 1)
        set_V(b00, 1); set_V(b11, 1)
        set_W(c00, 1); set_W(c10, -1); set_W(c11, 1); set_W(c11, -1) # Wait, C11 is 1?
        # C00: +1, C10: -1, C11: +1?
        # C00 = M1 ...
        # C11 = M1 ...
        # So W(c00)=1, W(c11)=1.
        # What about C10 and C01?
        # C10 = M2 + M4. So M1 contributes 0 to C10.
        # C01 = M3 + M5. So M1 contributes 0 to C01.
        set_W(c00, 1)
        set_W(c11, 1)
        
        # M2
        r = base_r + 1
        set_U(2, 1); set_U(3, 1)
        set_V(b00, 1)
        set_W(c10, 1)
        
        # M3
        r = base_r + 2
        set_U(0, 1)
        set_V(b01, 1); set_V(b11, -1)
        set_W(c01, 1)
        set_W(c11, -1)
        
        # M4
        r = base_r + 3
        set_U(3, 1)
        set_V(b10, 1); set_V(b00, -1)
        set_W(c00, -1)
        set_W(c10, 1)
        
        # M5
        r = base_r + 4
        set_U(0, 1); set_U(1, 1)
        set_V(b11, 1)
        set_W(c01, 1)
        
        # M6
        r = base_r + 5
        set_U(2, 1); set_U(0, -1)
        set_V(b00, 1); set_V(b01, 1)
        set_W(c00, -1)
        set_W(c11, 1)
        
        # M7
        r = base_r + 6
        set_U(1, 1); set_U(3, -1)
        set_V(b10, 1); set_V(b11, 1)
        set_W(c00, 1)
        set_W(c11, -1)

    for l in range(7):
        add_strassen_block(l)
        
    # Last column (k=14)
    k = 14
    base_r = 49
    
    # Schoolbook for 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 1: A00 B0 -> C0
    r = base_r
    set_U(0, 1)
    set_V(0 * p + k, 1)
    set_W(k * n + 0, 1)
    
    # Term 2: A01 B1 -> C0
    r = base_r + 1
    set_U(1, 1)
    set_V(1 * p + k, 1)
    set_W(k * n + 0, 1)
    
    # Term 3: A10 B0 -> C1
    r = base_r + 2
    set_U(2, 1)
    set_V(0 * p + k, 1)
    set_W(k * n + 1, 1)
    
    # Term 4: A11 B1 -> C1
    r = base_r + 3
    set_U(3, 1)
    set_V(1 * p + k, 1)
    set_W(k * n + 1, 1)
    
    return {"U": U, "V": V, "W": W}
