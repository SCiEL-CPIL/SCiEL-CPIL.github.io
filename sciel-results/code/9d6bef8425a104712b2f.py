import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = 46  # 6 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 46
    V = [[0] * rank for _ in range(m * p)]   # 26 x 46
    W = [[0] * rank for _ in range(p * n)]   # 26 x 46
    
    col = 0
    
    # Strassen coefficients for (A, B) -> C
    # M1: (a00+a11)(b00+b11)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    # We need to map these to the specific column blocks.
    # For a block starting at column k0, the B columns are k0 and k0+1.
    # B[0, k0] is V[0*p + k0]
    # B[0, k0+1] is V[0*p + k0+1]
    # B[1, k0] is V[1*p + k0]
    # B[1, k0+1] is V[1*p + k0+1]
    
    # C[0, k0] is W[k0*n + 0]
    # C[1, k0] is W[k0*n + 1]
    # C[0, k0+1] is W[(k0+1)*n + 0]
    # C[1, k0+1] is W[(k0+1)*n + 1]
    
    # A indices:
    # A[0,0] is U[0]
    # A[0,1] is U[1]
    # A[1,0] is U[2]
    # A[1,1] is U[3]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Process 6 blocks of 2 columns
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # V indices
        v00 = 0 * p + k0
        v01 = 0 * p + k1
        v10 = 1 * p + k0
        v11 = 1 * p + k1
        
        # W indices
        w00 = k0 * n + 0
        w10 = k0 * n + 1
        w01 = k1 * n + 0
        w11 = k1 * n + 1
        
        # M1: (a00+a11)(b00+b11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1
        add_term(
            {0: 1, 3: 1}, 
            {v00: 1, v11: 1}, 
            {w00: 1, w11: 1}
        )
        
        # M2: (a10+a11)b00
        # U: a10=1, a11=1
        # V: b00=1
        # W: c10=1, c11=-1
        add_term(
            {2: 1, 3: 1}, 
            {v00: 1}, 
            {w10: 1, w11: -1}
        )
        
        # M3: a00(b01-b11)
        # U: a00=1
        # V: b01=1, b11=-1
        # W: c01=1, c11=1
        add_term(
            {0: 1}, 
            {v01: 1, v11: -1}, 
            {w01: 1, w11: 1}
        )
        
        # M4: a11(b10-b00)
        # U: a11=1
        # V: b10=1, b00=-1
        # W: c00=1, c10=1
        add_term(
            {3: 1}, 
            {v10: 1, v00: -1}, 
            {w00: 1, w10: 1}
        )
        
        # M5: (a00+a01)b11
        # U: a00=1, a01=1
        # V: b11=1
        # W: c00=-1, c01=1
        add_term(
            {0: 1, 1: 1}, 
            {v11: 1}, 
            {w00: -1, w01: 1}
        )
        
        # M6: (a10-a00)(b00+b01)
        # U: a10=1, a00=-1
        # V: b00=1, b01=1
        # W: c11=1
        add_term(
            {2: 1, 0: -1}, 
            {v00: 1, v01: 1}, 
            {w11: 1}
        )
        
        # M7: (a01-a11)(b10+b11)
        # U: a01=1, a11=-1
        # V: b10=1, b11=1
        # W: c00=1
        add_term(
            {1: 1, 3: -1}, 
            {v10: 1, v11: 1}, 
            {w00: 1}
        )

    # Process last column k=12
    k = 12
    # Term 1: A00 B0k -> C0k
    add_term({0: 1}, {0 * p + k: 1}, {k * n + 0: 1})
    # Term 2: A01 B1k -> C0k
    add_term({1: 1}, {1 * p + k: 1}, {k * n + 0: 1})
    # Term 3: A10 B0k -> C1k
    add_term({2: 1}, {0 * p + k: 1}, {k * n + 1: 1})
    # Term 4: A11 B1k -> C1k
    add_term({3: 1}, {1 * p + k: 1}, {k * n + 1: 1})

    return {"U": U, "V": V, "W": W}
