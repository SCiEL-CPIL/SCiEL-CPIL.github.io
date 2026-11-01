"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 2x2x2 blocks and schoolbook for the remaining 2x2x1 block
    # 13 columns = 6 * 2 + 1
    # Rank for 2x2x2 is 7
    # Rank for 2x2x1 is 4
    # Total rank = 6 * 7 + 4 = 46
    
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 46
    V = [[0] * rank for _ in range(m * p)] # 26 x 46
    W = [[0] * rank for _ in range(p * n)] # 26 x 46
    
    col = 0
    
    # Define Strassen coefficients
    # M1: (a00+a11)(b00+b11)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Process 6 blocks of 2 columns
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # V indices: j*p + k
        # j=0 -> k, j=1 -> 13+k
        # W indices: k*n + i
        # i=0 -> 2k, i=1 -> 2k+1
        # U indices: i*m + j
        # 00:0, 01:1, 10:2, 11:3
        
        # M1
        add_term(
            {0: 1, 3: 1}, 
            {k0: 1, 13+k1: 1}, 
            {2*k0: 1, 2*k1+1: 1}
        )
        
        # M2
        add_term(
            {2: 1, 3: 1}, 
            {k0: 1}, 
            {2*k0+1: 1, 2*k1+1: -1}
        )
        
        # M3
        add_term(
            {0: 1}, 
            {k1: 1, 13+k1: -1}, 
            {2*k1: 1, 2*k1+1: 1}
        )
        
        # M4
        add_term(
            {3: 1}, 
            {13+k0: 1, k0: -1}, 
            {2*k0: 1, 2*k0+1: 1}
        )
        
        # M5
        add_term(
            {0: 1, 1: 1}, 
            {13+k1: 1}, 
            {2*k0: -1, 2*k1: 1}
        )
        
        # M6
        add_term(
            {2: 1, 0: -1}, 
            {k0: 1, k1: 1}, 
            {2*k1+1: 1}
        )
        
        # M7
        add_term(
            {1: 1, 3: -1}, 
            {13+k0: 1, 13+k1: 1}, 
            {2*k0: 1}
        )
        
    # Process last column k=12
    k = 12
    # Term 1: A00 B0k -> C0k
    add_term({0: 1}, {k: 1}, {2*k: 1})
    # Term 2: A01 B1k -> C0k
    add_term({1: 1}, {13+k: 1}, {2*k: 1})
    # Term 3: A10 B0k -> C1k
    add_term({2: 1}, {k: 1}, {2*k+1: 1})
    # Term 4: A11 B1k -> C1k
    add_term({3: 1}, {13+k: 1}, {2*k+1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
