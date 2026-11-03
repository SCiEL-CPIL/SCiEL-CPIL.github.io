"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Laderman's algorithm for the 4x4 sub-block and schoolbook for the rest.
    # Rank of 4x4 is 21.
    # Rank of cross terms:
    # A[0:4, 0:4] * B[0:4, 4] -> 16
    # A[4, 0:4] * B[0:4, 0:4] -> 16
    # A[0:4, 4] * B[4, 0:4] -> 4
    # A[0:4, 4] * B[4, 4] -> 4
    # A[4, 0:4] * B[4, 4] -> 4
    # A[4, 4] * B[4, 4] -> 1
    # Total rank = 21 + 16 + 16 + 4 + 4 + 4 + 1 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Laderman 4x4 decomposition
    # Indices 0..3 for the 4x4 block
    # z1 = a00(b33-b23)
    # z2 = a01(b00-b01)
    # z3 = (a00+a10)b01
    # z4 = (a11+a01)b10
    # z5 = (a11-a00)(b00+b11)
    # z6 = (a00+a01-a11-a10)b11
    # z7 = (a10-a00+a11)b00
    
    # We map the 4x4 Laderman to the 5x5 indices 0..3
    # U indices: i*m + j -> i*5 + j
    # V indices: j*p + k -> j*5 + k
    # W indices: k*n + i -> k*5 + i
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # Laderman terms for 4x4 block (indices 0-3)
    # z1
    add_term(0*5+0, 3*5+3, 3*5+0, 1, 1, 1) # a00 b33
    add_term(0*5+0, 2*5+3, 3*5+0, 1, -1, 1) # a00 b23
    # z2
    add_term(0*5+1, 0*5+0, 0*5+0, 1, 1, 1) # a01 b00
    add_term(0*5+1, 0*5+1, 0*5+0, 1, -1, 1) # a01 b01
    # z3
    add_term(0*5+0, 0*5+1, 1*5+0, 1, 1, 1) # a00 b01
    add_term(1*5+0, 0*5+1, 1*5+0, 1, 1, 1) # a10 b01
    # z4
    add_term(1*5+1, 1*5+0, 0*5+1, 1, 1, 1) # a11 b10
    add_term(0*5+1, 1*5+0, 0*5+1, 1, 1, 1) # a01 b10
    # z5
    add_term(1*5+1, 0*5+0, 0*5+1, 1, 1, 1) # a11 b00
    add_term(1*5+1, 1*5+1, 0*5+1, 1, 1, 1) # a11 b11
    add_term(0*5+0, 0*5+0, 0*5+1, 1, -1, 1) # a00 b00
    add_term(0*5+0, 1*5+1, 0*5+1, 1, -1, 1) # a00 b11
    # z6
    add_term(0*5+0, 1*5+1, 1*5+0, 1, 1, 1) # a00 b11
    add_term(0*5+1, 1*5+1, 1*5+0, 1, 1, 1) # a01 b11
    add_term(1*5+1, 1*5+1, 1*5+0, 1, -1, 1) # a11 b11
    add_term(1*5+0, 1*5+1, 1*5+0, 1, -1, 1) # a10 b11
    # z7
    add_term(1*5+0, 0*5+0, 0*5+0, 1, 1, 1) # a10 b00
    add_term(0*5+0, 0*5+0, 0*5+0, 1, -1, 1) # a00 b00
    add_term(1*5+1, 0*5+0, 0*5+0, 1, 1, 1) # a11 b00
    
    # Wait, the Laderman decomposition above is not exactly 21 terms?
    # z1: 2 terms
    # z2: 2 terms
    # z3: 2 terms
    # z4: 2 terms
    # z5: 4 terms
    # z6: 4 terms
    # z7: 3 terms
    # Total: 19 terms?
    # Laderman is 7 multiplications, but each multiplication is a product of sums.
    # The rank is the number of scalar multiplications.
    # My "add_term" adds a single scalar multiplication.
    # So the rank of the 4x4 block is 19? No, Laderman is rank 21.
    # The standard Laderman decomposition has 21 terms.
    # I will just use the schoolbook for the 4x4 block to be safe, then the rank is 16.
    # Total rank = 16 + 16 + 16 + 4 + 4 + 4 + 1 = 61.
    # This is simpler and valid.
    
    # Reset
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term_simple(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 4x4 block schoolbook (16 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term_simple(i*5+j, j*5+k, k*5+i)
                
    # Cross terms
    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for k in range(4):
            add_term_simple(i*5+k, k*5+4, 4*5+i)
            
    # A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    for j in range(4):
        for k in range(4):
            add_term_simple(4*5+j, j*5+k, k*5+4)
            
    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            add_term_simple(i*5+4, 4*5+j, j*5+i)
            
    # A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        add_term_simple(i*5+4, 4*5+4, 4*5+i)
        
    # A[4, 0:4] * B[4, 4] -> C[4, 0:4]
    for j in range(4):
        add_term_simple(4*5+j, 4*5+4, 4*5+j)
        
    # A[4, 4] * B[4, 4] -> C[4, 4]
    add_term_simple(4*5+4, 4*5+4, 4*5+4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
