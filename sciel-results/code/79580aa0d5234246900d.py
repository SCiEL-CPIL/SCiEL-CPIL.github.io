"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose into two 2x2 by 2x8 multiplications
    # Rank of M(2,2,8) using 4x Strassen(2,2,2) is 4 * 7 = 28
    # Total rank = 28 + 28 = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    # Process first block: A1 (2x2) * B1 (2x8) -> C1 (2x8)
    # A1 is rows 0,1 of A (cols 0,1)
    # B1 is rows 0,1 of B (cols 0..7)
    # C1 is rows 0,1 of C (cols 0..7)
    
    # Strassen factors for 2x2 * 2x2
    # We apply this to 4 blocks of B1
    
    def add_strassen_block(a_row_offset, b_row_offset, c_row_offset, b_col_offset):
        nonlocal col
        # A is 2x2 at [a_row_offset, a_row_offset+1] x [0,1]
        # B is 2x2 at [b_row_offset, b_row_offset+1] x [b_col_offset, b_col_offset+1]
        # C is 2x2 at [c_row_offset, c_row_offset+1] x [b_col_offset, b_col_offset+1]
        
        # M1 = (a00+a11)(b00+b10)
        # Indices:
        # A: a00 is (a_row_offset, 0), a11 is (a_row_offset+1, 1)
        # B: b00 is (b_row_offset, b_col_offset), b10 is (b_row_offset+1, b_col_offset)
        # C: c00 is (c_row_offset, b_col_offset)
        
        # U coefficients
        # We need to map Strassen's 7 mults to U, V, W
        # M1: (a00+a11) * (b00+b10)
        U[a_row_offset * m + 0][col] = 1
        U[a_row_offset * m + 1][col] = 1
        V[b_row_offset * p + b_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset + 1][col] = 1
        W[c_row_offset * n + b_col_offset][col] = 1
        W[c_row_offset * n + b_col_offset + 1][col] = 1
        col += 1
        
        # M2: (a10+a11) * b00
        # A: a10 is (a_row_offset+1, 0), a11 is (a_row_offset+1, 1)
        # B: b00 is (b_row_offset, b_col_offset)
        # C: c10 is (c_row_offset+1, b_col_offset)
        U[a_row_offset * m + 1][col] = 1 # a10? No, a10 is row 1 col 0.
        # Wait, A is 2x2.
        # A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11.
        # U index is i*m + j.
        # a10 is i=1, j=0 -> index 1*m + 0.
        # a11 is i=1, j=1 -> index 1*m + 1.
        U[a_row_offset * m + 1][col] = 1 # a10
        U[a_row_offset * m + 1][col] = 1 # a11? No, sum.
        # Correct:
        U[a_row_offset * m + 1][col] = 1
        U[a_row_offset * m + 1][col] = 1 # This overwrites. Need to add.
        # U is a matrix. We are building columns.
        # U[row][col] += val
        
        # Let's rewrite cleanly.
        
    # The above logic is messy. I will just hardcode the 28 factors for one block and replicate.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # A: (a,b; c,d)
    # B: (e,f; g,h)
    # C: (i,j; k,l)
    # M1=(a+d)(e+g) -> i+=M1, l+=M1
    # M2=(c+d)e -> k+=M2
    # M3=a(f-g) -> j+=M3
    # M4=d(h-f) -> i+=M4, k+=M4
    # M5=(a+b)h -> j+=M5, l-=M5
    # M6=(c-a)(e+f) -> l+=M6
    # M7=(b-d)(g+h) -> i+=M7, l+=M7
    
    # Map to indices:
    # A: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C: i=C[0,0], j=C[0,1], k=C[1,0], l=C[1,1]
    
    def add_factor(u_indices, v_indices, w_indices, signs):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx, s in zip(w_indices, signs):
            W[idx][col] += s
        col += 1

    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, cols 0-1) -> C1 (rows 0-1 of C, cols 0-1)
    # A1 indices in U: 0,1,2,3 (i*m+j)
    # B1 indices in V: 0,1,8,9 (j*p+k) -> j=0,1; k=0,1. 0*8+0=0, 0*8+1=1, 1*8+0=8, 1*8+1=9.
    # C1 indices in W: 0,1,8,9 (k*n+i) -> k=0,1; i=0,1. 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
    # Wait, W index is k*n+i. n=2.
    # C[0,0] -> k=0,i=0 -> 0.
    # C[0,1] -> k=1,i=0 -> 2.
    # C[1,0] -> k=0,i=1 -> 1.
    # C[1,1] -> k=1,i=1 -> 3.
    
    # M1: (a+d)(e+g)
    # a=U[0], d=U[3]
    # e=V[0], g=V[8]
    # i=C[0,0]=W[0], l=C[1,1]=W[3]
    add_factor([0, 3], [0, 8], [0, 3], [1, 1])
    
    # M2: (c+d)e
    # c=U[2], d=U[3]
    # e=V[0]
    # k=C[1,0]=W[1]
    add_factor([2, 3], [0], [1], [1])
    
    # M3: a(f-g)
    # a=U[0]
    # f=V[1], g=V[8]
    # j=C[0,1]=W[2]
    add_factor([0], [1, 8], [2], [1, -1]) # V[1]=1, V[8]=-1
    
    # M4: d(h-f)
    # d=U[3]
    # h=V[9], f=V[1]
    # i=W[0], k=W[1]
    add_factor([3], [9, 1], [0, 1], [1, -1])
    
    # M5: (a+b)h
    # a=U[0], b=U[1]
    # h=V[9]
    # j=W[2], l=W[3]
    add_factor([0, 1], [9], [2, 3], [1, -1])
    
    # M6: (c-a)(e+f)
    # c=U[2], a=U[0]
    # e=V[0], f=V[1]
    # l=W[3]
    add_factor([2, 0], [0, 1], [3], [1, -1])
    
    # M7: (b-d)(g+h)
    # b=U[1], d=U[3]
    # g=V[8], h=V[9]
    # i=W[0], l=W[3]
    add_factor([1, 3], [8, 9], [0, 3], [1, -1])
    
    # Repeat for B1 cols 2-3, 4-5, 6-7
    for k_offset in [2, 4, 6]:
        # B indices shift by k_offset
        # V indices: e=V[k_offset], f=V[k_offset+1], g=V[8+k_offset], h=V[9+k_offset]
        # C indices shift by k_offset
        # W indices: i=W[k_offset], j=W[2+k_offset], k=W[1+k_offset], l=W[3+k_offset]
        
        # M1
        add_factor([0, 3], [k_offset, 8+k_offset], [k_offset, 3+k_offset], [1, 1])
        # M2
        add_factor([2, 3], [k_offset], [1+k_offset], [1])
        # M3
        add_factor([0], [k_offset+1, 8+k_offset], [2+k_offset], [1, -1])
        # M4
        add_factor([3], [9+k_offset, k_offset+1], [k_offset, 1+k_offset], [1, -1])
        # M5
        add_factor([0, 1], [9+k_offset], [2+k_offset, 3+k_offset], [1, -1])
        # M6
        add_factor([2, 0], [k_offset, k_offset+1], [3+k_offset], [1, -1])
        # M7
        add_factor([1, 3], [8+k_offset, 9+k_offset], [k_offset, 3+k_offset], [1, -1])
        
    # Block 2: A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B, cols 0-7) -> C2 (rows 0-1 of C, cols 0-7)
    # A2 indices in U: i*m+j. i=0,1. j=2,3.
    # a=U[2], b=U[3], c=U[6], d=U[7]
    # B2 indices in V: j=2,3.
    # e=V[2*8+0]=V[16], f=V[17], g=V[3*8+0]=V[24], h=V[25]
    # C2 indices in W: same as C1? No, C is sum.
    # C = C1 + C2.
    # So W indices are the same.
    
    for k_offset in [0, 2, 4, 6]:
        # M1
        add_factor([2, 7], [16+k_offset, 24+k_offset], [k_offset, 3+k_offset], [1, 1])
        # M2
        add_factor([6, 7], [16+k_offset], [1+k_offset], [1])
        # M3
        add_factor([2], [17+k_offset, 24+k_offset], [2+k_offset], [1, -1])
        # M4
        add_factor([7], [25+k_offset, 17+k_offset], [k_offset, 1+k_offset], [1, -1])
        # M5
        add_factor([2, 3], [25+k_offset], [2+k_offset, 3+k_offset], [1, -1])
        # M6
        add_factor([6, 2], [16+k_offset, 17+k_offset], [3+k_offset], [1, -1])
        # M7
        add_factor([3, 7], [24+k_offset, 25+k_offset], [k_offset, 3+k_offset], [1, -1])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
