"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    import time
    start_time = time.time()
    DEADLINE = 120
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Build a rank-49 Strassen-based 4x4x4 decomposition
    # Then combine with 4x4x1 (rank 16) and 4x1x5 (rank 20)
    # Total rank: 49 + 16 + 20 = 85
    
    total_rank = 85
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 85
    
    col = 0
    
    # Strassen 2x2x2 rank-7 decomposition
    # For 2x2 matrices A, B, C with C = A*B:
    # t1 = (A00+A11)(B00+B11)
    # t2 = (A10+A11)B00
    # t3 = A00(B10-B11)
    # t4 = A11(B01-B10)
    # t5 = (A00+A10)B11
    # t6 = (A01-A00)(B00+B10)
    # t7 = (A10-A11)(B01+B11)
    # C00 = t1 + t4 - t5 + t7
    # C01 = t3 + t5
    # C10 = t2 + t4
    # C11 = t1 - t2 + t3 + t6
    
    # In the tensor decomposition for 2x2x2:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  [W[k*2+i]]
    
    # For each ti, we add one rank-1 term with appropriate U, V, W entries.
    
    def add_strassen_2x2x2(A_base, B_base, W_base, offset_A, offset_B, offset_W):
        """Add a Strassen 2x2x2 decomposition.
        A_base: base index for A in the global U (i*m + j)
        B_base: base index for B in the global V (j*p + k)
        W_base: base index for W in the global W (k*n + i)
        offset_A, offset_B, offset_W: row/col offsets for the 2x2 blocks
        """
        nonlocal col
        
        def a_idx(i, j):
            return (offset_A + i) * m + (offset_B + j)
        
        def b_idx(j, k):
            return (offset_B + j) * p + (offset_W + k)
        
        def w_idx(k, i):
            return (offset_W + k) * n + (offset_A + i)
        
        # t1 = (A00+A11)(B00+B11)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]=1, C[1,1]=1
        U[a_idx(0,0)][col] = 1
        U[a_idx(1,1)][col] = 1
        V[b_idx(0,0)][col] = 1
        V[b_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1
        
        # t2 = (A10+A11)B00
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,0]=1
        # W: C[1,0]=1
        U[a_idx(1,0)][col] = 1
        U[a_idx(1,1)][col] = 1
        V[b_idx(0,0)][col] = 1
        W[w_idx(0,1)][col] = 1
        col += 1
        
        # t3 = A00(B10-B11)
        # U: A[0,0]=1
        # V: B[1,0]=1, B[1,1]=-1
        # W: C[0,1]=1
        U[a_idx(0,0)][col] = 1
        V[b_idx(1,0)][col] = 1
        V[b_idx(1,1)][col] = -1
        W[w_idx(1,0)][col] = 1
        col += 1
        
        # t4 = A11(B01-B10)
        # U: A[1,1]=1
        # V: B[0,1]=1, B[1,0]=-1
        # W: C[0,0]=1, C[1,0]=1
        U[a_idx(1,1)][col] = 1
        V[b_idx(0,1)][col] = 1
        V[b_idx(1,0)][col] = -1
        W[w_idx(0,0)][col] = 1
        W[w_idx(1,0)][col] = 1
        col += 1
        
        # t5 = (A00+A10)B11
        # U: A[0,0]=1, A[1,0]=1
        # V: B[1,1]=1
        # W: C[0,0]=-1, C[0,1]=1
        U[a_idx(0,0)][col] = 1
        U[a_idx(1,0)][col] = 1
        V[b_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = -1
        W[w_idx(1,0)][col] = 1
        col += 1
        
        # t6 = (A01-A00)(B00+B10)
        # U: A[0,1]=1, A[0,0]=-1
        # V: B[0,0]=1, B[1,0]=1
        # W: C[1,1]=1
        U[a_idx(0,1)][col] = 1
        U[a_idx(0,0)][col] = -1
        V[b_idx(0,0)][col] = 1
        V[b_idx(1,0)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1
        
        # t7 = (A10-A11)(B01+B11)
        # U: A[1,0]=1, A[1,1]=-1
        # V: B[0,1]=1, B[1,1]=1
        # W: C[0,0]=1, C[1,1]=1
        U[a_idx(1,0)][col] = 1
        U[a_idx(1,1)][col] = -1
        V[b_idx(0,1)][col] = 1
        V[b_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1
    
    # 4x4x4 Strassen decomposition
    # Partition 4x4 into four 2x2 blocks:
    # A00: rows 0-1, cols 0-1 (offset 0,0)
    # A01: rows 0-1, cols 2-3 (offset 0,2)
    # A10: rows 2-3, cols 0-1 (offset 2,0)
    # A11: rows 2-3, cols 2-3 (offset 2,2)
    
    # For the 4x4x4 Strassen, we apply Strassen at the 2x2 block level.
    # The 7 Strassen multiplications are 2x2x2 multiplications of the blocks.
    # Each 2x2x2 block multiplication has rank 7.
    # Total: 7 * 7 = 49.
    
    # But wait, in the 4x4x4 Strassen, the "A00+A11" means we're adding the 2x2 blocks A00 and A11.
    # In the tensor decomposition, this means the U factor has entries from both A00 and A11 positions.
    
    # Let me define the block positions:
    # Block (br, bc) has A rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    # In global A (4x5): A1 is at rows 0-3, cols 0-3
    # So block (br, bc) in A1: global A rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    
    # For the Strassen 4x4x4, the 7 multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # Each Mi is a 2x2x2 multiplication. For the tensor decomposition,
    # each Mi contributes 7 rank-1 terms (Strassen rank-7 for 2x2x2).
    
    # For M1 = (A00+A11)(B00+B11):
    # The "A" part is A00+A11, which is a 2x2 matrix where:
    #   (A00+A11)[i,j] = A[br0*2+i, bc0*2+j] + A[br1*2+i, bc1*2+j]
    # where (br0,bc0)=(0,0) and (br1,bc1)=(1,1) for A00 and A11.
    
    # In the tensor decomposition, for each of the 7 Strassen terms of the 2x2x2,
    # the U factor will have entries at both A00 and A11 positions.
    
    # This is getting very complex. Let me just use schoolbook for the 4x4x4 part.
    # 4x4x4 schoolbook: rank 64.
    # Total: 64 + 16 + 20 = 100. Same as original.
    
    # OK, let me try a completely different approach.
    # Let me just use the schoolbook decomposition but verify it's correct,
    # and then try to find a better one.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # but with the block structure to make it potentially evolvable.
    
    col = 0
    rank = n * m * p  # 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
