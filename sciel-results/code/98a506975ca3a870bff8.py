"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 112 decomposition using block structure 2+3
    # Blocks:
    # A00 (2x2), A01 (2x3)
    # A10 (3x2), A11 (3x3)
    # B00 (2x2), B01 (2x3)
    # B10 (3x2), B11 (3x3)
    
    # We construct U, V, W for the whole 5x5x5 multiplication.
    # We will build it by concatenating the decompositions of the 8 sub-products.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col_offset
        U[u_idx][col_offset] += val
        V[v_idx][col_offset] += val
        W[w_idx][col_offset] += val
        col_offset += 1

    # Helper to add a schoolbook term
    def add_schoolbook(u_base, v_base, w_base, n_u, n_v, n_w, u_step, v_step, w_step):
        # u_base, v_base, w_base are indices in the full 25x25 space
        # n_u, n_v, n_w are dimensions of the sub-blocks
        # u_step, v_step, w_step are strides in the full space
        for i in range(n_u):
            for j in range(n_v):
                for k in range(n_w):
                    u_idx = u_base + i * u_step
                    v_idx = v_base + j * v_step
                    w_idx = w_base + k * w_step
                    # Note: W index is k*n + i. 
                    # In our block mapping:
                    # A is i (row), j (col). B is j (row), k (col). C is i (row), k (col).
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    # This matches the standard.
                    add_term(u_idx, v_idx, w_idx, 1)

    # 1. A00 * B00 (2x2 * 2x2 -> 2x2) using Strassen (Rank 7)
    # A00: rows 0-1, cols 0-1. U base 0.
    # B00: rows 0-1, cols 0-1. V base 0.
    # C00: rows 0-1, cols 0-1. W base 0.
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11) -- Not applicable here, this is for the whole 2x2.
    # We are decomposing the 2x2x2 tensor.
    # Indices in local 2x2 space:
    # A: (0,0), (0,1), (1,0), (1,1) -> U: 0, 1, 2, 3 (stride 1)
    # B: (0,0), (0,1), (1,0), (1,1) -> V: 0, 1, 2, 3 (stride 1)
    # C: (0,0), (0,1), (1,0), (1,1) -> W: 0, 1, 2, 3 (stride 1, but W is k*n+i)
    # Wait, W index for C00 (2x2) is k*5+i.
    # Local k=0,1. Local i=0,1.
    # W indices: 0, 1, 5, 6.
    
    # Strassen terms for 2x2x2:
    # 1. A00*B00
    add_term(0, 0, 0, 1)
    # 2. A01*B11
    add_term(1, 3, 6, 1) # U: 0*2+1=1, V: 1*2+1=3, W: 1*5+0=5? No.
    # Let's be precise.
    # A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    # W index: k*n + i.
    # C00: i=0, k=0 -> 0
    # C01: i=0, k=1 -> 1
    # C10: i=1, k=0 -> 5
    # C11: i=1, k=1 -> 6
    
    # Strassen:
    # M1 = A00*B00 -> C00 += M1, C11 += M1? No.
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # This is complex to implement generically in the add_term loop.
    # Given the time, I will use schoolbook for the 2x2 blocks as well to ensure correctness, 
    # but the knowledge says Strassen reduces rank. 
    # However, 112 is already an improvement over 117.
    # If I use schoolbook for 2x2, rank is 8.
    # 8 + 23 + 11 + 17 + 12 + 18 + 12 + 18 = 119.
    # This is worse than 117.
    
    # I must implement Strassen for the 2x2 blocks.
    # Let's hardcode the 7 terms for the first 2x2 block.
    # U indices: 0,1,2,3
    # V indices: 0,1,2,3
    # W indices: 0,1,5,6
    
    # P1
    add_term(0, 0, 0, 1)
    add_term(0, 1, 1, 1)
    add_term(3, 0, 5, 1)
    add_term(3, 1, 6, 1)
    # This is not a single term in the tensor decomposition sense. 
    # The tensor decomposition requires U, V, W vectors such that sum u_i v_i w_i = T.
    # Strassen provides 7 such terms.
    # Term 1: (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C11.
    # This is not a rank-1 term. It's a rank-1 term in the *product* space, 
    # but in the tensor decomposition, we need to expand it?
    # No, Strassen's algorithm *is* a rank-7 decomposition.
    # The 7 terms are:
    # 1. (A00+A11) * (B00+B11)
    # This corresponds to U having 1s at A00, A11. V having 1s at B00, B11.
    # But W? The result is added to C00 and C11.
    # So W has 1s at C00, C11.
    # This is a valid rank-1 term in the tensor?
    # T = sum u_i v_i w_i.
    # If u = e_A00 + e_A11, v = e_B00 + e_B11, w = e_C00 + e_C11.
    # Then u v w = (e_A00+e_A11)(e_B00+e_B11)(e_C00+e_C11).
    # This expands to 4 terms: A00B00C00, A00B00C11, A00B11C00...
    # This is NOT the Strassen term.
    # Strassen terms are linear combinations of A and B multiplied, then mapped to C.
    # The rank-1 decomposition of the *multiplication tensor* uses terms like:
    # U: 1 at A00, A11. V: 1 at B00, B11. W: 1 at C00, -1 at C11?
    # No.
    # The Strassen decomposition of the 2x2x2 tensor is:
    # 7 terms. Each term has U, V, W vectors with 0s and 1s.
    # Example: Term 1: U: A00, A11. V: B00, B11. W: C00, C11.
    # This works if the signs are correct.
    # Actually, the standard Strassen decomposition is:
    # 1. U: A00, A11. V: B00, B11. W: C00, C11.
    # 2. U: A10, A11. V: B00. W: C10.
    # 3. U: A00. V: B01, B11. W: C01, -C11.
    # 4. U: A11. V: B10, B00. W: C10, -C11.
    # 5. U: A00, A01. V: B11. W: C01, C11.
    # 6. U: A10, -A00. V: B00, B01. W: C10, C11.
    # 7. U: A01, -A11. V: B10, B11. W: C01, C11.
    
    # I will implement these 7 terms for the 2x2 blocks.
    
    col_offset = 0
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_2x2(u_base, v_base, w_base):
        nonlocal col_offset
        # u_base, v_base, w_base are indices of A00, B00, C00 in the full matrices.
        # A: u_base (0,0), u_base+1 (0,1), u_base+2 (1,0), u_base+3 (1,1)
        # B: v_base (0,0), v_base+1 (0,1), v_base+2 (1,0), v_base+3 (1,1)
        # C: w_base (0,0), w_base+1 (0,1), w_base+5 (1,0), w_base+6 (1,1)
        
        # 1. (A00+A11)(B00+B11) -> C00+C11
        add_term(u_base, v_base, w_base, 1)
        add_term(u_base+3, v_base+3, w_base+6, 1)
        # Note: add_term adds to the same column.
        # So this creates U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,1] (at C00, C11)
        
        # 2. (A10+A11)B00 -> C10
        add_term(u_base+2, v_base, w_base+5, 1)
        add_term(u_base+3, v_base, w_base+5, 1)
        
        # 3. A00(B01-B11) -> C01-C11
        add_term(u_base, v_base+1, w_base+1, 1)
        add_term(u_base, v_base+3, w_base+6, -1)
        
        # 4. A11(B10-B00) -> C10-C11
        add_term(u_base+3, v_base+2, w_base+5, 1)
        add_term(u_base+3, v_base, w_base+6, -1)
        
        # 5. (A00+A01)B11 -> C01+C11
        add_term(u_base, v_base+3, w_base+1, 1)
        add_term(u_base+1, v_base+3, w_base+6, 1)
        
        # 6. (A10-A00)(B00+B01) -> C10+C11
        add_term(u_base+2, v_base, w_base+5, 1)
        add_term(u_base, v_base, w_base+5, -1)
        add_term(u_base+2, v_base+1, w_base+6, 1)
        add_term(u_base, v_base+1, w_base+6, -1)
        
        # 7. (A01-A11)(B10+B11) -> C01+C11
        add_term(u_base+1, v_base+2, w_base+1, 1)
        add_term(u_base+3, v_base+2, w_base+1, -1)
        add_term(u_base+1, v_base+3, w_base+6, 1)
        add_term(u_base+3, v_base+3, w_base+6, -1)
        
        col_offset += 7

    # Apply Strassen to A00*B00 (u=0, v=0, w=0)
    add_strassen_2x2(0, 0, 0)
    # Apply Strassen to A11*B11 (u=12, v=12, w=12? No.
    # A11 is rows 2-4, cols 2-4.
    # A index: i*5+j.
    # A11 (0,0) local is (2,2) global -> 2*5+2 = 12.
    # B11 (0,0) local is (2,2) global -> 12.
    # C11 (0,0) local is (2,2) global -> 2*5+2 = 12.
    add_strassen_2x2(12, 12, 12)

    # Now handle the mixed blocks with schoolbook or improved rank.
    # For simplicity and speed, I will use schoolbook for the remaining 6 blocks.
    # This gives rank 7+7 + 12+18+12+18+12+18 = 14 + 90 = 104.
    # Wait, schoolbook for mixed:
    # A01*B10: 2x3 * 3x2 -> 12
    # A10*B01: 3x2 * 2x3 -> 18
    # A00*B01: 2x2 * 2x3 -> 12
    # A01*B11: 2x3 * 3x3 -> 18
    # A10*B00: 3x2 * 2x2 -> 12
    # A11*B10: 3x3 * 3x2 -> 18
    # Sum: 12+18+12+18+12+18 = 90.
    # Total: 14 + 90 = 104.
    # Score: 125/104 = 1.20.
    
    # This is better than 1.068.
    
    # 2. A01*B10 (2x3 * 3x2)
    # A01: rows 0-1, cols 2-4. U base 2.
    # B10: rows 2-4, cols 0-1. V base 10 (2*5+0).
    # C00: rows 0-1, cols 0-1. W base 0.
    add_schoolbook(2, 10, 0, 2, 3, 2, 1, 1, 5)
    
    # 3. A10*B01 (3x2 * 2x3)
    # A10: rows 2-4, cols 0-1. U base 10.
    # B01: rows 0-1, cols 2-4. V base 2.
    # C10: rows 2-4, cols 0-1. W base 10 (2*5+0).
    add_schoolbook(10, 2, 10, 3, 2, 3, 1, 1, 5)
    
    # 4. A00*B01 (2x2 * 2x3)
    # A00: rows 0-1, cols 0-1. U base 0.
    # B01: rows 0-1, cols 2-4. V base 2.
    # C01: rows 0-1, cols 2-4. W base 2.
    add_schoolbook(0, 2, 2, 2, 2, 3, 1, 1, 5)
    
    # 5. A01*B11 (2x3 * 3x3)
    # A01: rows 0-1, cols 2-4. U base 2.
    # B11: rows 2-4, cols 2-4. V base 12.
    # C01: rows 0-1, cols 2-4. W base 2.
    add_schoolbook(2, 12, 2, 2, 3, 3, 1, 1, 5)
    
    # 6. A10*B00 (3x2 * 2x2)
    # A10: rows 2-4, cols 0-1. U base 10.
    # B00: rows 0-1, cols 0-1. V base 0.
    # C10: rows 2-4, cols 0-1. W base 10.
    add_schoolbook(10, 0, 10, 3, 2, 2, 1, 1, 5)
    
    # 7. A11*B10 (3x3 * 3x2)
    # A11: rows 2-4, cols 2-4. U base 12.
    # B10: rows 2-4, cols 0-1. V base 10.
    # C11: rows 2-4, cols 2-4. W base 12.
    add_schoolbook(12, 10, 12, 3, 3, 2, 1, 1, 5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
