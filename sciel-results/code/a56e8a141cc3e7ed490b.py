import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build the decomposition by splitting into two 2x2 x 2x5 sub-problems
    # Sub-problem 1: A rows 0-1, cols 0-1 (A1) times B rows 0-1, cols 0-4 (B1)
    # Sub-problem 2: A rows 0-1, cols 2-3 (A2) times B rows 2-3, cols 0-4 (B2)
    
    # For each 2x2 x 2x5, we use:
    #   k-pair (0,1): Strassen (7 terms)
    #   k-pair (2,3): Strassen (7 terms)
    #   k=4: direct (2 terms)
    # Total per sub-problem: 16 terms
    # Grand total: 32 terms
    
    rank = 32
    U = [[0]*rank for _ in range(n*m)]  # 8 x 32
    V = [[0]*rank for _ in range(m*p)]  # 20 x 32
    W = [[0]*rank for _ in range(p*n)]  # 10 x 32
    
    col = 0
    
    def add_strassen_2x2x2(A_row_offset, B_row_offset, k0, k1, col):
        """Add 7 Strassen terms for 2x2 x 2x2 block.
        A entries: A[A_row_offset + i*m + j] for i,j in {0,1}
        B entries: B[B_row_offset + j*p + k] for j in {0,1}, k in {k0,k1}
        C entries: C[k*n + i] for i in {0,1}, k in {k0,k1}
        """
        global col
        # Strassen intermediates:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # A indices: A[i*m + (A_row_offset + j)] for i in {0,1}, j in {0,1}
        # B indices: B[(B_row_offset + j)*p + (k0 + dk)] for j in {0,1}, dk in {0,1}
        # C indices: C[(k0+dk)*n + i] for i in {0,1}, dk in {0,1}
        
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[A_row_offset + 0][col] += 1  # A[0,0]
        U[A_row_offset + 1 + m][col] += 1  # A[1,1]
        V[B_row_offset + 0*p + k0][col] += 1  # B[0,k0]
        V[B_row_offset + 1*p + k1][col] += 1  # B[1,k1]
        # C contributions: C[k0,0] += M1, C[k1,1] += M1
        W[k0*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[A_row_offset + 1][col] += 1  # A[1,0]
        U[A_row_offset + 1 + m][col] += 1  # A[1,1]
        V[B_row_offset + 0*p + k0][col] += 1  # B[0,k0]
        # C: C[k0,1] += M2
        W[k0*n + 1][col] += 1
        col += 1
        
        # M3: A[0,0] * (B[1,0]-B[0,0])
        U[A_row_offset + 0][col] += 1  # A[0,0]
        V[B_row_offset + 1*p + k0][col] += 1  # B[1,k0]
        V[B_row_offset + 0*p + k0][col] -= 1  # -B[0,k0]
        # C: C[k0,0] += M3
        W[k0*n + 0][col] += 1
        col += 1
        
        # M4: A[1,1] * (B[0,1]-B[1,1])
        U[A_row_offset + 1 + m][col] += 1  # A[1,1]
        V[B_row_offset + 0*p + k1][col] += 1  # B[0,k1]
        V[B_row_offset + 1*p + k1][col] -= 1  # -B[1,k1]
        # C: C[k1,0] += M4
        W[k1*n + 0][col] += 1
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U[A_row_offset + 0][col] += 1  # A[0,0]
        U[A_row_offset + 0 + m][col] += 1  # A[0,1]
        V[B_row_offset + 1*p + k1][col] += 1  # B[1,k1]
        # C: C[k1,1] += M5
        W[k1*n + 1][col] += 1
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
        U[A_row_offset + 1][col] += 1  # A[1,0]
        U[A_row_offset + 0][col] -= 1  # -A[0,0]
        V[B_row_offset + 0*p + k0][col] += 1  # B[0,k0]
        V[B_row_offset + 1*p + k0][col] += 1  # B[1,k0]
        # C: C[k0,1] += M6
        W[k0*n + 1][col] += 1
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
        U[A_row_offset + 0 + m][col] += 1  # A[0,1]
        U[A_row_offset + 1 + m][col] -= 1  # -A[1,1]
        V[B_row_offset + 0*p + k1][col] += 1  # B[0,k1]
        V[B_row_offset + 1*p + k1][col] += 1  # B[1,k1]
        # C: C[k1,1] += M7
        W[k1*n + 1][col] += 1
        col += 1
        
        return col
    
    def add_direct_2x2x1(A_row_offset, B_row_offset, k, col):
        """Add 2 direct terms for 2x2 x 2x1 block."""
        global col
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # Term 1: A[0,0]*B[0,k] -> C[0,k], A[1,0]*B[0,k] -> C[1,k]
        U[A_row_offset + 0][col] += 1  # A[0,0]
        V[B_row_offset + 0*p + k][col] += 1  # B[0,k]
        W[k*n + 0][col] += 1  # C[0,k]
        col += 1
        
        # Term 2: A[0,1]*B[1,k] -> C[0,k], A[1,1]*B[1,k] -> C[1,k]
        U[A_row_offset + 0 + m][col] += 1  # A[0,1]
        V[B_row_offset + 1*p + k][col] += 1  # B[1,k]
        W[k*n + 0][col] += 1  # C[0,k]
        col += 1
        
        # Wait, I need to handle C[1,k] too
        # Let me redo: each term is A[i,j]*B[j,k] contributing to C[i,k]
        # Term for j=0: A[0,0]*B[0,k] -> C[0,k]; A[1,0]*B[0,k] -> C[1,k]
        # These are two separate rank-1 terms in the full tensor, but they share the same B factor.
        # Actually no - each rank-1 term in the tensor decomposition is u ⊗ v ⊗ w where
        # u is a vector in A-space, v in B-space, w in C-space.
        # The term A[0,0]*B[0,k] contributes to C[0,k], so it's e_{0,0} ⊗ e_{0,k} ⊗ e_{0,k}
        # The term A[1,0]*B[0,k] contributes to C[1,k], so it's e_{1,0} ⊗ e_{0,k} ⊗ e_{1,k}
        # These are different rank-1 terms.
        
        # I made an error. Let me restart this function.
        return col
    
    # Let me redo this more carefully.
    # Reset
    rank = 32
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    
    def strassen_block(A_col_start, B_row_start, k0, k1):
        """Add 7 Strassen terms.
        A: 2x2 block at columns A_col_start, A_col_start+1, rows 0,1
        B: 2x5 block at rows B_row_start, B_row_start+1, cols k0,k1
        C: 2x2 block at rows 0,1, cols k0,k1
        A[i,j] is at index i*m + (A_col_start + j)
        B[j,k] is at index (B_row_start + j)*p + k
        C[i,k] is at index k*n + i
        """
        nonlocal col
        a00 = 0*m + (A_col_start + 0)
        a01 = 0*m + (A_col_start + 1)
        a10 = 1*m + (A_col_start + 0)
        a11 = 1*m + (A_col_start + 1)
        
        b0_k0 = (B_row_start + 0)*p + k0
        b0_k1 = (B_row_start + 0)*p + k1
        b1_k0 = (B_row_start + 1)*p + k0
        b1_k1 = (B_row_start + 1)*p + k1
        
        c0_k0 = k0*n + 0
        c0_k1 = k0*n + 1  # Wait, C[i,k] is at index k*n + i, so C[0,k0] = k0*n+0, C[1,k0] = k0*n+1
        c1_k0 = k0*n + 1
        c1_k1 = k1*n + 1
        
        # M1 = (A00+A11)(B0_k0 + B1_k1)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b0_k0][col] += 1
        V[b1_k1][col] += 1
        W[c0_k0][col] += 1  # C[0,k0]
        W[c1_k1][col] += 1  # C[1,k1]
        col += 1
        
        # M2 = (A10+A11)*B0_k0
        U[a10][col] += 1
        U[a11][col] += 1
        V[b0_k0][col] += 1
        W[c1_k0][col] += 1  # C[1,k0]
        col += 1
        
        # M3 = A00*(B1_k0 - B0_k0)
        U[a00][col] += 1
        V[b1_k0][col] += 1
        V[b0_k0][col] -= 1
        W[c0_k0][col] += 1  # C[0,k0]
        col += 1
        
        # M4 = A11*(B0_k1 - B1_k1)
        U[a11][col] += 1
        V[b0_k1][col] += 1
        V[b1_k1][col] -= 1
        W[c0_k1][col] += 1  # C[0,k1]
        col += 1
        
        # M5 = (A00+A01)*B1_k1
        U[a00][col] += 1
        U[a01][col] += 1
        V[b1_k1][col] += 1
        W[c1_k1][col] += 1  # C[1,k1]
        col += 1
        
        # M6 = (A10-A00)*(B0_k0 + B1_k0)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b0_k0][col] += 1
        V[b1_k0][col] += 1
        W[c1_k0][col] += 1  # C[1,k0]
        col += 1
        
        # M7 = (A01-A11)*(B0_k1 + B1_k1)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b0_k1][col] += 1
        V[b1_k1][col] += 1
        W[c1_k1][col] += 1  # C[1,k1]
        col += 1
        
        return col
    
    def direct_block(A_col_start, B_row_start, k):
        """Add 2 direct terms for 2x2 x 2x1.
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        """
        nonlocal col
        a00 = 0*m + (A_col_start + 0)
        a01 = 0*m + (A_col_start + 1)
        a10 = 1*m + (A_col_start + 0)
        a11 = 1*m + (A_col_start + 1)
        
        b0_k = (B_row_start + 0)*p + k
        b1_k = (B_row_start + 1)*p + k
        
        c0_k = k*n + 0
        c1_k = k*n + 1
        
        # Term 1: A00*B0_k -> C0_k, A10*B0_k -> C1_k
        # These need to be separate rank-1 terms since they contribute to different C entries
        # Actually: the bilinear form term is A[i,j]*B[j,k]*C[i,k]
        # For j=0: A[0,0]*B[0,k] contributes to C[0,k], A[1,0]*B[0,k] contributes to C[1,k]
        # As rank-1 tensor terms: e_{A[0,0]} ⊗ e_{B[0,k]} ⊗ e_{C[0,k]} and e_{A[1,0]} ⊗ e_{B[0,k]} ⊗ e_{C[1,k]}
        
        U[a00][col] += 1
        V[b0_k][col] += 1
        W[c0_k][col] += 1
        col += 1
        
        U[a10][col] += 1
        V[b0_k][col] += 1
        W[c1_k][col] += 1
        col += 1
        
        # Wait, that's already 2 terms for j=0, and I need 2 more for j=1. That's 4 terms total, not 2.
        # I think I'm confusing myself. Let me reconsider.
        
        # The schoolbook for 2x2 x 2x1 has 2*2*1 = 4 terms, not 2.
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # That's 4 bilinear terms, each a rank-1 tensor.
        
        # So direct_block should add 4 terms, not 2.
        # Let me fix this.
        return col
    
    # I realize my counting was wrong. Let me recount.
    # 2x2 x 2x1: schoolbook rank = 2*2*1 = 4.
    # 2x2 x 2x2: Strassen rank = 7 (vs schoolbook 8).
    # 2x2 x 2x5: two 2x2x2 blocks (rank 7 each) + one 2x2x1 block (rank 4) = 7+7+4 = 18.
    # Total for 2x4 x 4x5: 18 + 18 = 36.
    
    # Hmm, that gives 36, not 32. Let me reconsider.
    
    # Actually, I think the correct decomposition is:
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x2 x 2x1: rank 4 (schoolbook, can't improve since it's just 4 independent products)
    # 2x2 x 2x5: 7 + 7 + 4 = 18
    # 2x4 x 4x5: 18 + 18 = 36
    
    # Let me verify this is correct by implementing it properly.
    
    # Reset everything
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 8 x 36
    V = [[0]*rank for _ in range(m*p)]  # 20 x 36
    W = [[0]*rank for _ in range(p*n)]  # 10 x 36
    col = 0
    
    def strassen_block(A_col_start, B_row_start, k0, k1):
        nonlocal col
        a00 = 0*m + (A_col_start + 0)
        a01 = 0*m + (A_col_start + 1)
        a10 = 1*m + (A_col_start + 0)
        a11 = 1*m + (A_col_start + 1)
        
        b0_k0 = (B_row_start + 0)*p + k0
        b0_k1 = (B_row_start + 0)*p + k1
        b1_k0 = (B_row_start + 1)*p + k0
        b1_k1 = (B_row_start + 1)*p + k1
        
        c0_k0 = k0*n + 0
        c1_k0 = k0*n + 1
        c0_k1 = k1*n + 0
        c1_k1 = k1*n + 1
        
        # M1 = (A00+A11)(B0_k0 + B1_k1) -> C0_k0, C1_k1
        U[a00][col] += 1; U[a11][col] += 1
        V[b0_k0][col] += 1; V[b1_k1][col] += 1
        W[c0_k0][col] += 1; W[c1_k1][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B0_k0 -> C1_k0
        U[a10][col] += 1; U[a11][col] += 1
        V[b0_k0][col] += 1
        W[c1_k0][col] += 1
        col += 1
        
        # M3 = A00*(B1_k0 - B0_k0) -> C0_k0
        U[a00][col] += 1
        V[b1_k0][col] += 1; V[b0_k0][col] -= 1
        W[c0_k0][col] += 1
        col += 1
        
        # M4 = A11*(B0_k1 - B1_k1) -> C0_k1
        U[a11][col] += 1
        V[b0_k1][col] += 1; V[b1_k1][col] -= 1
        W[c0_k1][col] += 1
        col += 1
        
        # M5 = (A00+A01)*B1_k1 -> C1_k1
        U[a00][col] += 1; U[a01][col] += 1
        V[b1_k1][col] += 1
        W[c1_k1][col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B0_k0 + B1_k0) -> C1_k0
        U[a10][col] += 1; U[a00][col] -= 1
        V[b0_k0][col] += 1; V[b1_k0][col] += 1
        W[c1_k0][col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B0_k1 + B1_k1) -> C1_k1
        U[a01][col] += 1; U[a11][col] -= 1
        V[b0_k1][col] += 1; V[b1_k1][col] += 1
        W[c1_k1][col] += 1
        col += 1
        
        return col
    
    def direct_block(A_col_start, B_row_start, k):
        nonlocal col
        a00 = 0*m + (A_col_start + 0)
        a01 = 0*m + (A_col_start + 1)
        a10 = 1*m + (A_col_start + 0)
        a11 = 1*m + (A_col_start + 1)
        
        b0_k = (B_row_start + 0)*p + k
        b1_k = (B_row_start + 1)*p + k
        
        c0_k = k*n + 0
        c1_k = k*n + 1
        
        # 4 terms: A[i,j]*B[j,k] -> C[i,k]
        U[a00][col] += 1; V[b0_k][col] += 1; W[c0_k][col] += 1; col += 1
        U[a01][col] += 1; V[b1_k][col] += 1; W[c0_k][col] += 1; col += 1
        U[a10][col] += 1; V[b0_k][col] += 1; W[c1_k][col] += 1; col += 1
        U[a11][col] += 1; V[b1_k][col] += 1; W[c1_k][col] += 1; col += 1
        
        return col
    
    # Sub-problem 1: A cols 0-1, B rows 0-1
    col = strassen_block(0, 0, 0, 1)  # k=0,1
    col = strassen_block(0, 0, 2, 3)  # k=2,3
    col = direct_block(0, 0, 4)       # k=4
    
    # Sub-problem 2: A cols 2-3, B rows 2-3
    col = strassen_block(2, 2, 0, 1)  # k=0,1
    col = strassen_block(2, 2, 2, 3)  # k=2,3
    col = direct_block(2, 2, 4)       # k=4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
