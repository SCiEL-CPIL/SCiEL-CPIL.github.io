"""Matrix multiplication 4,4,8 - Strassen-based rank-98 decomposition."""

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-98 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    
    Uses Strassen's 2x2x2 algorithm (rank 7) applied to each half
    of the 4x8 output, giving 2 * 7 = 14 terms per 4x4x4 subproblem,
    and two such subproblems = 28... 
    
    Wait, let me think more carefully.
    
    A is 4x4, B is 4x8, C = A*B is 4x8.
    
    We can split B into B1 (4x4, cols 0-3) and B2 (4x4, cols 4-7).
    C1 = A*B1 (4x4), C2 = A*B2 (4x4).
    
    For each 4x4x4 multiplication, we can use a recursive Strassen approach:
    Split 4x4 into 2x2 blocks. Use Strassen for 2x2x2 (rank 7).
    A 4x4x4 via one level of Strassen: 7 * 2^3 = 56? No.
    
    Actually, for 4x4x4, we split each matrix into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Strassen gives 7 multiplications of 2x2 matrices.
    Each 2x2x2 multiplication has rank 4 (schoolbook) or rank 7 (Strassen).
    But we want exact decomposition with rank counting each rank-1 term.
    
    If we use schoolbook for the 2x2x2: each 2x2x2 = 4 rank-1 terms.
    7 Strassen terms * 4 = 28 rank-1 terms per 4x4x4.
    
    Total: 2 * 28 = 56 rank-1 terms.
    
    Wait, but the knowledge says rank 49 for 4x4x4 using Strassen recursively.
    That means they used Strassen for the 2x2x2 as well: 7 * 7 = 49.
    
    So for 4x4x8: 2 * 49 = 98 rank-1 terms.
    
    Let me implement this properly.
    
    For 4x4x4 with recursive Strassen:
    Level 1: Split into 2x2 blocks, 7 Strassen products of 2x2x2.
    Level 2: Each 2x2x2 uses 7 Strassen products of 1x1x1.
    Each 1x1x1 is just a scalar multiply = 1 rank-1 term.
    Total: 7 * 7 = 49 rank-1 terms per 4x4x4.
    
    For 4x4x8: split B into two 4x4 halves.
    C1 = A * B1: 49 terms
    C2 = A * B2: 49 terms
    Total: 98 terms.
    """
    n, m, p = 4, 4, 8
    
    # Build U, V, W as lists of lists
    # U shape: (n*m, rank) = (16, 98)
    # V shape: (m*p, rank) = (32, 98)
    # W shape: (p*n, rank) = (32, 98)
    
    U = []
    V = []
    W = []
    
    # Strassen 2x2x2: 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In 0-indexed 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Strassen 1x1x1: just one rank-1 term
    # For 2x2x2 using Strassen, we have 7 terms, each being a product of 
    # linear forms. Each linear form has coefficients from {-1, 0, 1}.
    # Each term contributes to output with coefficients from {-1, 0, 1}.
    
    # Let's define the 7 Strassen terms for 2x2x2:
    # term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 4 values for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_coeffs: 4 values for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_coeffs: 4 values for C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_2x2 = [
        # M1 = (a00+a11)(b00+b11), contributes to c00 (+1), c11 (+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a10+a11)(b00), contributes to c10 (+1), c11 (-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = (a00)(b01-b11), contributes to c01 (+1), c11 (+1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = (a11)(b10-b00), contributes to c00 (+1), c10 (+1)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5 = (a00+a01)(b11), contributes to c00 (-1), c01 (+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a10-a00)(b00+b01), contributes to c11 (+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a01-a11)(b10+b11), contributes to c00 (+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Now build 4x4x4 decomposition using one level of Strassen on 2x2 blocks
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # For the outer Strassen, the "elements" are 2x2 matrices.
    # We need to apply Strassen at the 2x2 level, then each 2x2 product is 
    # done via inner Strassen (7 more terms).
    #
    # Outer Strassen terms (for 4x4x4 with 2x2 block elements):
    # Let's denote 2x2 blocks as elements.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2x2 product of 2x2 matrices.
    # Using Strassen for each 2x2x2: 7 terms.
    # Total: 7 * 7 = 49 terms.
    
    # For each outer term, we have linear combinations of 2x2 blocks.
    # We need to expand these into 1x1 elements.
    
    # Let me define the outer Strassen terms as linear combinations of 2x2 blocks.
    # For U (input A, 4x4): each 2x2 block has 4 elements.
    # A element indexing: A[i,j] where i in 0..3, j in 0..3.
    # Block A00: elements (0,0),(0,1),(1,0),(1,1)
    # Block A01: elements (0,2),(0,3),(1,2),(1,3)
    # Block A10: elements (2,0),(2,1),(3,0),(3,1)
    # Block A11: elements (2,2),(2,3),(3,2),(3,3)
    
    # For V (input B, 4x4 for each half): same block structure.
    # For W (output C, 4x4 for each half): same block structure.
    
    # Let me define block element indices:
    # Block (bi, bj) has elements at positions:
    # (bi*2 + di, bj*2 + dj) for di,dj in 0,1
    
    def block_elements(bi, bj):
        """Return list of (i,j) positions for block (bi,bj) in 4x4 matrix."""
        return [(bi*2 + di, bj*2 + dj) for di in range(2) for dj in range(2)]
    
    # Outer Strassen: 7 terms, each with coefficients for the 4 blocks of A and B
    # and the 4 blocks of C.
    outer_strassen = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # M1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A10+A11)(B00) -> C10:+1, C11:-1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # M3 = (A00)(B01-B11) -> C01:+1, C11:+1
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # M4 = (A11)(B10-B00) -> C00:+1, C10:+1
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        # M5 = (A00+A01)(B11) -> C00:-1, C01:+1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6 = (A10-A00)(B00+B01) -> C11:+1
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A01-A11)(B10+B11) -> C00:+1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Now build the full 4x4x4 decomposition (49 terms)
    # For each outer term (ao, bo, co), for each inner Strassen term (ai, bi, ci):
    # The combined U coefficient for A element (i,j) is:
    #   sum over blocks b where A_block coefficient is nonzero:
    #     ao[b] * ai[local_index_of_(i,j)_in_block_b]
    # Similarly for V and W.
    
    def build_4x4x4_terms():
        """Build 49 rank-1 terms for 4x4x4 matrix multiplication."""
        terms = []
        for outer_idx, (ao, bo, co) in enumerate(outer_strassen):
            for inner_idx, (ai, bi, ci) in enumerate(strassen_2x2):
                # U: 16 values (4x4 A matrix)
                u = [0] * 16
                # V: 16 values (4x4 B matrix)
                v = [0] * 16
                # W: 16 values (4x4 C matrix)
                w = [0] * 16
                
                # For each block of A:
                for block_idx in range(4):
                    if ao[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        u[ei * 4 + ej] = ao[block_idx] * ai[local_idx]
                
                # For each block of B:
                for block_idx in range(4):
                    if bo[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        v[ei * 4 + ej] = bo[block_idx] * bi[local_idx]
                
                # For each block of C:
                for block_idx in range(4):
                    if co[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        w[ei * 4 + ej] = co[block_idx] * ci[local_idx]
                
                terms.append((u, v, w))
        
        return terms
    
    # Build the full 4x4x8 decomposition
    # A is 4x4 (16 elements), B is 4x8 (32 elements), C is 4x8 (32 elements)
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # Split C into C1 (cols 0-3) and C2 (cols 4-7)
    
    terms_4x4x4 = build_4x4x4_terms()
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    col = 0
    for half in range(2):
        for (u, v, w) in terms_4x4x4:
            # u: 16 values for A (4x4)
            # v: 16 values for B_half (4x4) -> map to B (4x8)
            # w: 16 values for C_half (4x4) -> map to C (4x8)
            
            # U: A is the same for both halves
            for i in range(16):
                U[i][col] = u[i]
            
            # V: B_half elements mapped to B (4x8)
            # B_half has elements (i,j) for i in 0..3, j in 0..3
            # In B (4x8), element (i, j) is at position i*8 + (j + half*4)
            for i in range(4):
                for j in range(4):
                    b_idx = i * 8 + (j + half * 4)
                    v_local_idx = i * 4 + j
                    V[b_idx][col] = v[v_local_idx]
            
            # W: C_half elements mapped to C (4x8)
            # C_half has elements (i,j) for i in 0..3, j in 0..3
            # In C (4x8), element (i, j) is at position i*8 + (j + half*4)
            # But W follows trace convention: W[k*n+i, r] where k is col of C, i is row of C
            # Wait, the task says: W[k*n+i, r]
            # n=4, so W[k*4+i, r]
            # k is the column index of C (0..7), i is the row index of C (0..3)
            # So W has p*n = 8*4 = 32 rows
            # Row index: k*4 + i where k is col, i is row
            #
            # In C (4x8), element (row=i, col=k) is at position... 
            # The tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where A is n x m (4x4), B is m x p (4x8), C is p x n (8x4)?
            # Wait, let me re-read: "multiplying a 4 by 4 matrix by a 4 by 8 matrix"
            # A: 4x4, B: 4x8, C = A*B: 4x8
            # 
            # U shape: (n*m, rank) = (4*4, rank) = (16, rank)
            # V shape: (m*p, rank) = (4*8, rank) = (32, rank)
            # W shape: (p*n, rank) = (8*4, rank) = (32, rank)
            #
            # The decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_r U[i*m+j, r] ... no.
            #
            # Standard: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
            # where T is the tensor for matrix mult: T_{ijk} = A_{ij} B_{jk} = C_{ik}
            # But here C is 4x8, so i in 0..3, k in 0..7, j in 0..3
            #
            # U indexed by (i,j): i in 0..3, j in 0..3 -> 16 entries
            # V indexed by (j,k): j in 0..3, k in 0..7 -> 32 entries
            # W indexed by (k,i): k in 0..7, i in 0..3 -> 32 entries
            #
            # W[k*n+i] = W[k*4+i]
            # So W row index = k*4 + i where k is col of C (0..7), i is row of C (0..3)
            
            for i in range(4):  # row of C
                for j in range(4):  # col of C_half
                    k_global = j + half * 4  # global col of C
                    w_row_idx = k_global * 4 + i
                    w_local_idx = i * 4 + j
                    W[w_row_idx][col] = w[w_local_idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
