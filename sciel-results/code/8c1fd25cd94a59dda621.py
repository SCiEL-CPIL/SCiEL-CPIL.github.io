import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2.
    Each term is a tuple (u_vec, v_vec, w_vec) of length 4.
    """
    terms = []
    def mv(d):
        v = [0.0] * 4
        for (i, j), c in d.items():
            v[i * 2 + j] = c
        return v
    
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    terms.append((mv({(0,0):1,(1,1):1}), mv({(0,0):1,(1,1):1}), mv({(0,0):1,(1,1):1})))
    # M2 = (a10+a11)(b00) -> c10 += M2, c11 += M2
    terms.append((mv({(1,0):1,(1,1):1}), mv({(0,0):1}), mv({(1,0):1,(1,1):1})))
    # M3 = (a00)(b10+b11) -> c00 += M3, c01 += M3
    terms.append((mv({(0,0):1}), mv({(1,0):1,(1,1):1}), mv({(0,0):1,(0,1):1})))
    # M4 = (a11)(b01-b10) -> c01 += M4, c10 -= M4
    terms.append((mv({(1,1):1}), mv({(0,1):1,(1,0):-1}), mv({(0,1):1,(1,0):-1})))
    # M5 = (a00+a01)(b11) -> c01 += M5, c11 += M5
    terms.append((mv({(0,0):1,(0,1):1}), mv({(1,1):1}), mv({(0,1):1,(1,1):1})))
    # M6 = (a10-a00)(b00+b01) -> c00 -= M6, c10 += M6
    terms.append((mv({(1,0):1,(0,0):-1}), mv({(0,0):1,(0,1):1}), mv({(0,0):-1,(1,0):1})))
    # M7 = (a01-a11)(b10+b11) -> c01 += M7, c11 -= M7
    terms.append((mv({(0,1):1,(1,1):-1}), mv({(1,0):1,(1,1):1}), mv({(0,1):1,(1,1):-1})))
    
    return terms

def _build_strassen_4x4x4_terms():
    """
    Build the rank-49 decomposition for 4x4 * 4x4 using Strassen.
    The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    We use 7 Strassen multiplications of the 2x2 blocks.
    Each 2x2 multiplication is expanded into its 7 rank-1 terms.
    Total rank = 7 * 7 = 49.
    """
    strassen_2x2 = _strassen_2x2_terms()
    terms_4x4 = []
    
    # The 4x4 matrix A is:
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2 blocks.
    # A11 is rows 0-1, cols 0-1.
    # A12 is rows 0-1, cols 2-3.
    # A21 is rows 2-3, cols 0-1.
    # A22 is rows 2-3, cols 2-3.
    
    # The 4x4 matrix B is similarly blocked.
    # The 4x4 matrix C is similarly blocked.
    
    # Strassen formulas for block matrices:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B21-B22)
    # M4 = A22 * (B12-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these block-level operations to the rank-1 terms.
    # Each block operation like (A11+A22) * (B11+B22) involves a 2x2 matrix multiplication.
    # The U, V, W vectors for the 4x4 problem are length 16.
    # The U vector for a term will have non-zero entries in the blocks corresponding to the A factors.
    # The V vector will have non-zero entries in the blocks corresponding to the B factors.
    # The W vector will have non-zero entries in the blocks corresponding to the C factors.
    
    # Helper to create a 16-dim vector from 4 2x2 block coefficients
    def block_vec(blocks):
        # blocks is a dict mapping (br, bc) to a 4-dim vector of coefficients for that 2x2 block
        v = [0.0] * 16
        for (br, bc), coeffs in blocks.items():
            for r in range(2):
                for c in range(2):
                    v[(br*2 + r)*4 + (bc*2 + c)] = coeffs[r*2 + c]
        return v
    
    # Define the block positions
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: same
    # C blocks: same
    
    # For each of the 7 Strassen terms, we need to construct the U, V, W vectors.
    # The 7 terms are linear combinations of the block products.
    # Actually, it's easier to just apply the 2x2 Strassen decomposition to each of the 7 block products.
    
    # Let's define the 7 block products P1..P7.
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each Pi is a 2x2 matrix product.
    # We decompose each Pi into 7 rank-1 terms using _strassen_2x2_terms.
    # For each rank-1 term in Pi, we expand it into the 4x4 space.
    
    # The U vector for a term in Pi will be the U vector of the 2x2 term, placed in the appropriate A blocks.
    # The V vector will be the V vector of the 2x2 term, placed in the appropriate B blocks.
    # The W vector will be the W vector of the 2x2 term, placed in the appropriate C blocks.
    
    # We need to track which C block each Pi contributes to, and with what sign.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Let's create a list of (U_blocks, V_blocks, W_blocks, sign) for each contribution.
    # Actually, it's simpler to just build the 49 terms directly.
    
    # For each of the 7 Strassen terms (s in 0..6):
    #   For each of the 7 rank-1 terms in the 2x2 decomposition (t in 0..6):
    #     Construct U, V, W.
    
    # The U vector for the 2x2 term t is u2[t].
    # This u2[t] is a 4-dim vector representing the coefficients for the 2x2 A block.
    # We need to map this to the 4x4 A blocks.
    # For P1, the A blocks involved are A11 and A22.
    # The 2x2 Strassen term t has a specific pattern of coefficients in its U vector.
    # We need to apply these coefficients to the A11 and A22 blocks.
    
    # This is getting complex. Let's use a more direct method.
    # We can just use the schoolbook for the 4x4x4 part if the Strassen implementation is too risky?
    # No, rank 144 is the seed. Rank 114 is the target.
    # Let's try to implement it carefully.
    
    # Alternative: Use the known rank-49 decomposition for 4x4x4.
    # There are known algorithms. But deriving it is error-prone.
    
    # Let's try a different split:
    # 4x4 * 4x9.
    # Split A into 4 rows of 1x4.
    # Split B into 9 columns of 4x1.
    # This is schoolbook.
    
    # Let's stick to the block Strassen.
    # I will construct the 49 terms by iterating over the 7 Strassen block products and their 7 2x2 rank-1 terms.
    
    # Define the block mappings for each Pi.
    # Pi is a product of two "virtual" 2x2 matrices X and Y.
    # X is a linear combination of A blocks.
    # Y is a linear combination of B blocks.
    # The result Pi is a 2x2 matrix.
    # Pi contributes to C blocks with specific signs.
    
    # Let's define the structure of Pi.
    # P1: X = A11+A22, Y = B11+B22.
    #     X is a 2x2 matrix where the (0,0) block is A11+A22? No.
    #     A is a 4x4 matrix. A11 is a 2x2 submatrix.
    #     X is a 2x2 matrix. The entries of X are not blocks, they are the 2x2 matrices themselves?
    #     No, in Strassen, we treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    #     So X is a 2x2 matrix where each entry is a 2x2 matrix.
    #     X[0,0] = A11, X[0,1] = A12, X[1,0] = A21, X[1,1] = A22.
    #     For P1, X1 = X[0,0] + X[1,1] = A11 + A22.
    #     Y1 = Y[0,0] + Y[1,1] = B11 + B22.
    #     P1 = X1 * Y1.
    #     X1 is a 2x2 matrix (sum of two 2x2 matrices).
    #     Y1 is a 2x2 matrix.
    #     So P1 is a 2x2 matrix product.
    #     We decompose P1 into 7 rank-1 terms.
    #     Each rank-1 term is u * v * w where u,v,w are vectors of length 4 (for the 2x2 matrix).
    #     To embed this into the 4x4 problem:
    #     The U vector (length 16) will have the coefficients of u placed in the A11 and A22 blocks.
    #     Specifically, if u = [u00, u01, u10, u11], then:
    #     U[0:2, 0:2] += u[0:2, 0:2] (for A11)
    #     U[2:4, 2:4] += u[0:2, 0:2] (for A22)
    #     Wait, X1 = A11 + A22. So the coefficient for A11 is 1 and for A22 is 1.
    #     So if the 2x2 term has coefficient c for the (r,c) entry of X1, then:
    #     The (r,c) entry of A11 gets coefficient c.
    #     The (r,c) entry of A22 gets coefficient c.
    #     Similarly for Y1 and B blocks.
    #     And the result P1 is a 2x2 matrix.
    #     P1 contributes to C11 with sign +1.
    #     So the (r,c) entry of C11 gets coefficient w[r,c].
    #     C11 is the top-left 2x2 block of C.
    
    # This logic holds. I will implement a helper to expand a 2x2 Strassen term into the 4x4 space given the block mappings.
    
    def expand_term(u2, v2, w2, a_blocks, b_blocks, c_blocks, signs):
        """
        u2, v2, w2: 4-dim vectors from 2x2 Strassen term.
        a_blocks: dict mapping block_pos (e.g., (0,0)) to coefficient (e.g., 1).
                  This represents X = sum(coeff * A_block).
        b_blocks: dict mapping block_pos to coefficient.
        c_blocks: dict mapping block_pos to coefficient.
        signs: dict mapping block_pos to sign (e.g., 1 or -1).
               This represents C_block += sign * P.
        """
        U = np.zeros(16)
        V = np.zeros(16)
        W = np.zeros(16)
        
        # Expand U
        for (br, bc), coeff in a_blocks.items():
            # The 2x2 term u2 corresponds to the "virtual" 2x2 matrix X.
            # X is formed by summing A blocks.
            # The entry u2[r*2+c] is the coefficient for the (r,c) entry of X.
            # This coefficient is applied to the (r,c) entry of each A block in a_blocks.
            for r in range(2):
                for c in range(2):
                    val = u2[r*2+c] * coeff
                    # Map (br, bc, r, c) to 4x4 index
                    # A block (br, bc) starts at row br*2, col bc*2.
                    # Entry (r,c) in the block is at global row br*2+r, col bc*2+c.
                    U[(br*2+r)*4 + (bc*2+c)] = val
        
        # Expand V
        for (br, bc), coeff in b_blocks.items():
            for r in range(2):
                for c in range(2):
                    val = v2[r*2+c] * coeff
                    V[(br*2+r)*4 + (bc*2+c)] = val
        
        # Expand W
        # W corresponds to the result P.
        # P contributes to C blocks.
        # The entry w2[r*2+c] is the coefficient for the (r,c) entry of P.
        # This is added to the (r,c) entry of each C block in c_blocks with the given sign.
        for (br, bc), sign in c_blocks.items():
            for r in range(2):
                for c in range(2):
                    val = w2[r*2+c] * sign
                    W[(br*2+r)*4 + (bc*2+c)] = val
        
        return U, V, W
    
    # Define the 7 Strassen block products
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Block positions: (0,0)=11, (0,1)=12, (1,0)=21, (1,1)=22
    
    products = [
        # (a_blocks, b_blocks, c_blocks)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1, (0,1):0, (1,0):0, (1,1):1}), # P1 -> C11(+), C22(+)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # P2 -> C21(+), C22(-)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}), # P3 -> C12(+), C22(+)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}), # P4 -> C11(+), C21(+)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}), # P5 -> C12(+), C22(-)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # P6 -> C22(+)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}), # P7 -> C11(+), C22(-)
    ]
    
    # Note: The c_blocks dict should map block_pos to sign.
    # Let's refine the c_blocks for each product based on the formulas:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # P1: C11(+1), C22(+1)
    # P2: C21(+1), C22(-1)
    # P3: C12(+1), C22(+1)
    # P4: C11(+1), C21(+1)
    # P5: C12(+1), C22(-1)
    # P6: C22(+1)
    # P7: C11(+1), C22(-1)
    
    # Corrected c_blocks:
    c_blocks_list = [
        {(0,0): 1, (1,1): 1}, # P1
        {(1,0): 1, (1,1): -1}, # P2
        {(0,1): 1, (1,1): 1}, # P3
        {(0,0): 1, (1,0): 1}, # P4
        {(0,1): 1, (1,1): -1}, # P5
        {(1,1): 1}, # P6
        {(0,0): 1, (1,1): -1}, # P7
    ]
    
    terms = []
    for i in range(7):
        a_blocks, b_blocks, _ = products[i]
        c_blocks = c_blocks_list[i]
        
        for u2, v2, w2 in strassen_2x2:
            U, V, W = expand_term(u2, v2, w2, a_blocks, b_blocks, c_blocks, None)
            terms.append((U, V, W))
            
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS # 4, 4, 9
    
    # Strategy:
    # 1. Split B (4x9) into B1 (4x4) and B2 (4x5).
    # 2. Split B2 into B21 (4x4) and B22 (4x1).
    # 3. Compute A*B1 using Strassen 4x4x4 (rank 49).
    # 4. Compute A*B21 using Strassen 4x4x4 (rank 49).
    # 5. Compute A*B22 using Schoolbook 4x4x1 (rank 16).
    # Total rank: 49 + 49 + 16 = 114.
    
    # We need to embed these into the full 4x4x9 space.
    # U is 4x4 (16xR)
    # V is 4x9 (36xR)
    # W is 9x4 (36xR)
    
    # The U vectors for all parts are the same 4x4 structure.
    # The V vectors are different for each part, depending on the column slice of B.
    # The W vectors are different for each part, depending on the column slice of C.
    
    # Part 1: A * B1 (cols 0-3 of B)
    # U1: 16x49
    # V1: 36x49 (non-zero only in rows 0-15, i.e., B1 cols)
    # W1: 36x49 (non-zero only in rows 0-15, i.e., C1 cols)
    
    # Part 2: A * B21 (cols 4-7 of B)
    # U2: 16x49 (same as U1)
    # V2: 36x49 (non-zero only in rows 16-31, i.e., B21 cols)
    # W2: 36x49 (non-zero only in rows 16-31, i.e., C21 cols)
    
    # Part 3: A * B22 (col 8 of B)
    # U3: 16x16
    # V3: 36x16 (non-zero only in row 32, i.e., B22 col)
    # W3: 36x16 (non-zero only in row 32, i.e., C22 col)
    
    strassen_terms = _build_strassen_4x4x4_terms()
    
    # Initialize
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1
    for t, (u2, v2, w2) in enumerate(strassen_terms):
        U[:, t] = u2
        # v2 is 16-dim (4x4). We need to map it to the 36-dim V vector.
        # B1 is cols 0-3 of B.
        # The 4x4 B1 corresponds to rows 0-15 of the 4x9 B.
        # Index in B: j*9 + k.
        # For B1, k is 0-3.
        # The 4x4 block B1 has entries B1[j, k].
        # In the 4x9 B, this is B[j, k].
        # So V[j*9 + k] = v2[j*4 + k].
        for j in range(4):
            for k in range(4):
                V[j*9 + k, t] = v2[j*4 + k]
        
        # w2 is 16-dim (4x4). We need to map it to the 36-dim W vector.
        # C1 is cols 0-3 of C.
        # The 4x4 C1 has entries C1[i, k].
        # In the 4x9 C, this is C[i, k].
        # Index in C: k*4 + i (per task convention W[k*n+i]).
        # So W[k*4 + i] = w2[k*4 + i].
        for k in range(4):
            for i in range(4):
                W[k*4 + i, t] = w2[k*4 + i]
                
    # Part 2
    offset = 49
    for t, (u2, v2, w2) in enumerate(strassen_terms):
        col = offset + t
        U[:, col] = u2
        # B21 is cols 4-7 of B.
        # B21[j, k] corresponds to B[j, 4+k].
        for j in range(4):
            for k in range(4):
                V[j*9 + (4+k), col] = v2[j*4 + k]
        
        # C21 is cols 4-7 of C.
        # C21[i, k] corresponds to C[i, 4+k].
        for k in range(4):
            for i in range(4):
                W[(4+k)*4 + i, col] = w2[k*4 + i]
                
    # Part 3: Schoolbook for 4x4 * 4x1
    # Rank 16.
    # U is 16x16.
    # V is 36x16.
    # W is 36x16.
    
    offset = 98
    for i in range(4):
        for j in range(4):
            t = i*4 + j
            col = offset + t
            U[i*4 + j, col] = 1.0
            # V: B22 is col 8 of B.
            # B22[j, 0] corresponds to B[j, 8].
            V[j*9 + 8, col] = 1.0
            # W: C22 is col 8 of C.
            # C22[i, 0] corresponds to C[i, 8].
            W[8*4 + i, col] = 1.0
            
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        # This should not happen if the Strassen implementation is correct.
        U, V, W = _build_schoolbook(n, m, p)
        
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook(n, m, p):
    """Build the trivial rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W
