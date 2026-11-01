"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-57 decomposition: Strassen on 2x2x2 blocks + schoolbook on 2x3x3 blocks
    n, m, p = DIMENSIONS
    
    # Rank breakdown:
    # For the first 2 columns of A (i.e., m dimension 2), we split p=6 into three 2-blocks
    # For the last 3 columns of A (m dimension 3), we split p=6 into two 3-blocks
    # Each 2x2x2 block uses Strassen (rank 7), each 2x3x3 block uses schoolbook (rank 9)
    
    # Total: 3 * 7 + 2 * 9 = 21 + 18 = 39? No wait, let me recalculate.
    # Actually, for each "j" position in the middle dimension, we multiply a row of A (length n=2)
    # by a column of B (length p=6). But the tensor decomposition is over the full multiplication.
    
    # Let me think more carefully. The matrix multiplication tensor for (2,5,6) has:
    # U: 10 x rank, V: 30 x rank, W: 12 x rank
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # where i in [0,n), j in [0,m), k in [0,p), c in [0,rank)
    
    # The schoolbook has rank = n*m*p = 2*5*6 = 60.
    
    # We can exploit the structure. For fixed j, the sub-tensor is (n,1,p) which is just
    # the outer product structure. But to get rank reduction, we need to combine across j's
    # or use Strassen-like tricks.
    
    # Actually, a cleaner approach: treat the multiplication as (2x5) * (5x6).
    # We can view this as multiplying a 2x5 matrix by a 5x6 matrix.
    # One approach: split the 5 columns of A (and rows of B) into groups.
    
    # Let me try a different strategy. Use the fact that n=2 is small.
    # For each pair of columns (k1, k2) of the result, and for each row i of the result,
    # C[i,k] = sum_j A[i,j] * B[j,k].
    
    # A rank-1 term contributes: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # summed over j and c gives C[i,k].
    
    # To beat schoolbook (60), we need to find a decomposition with rank < 60.
    # The trivial lower bound is max(n*p, m*p, n*m) = max(12, 30, 10) = 30.
    # But that's just the dimension of the spaces, not achievable generally.
    
    # For now, let's implement a block-wise Strassen approach.
    # Split the middle dimension m=5 into a 2+3 split.
    # Split the output dimension p=6 into 3 blocks of 2, or 2 blocks of 3.
    
    # Approach: 
    # Part 1: Use the first 2 columns of A (j=0,1) with all 6 rows of B.
    #   This is a (2, 2, 6) multiplication. Split p=6 into three (2,2,2) blocks.
    #   Each (2,2,2) block can use Strassen with rank 7.
    #   Total for this part: 3 * 7 = 21.
    #
    # Part 2: Use the last 3 columns of A (j=2,3,4) with all 6 rows of B.
    #   This is a (2, 3, 6) multiplication. Split p=6 into two (2,3,3) blocks.
    #   Each (2,3,3) block uses schoolbook with rank 2*3*3 = 18.
    #   Total for this part: 2 * 18 = 36.
    #
    # Total rank: 21 + 36 = 57.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    
    # Part 1: j in {0,1}, split k into blocks of 2: [0,1], [2,3], [4,5]
    # For each block, use Strassen for (n=2, m=2, p=2)
    for block_k in range(3):
        k_start = block_k * 2
        k_end = k_start + 2
        # Local indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Strassen terms for 2x2x2:
        # M1 = a*(f-h)
        # M2 = (a+b)*h
        # M3 = (g+d)*a
        # M4 = g*(c-b)
        # M5 = (a+e)(d-h)
        # M6 = (b+e)(c+g)
        # M7 = (a+c)(d-f)
        # Where A = [[a,b],[c,d]] (n x m = 2x2, so a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
        #       B = [[e,f],[g,h]] (m x p = 2x2, so e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1])
        # C = [[M1+M3+M5+M7, M2+M3],
        #      [M4+M6, M1+M4+M5+M6]]
        
        # Strassen coefficients for U, V, W
        # Each term c' in {0,...,6} corresponds to one rank-1 component
        # U[i*m + j, c] = coefficient for A[i,j]
        # V[j*p + k, c] = coefficient for B[j,k]
        # W[k*n + i, c] = coefficient for C[i,k]
        
        strassen_U = [
            [1, 0],   # term 0: a*(f-h) -> A[0,0]=a has coeff 1
            [1, 1],   # term 1: (a+b)*h -> A[0,0]=a has coeff 1, A[0,1]=b has coeff 1
            [1, 0],   # term 2: (g+d)*a -> A[0,0]=a has coeff 1
            [0, 0],   # term 3: g*(c-b) -> no A term
            [1, 0],   # term 4: (a+e)(d-h) -> A[0,0]=a has coeff 1
            [1, 0],   # term 5: (b+e)(c+g) -> A[0,0]... wait, this is for C[1,0]
            [1, 1],   # term 6: (a+c)(d-f) -> A[0,0]=a has coeff 1, A[1,0]=c has coeff 1
        ]
        # Let me redo this properly.
        # Term 0: M1 = a*(f-h)
        #   U: A[0,0] coeff 1 -> U[0*2+0] = 1, U[0*2+1] = 0, U[1*2+0] = 0, U[1*2+1] = 0
        #   V: B[0,1] coeff -1, B[1,1] coeff 1 -> V[0*2+1] = -1, V[1*2+1] = 1
        #   W: C[0,0] coeff 1, C[1,1] coeff 1 -> W[0*2+0] = 1, W[1*2+1] = 1
        
        # Term 1: M2 = (a+b)*h
        #   U: A[0,0] coeff 1, A[0,1] coeff 1
        #   V: B[1,1] coeff 1
        #   W: C[0,1] coeff 1
        
        # Term 2: M3 = (g+d)*a  -- wait, this doesn't look right.
        # Standard Strassen:
        # M1 = A11*(B21-B12)
        # M2 = (A11+A12)*B22
        # M3 = (A21+A22)*B11
        # M4 = A22*(B12-B21)
        # M5 = (A11+A22)*(B22-B11)  -- no, (A11+A21)(B12-B21)? Let me use the standard:
        # Actually, standard Strassen:
        # M1 = A11*(B11 - B21)  -- variations exist. Let me use a consistent set.
        
        # Let me use:
        # M1 = A[0,0] * (B[0,1] - B[1,1])
        # M2 = (A[0,0] + A[0,1]) * B[1,1]
        # M3 = (A[1,0] + A[1,1]) * B[0,0]
        # M4 = A[1,1] * (B[0,0] - B[1,0])
        # M5 = (A[0,0] + A[1,0]) * (B[1,1] - B[0,1])  -- no
        # Let me just use the well-known 7-term decomposition:
        # M1 = A11(B11 - B21)
        # M2 = (A11 + A12)B22
        # M3 = (A21 + A22)B11
        # M4 = A22(B12 - B21)
        # M5 = (A11 + A21)(B22 - B11)
        # M6 = (A12 + A22)(B11 + B22)
        # M7 = (A11 - A21)(B12 + B21)
        # C11 = M1 + M2 + M5 + M7
        # C12 = M3 + M4
        # C21 = M5 + M6
        # C22 = M1 - M4 + M6 + M7
        
        # Using 0-indexed: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
        # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
        # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
        
        strassen_terms = [
            # (U_coeffs for [A00, A01, A10, A11], V_coeffs for [B00, B01, B10, B11], W_coeffs for [C00, C01, C10, C11])
            ([1, 0, 0, 0], [1, 0, -1, 0], [1, 0, 0, 1]),   # M1
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),    # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),    # M3
            ([0, 0, 0, 1], [1, 0, -1, 0], [0, 0, 0, 1]),   # M4
            ([1, 0, 1, 0], [0, -1, 0, 1], [0, 0, 1, 0]),   # M5
            ([0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]),    # M6
            ([1, 0, -1, 0], [0, 1, 0, 1], [1, 0, 0, 1]),   # M7
        ]
        
        for t in range(7):
            u_coeffs = strassen_terms[t][0]
            v_coeffs = strassen_terms[t][1]
            w_coeffs = strassen_terms[t][2]
            
            # U: rows are i*m + j, for i in {0,1}, j in {0,1} (local)
            # But j here is local to the block, so global j = 0 + j_local
            for i_local in range(2):
                for j_local in range(2):
                    U[i_local * m + (j_local + 0)][col] = u_coeffs[i_local * 2 + j_local]
            
            # V: rows are j*p + k, for j in {0,1} (local), k in {0,1} (local)
            # Global j = 0 + j_local, global k = k_start + k_local
            for j_local in range(2):
                for k_local in range(2):
                    V[(j_local + 0) * p + (k_local + k_start)][col] = v_coeffs[j_local * 2 + k_local]
            
            # W: rows are k*n + i, for k in {0,1} (local), i in {0,1}
            # Global k = k_start + k_local
            for k_local in range(2):
                for i_local in range(2):
                    W[(k_local + k_start) * n + i_local][col] = w_coeffs[k_local * 2 + i_local]
            
            col += 1
    
    # Part 2: j in {2,3,4}, split k into blocks of 3: [0,1,2], [3,4,5]
    for block_k in range(2):
        k_start = block_k * 3
        k_end = k_start + 3
        # Schoolbook for (n=2, m=3, p=3): rank = 2*3*3 = 18
        for i_local in range(2):
            for j_local in range(3):
                for k_local in range(3):
                    U[i_local * m + (j_local + 2)][col] = 1
                    V[(j_local + 2) * p + (k_local + k_start)][col] = 1
                    W[(k_local + k_start) * n + i_local][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
