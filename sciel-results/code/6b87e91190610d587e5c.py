"""Matrix multiplication 2x2 x 2x11 via Strassen on 2x2 blocks."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # We'll group the 11 columns of B into 5 blocks of 2 columns + 1 single column.
    # For each block of 2 columns, we use a Strassen-like decomposition for 2x2 * 2x2.
    # For the last single column, we use schoolbook (4 terms).
    # Total rank = 5 * 7 + 4 = 39.
    
    # Strassen for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # The 7 products are:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)*e
    # P3 = a*(f-h)
    # P4 = d*(g-e)
    # P5 = (a+b)*h
    # P6 = (c-a)*(e+f)
    # P7 = (b-d)*(g+h)
    # And the results are:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For our problem: A is 2x2, B is 2x11.
    # We split B into columns: for block t (0..4), columns 2t and 2t+1 form a 2x2 block.
    # The last column (k=10) is handled separately.
    
    num_blocks = (p + 1) // 2  # 6 blocks (5 full + 1 single)
    # Actually: 11 columns -> 5 pairs (10 cols) + 1 single (col 10)
    num_full_blocks = p // 2  # 5
    has_single = (p % 2 != 0)  # True
    
    rank = num_full_blocks * 7 + (4 if has_single else 0)
    
    U = [[0] * rank for _ in range(n * m)]  # (4, rank)
    V = [[0] * rank for _ in range(m * p)]  # (22, rank)
    W = [[0] * rank for _ in range(p * n)]  # (22, rank)
    
    col = 0
    
    for t in range(num_full_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # For this block, we have 7 Strassen products.
        # The A part is the same 2x2 matrix A.
        # The B part is the 2x2 submatrix [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # P1 = (a+d)(e+h) where e=B[0,k0], h=B[1,k1]
        # U: a=1, d=1 -> U[0]=1, U[3]=1 (indices: i*m+j, so a=A[0,0]->idx0, b=A[0,1]->idx1, c=A[1,0]->idx2, d=A[1,1]->idx3)
        # V: e=B[0,k0]->idx k0, h=B[1,k1]->idx p+k1
        # W: C00=P1+... -> W[k0*n+0] += 1, C11=P1-... -> W[k1*n+1] += 1
        
        # Let's define the 7 products systematically.
        # U vectors (length 4): [coeff for A[0,0], A[0,1], A[1,0], A[1,1]]
        # V vectors (length 22): coefficients for B[j,k]
        # W vectors (length 22): coefficients for C[i,k] (indexed as k*n+i)
        
        products = [
            # (U_vec, V_indices, V_coeffs, W_indices, W_coeffs)
            # P1 = (A00+A11)*(B0k0+B1k1)
            ([1, 0, 0, 1], [(0, k0, 1), (1, k1, 1)], [(k0, 0, 1), (k1, 1, 1)]),
            # P2 = (A10+A11)*B0k0
            ([0, 0, 1, 1], [(0, k0, 1)], [(k0, 1, 1)]),
            # P3 = A00*(B0k1-B1k1)
            ([1, 0, 0, 0], [(0, k1, 1), (1, k1, -1)], [(k1, 0, 1)]),
            # P4 = A11*(B1k0-B0k0)
            ([0, 0, 0, 1], [(1, k0, 1), (0, k0, -1)], [(k0, 0, 1), (k0, 1, 1)]),
            # P5 = (A00+A01)*B1k1
            ([1, 1, 0, 0], [(1, k1, 1)], [(k1, 0, 1), (k0, 0, -1)]),
            # P6 = (A10-A00)*(B0k0+B0k1)
            ([-1, 0, 1, 0], [(0, k0, 1), (0, k1, 1)], [(k1, 1, 1)]),
            # P7 = (A01-A11)*(B1k0+B1k1)
            ([0, 1, 0, -1], [(1, k0, 1), (1, k1, 1)], [(k0, 0, 1)]),
        ]
        
        for U_vec, V_data, W_data in products:
            for i in range(4):
                U[i][col] = U_vec[i]
            for (j, k, coeff) in V_data:
                V[j * p + k][col] += coeff
            for (k, i, coeff) in W_data:
                W[k * n + i][col] += coeff
            col += 1
    
    if has_single:
        k = p - 1  # 10
        # Schoolbook for the last column:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # Term 1: A[0,0]*B[0,k] -> C[0,k]
        U[0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,1]*B[1,k] -> C[0,k]
        U[1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,0]*B[0,k] -> C[1,k]
        U[2][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,1]*B[1,k] -> C[1,k]
        U[3][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
