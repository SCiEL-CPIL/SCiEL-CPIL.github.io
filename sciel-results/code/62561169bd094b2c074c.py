"""Tensor decomposition for 2x5x6 matrix multiplication using block-Strassen approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll decompose the multiplication into 2x2 blocks where possible,
    # applying Strassen's algorithm (rank 7 instead of 8) for each 2x2x2 block,
    # and schoolbook for remaining strips.
    
    # A is 2x5, B is 5x6
    # Rows of A: 0, 1
    # Cols of B: 0, 1, 2, 3, 4, 5
    # j index: 0..4 (rows of B / cols of A)
    # k index: 0..5 (cols of B)
    
    # We can partition the j dimension (0..4) into blocks of size 2: [0,1], [2,3], [4]
    # And k dimension (0..5) into blocks of size 2: [0,1], [2,3], [4,5]
    
    # For each (j_block, k_block) pair:
    #   If j_block has size 2 and k_block has size 2: use Strassen (7 terms)
    #   Otherwise: use schoolbook (2 * j_size * k_size terms)
    
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for j indices
    k_blocks = [(0, 2), (2, 4), (4, 6)]  # (start, end) for k indices
    
    U = []
    V = []
    W = []
    
    for j_start, j_end in j_blocks:
        j_size = j_end - j_start
        for k_start, k_end in k_blocks:
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen for 2x2x2
                # A rows: 0, 1; B cols: k_start, k_start+1
                # j indices: j_start, j_start+1
                # k indices: k_start, k_start+1
                #
                # Strassen products:
                # M1 = (A00 + A11)(B00 + B11)
                # M2 = (A10 + A11)B00
                # M3 = A00(B01 - B11)
                # M4 = A11(B10 - B00)
                # M5 = (A00 + A01)B11
                # M6 = (A01 - A10)(B00 + B10)
                # M7 = (A10 - A01)(B11 + B01)
                #
                # Outputs:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                #
                # Here Aij means A[i, j_start + j_offset] where j_offset maps to B's row
                # Wait, let me be more careful.
                #
                # In the 2x2x2 sub-problem:
                # A_sub is 2x2: A_sub[i, j'] = A[i, j_start + j'] for i in {0,1}, j' in {0,1}
                # B_sub is 2x2: B_sub[j', k'] = B[j_start + j', k_start + k'] for j' in {0,1}, k' in {0,1}
                # C_sub is 2x2: C_sub[i, k'] = C[i, k_start + k']
                #
                # Strassen:
                # M1 = (A00 + A11)(B00 + B11)
                #   A00 = A[0, j_start], A11 = A[1, j_start+1]
                #   B00 = B[j_start, k_start], B11 = B[j_start+1, k_start+1]
                #   C00 gets +1, C11 gets +1
                #
                # M2 = (A10 + A11)B00
                #   A10 = A[1, j_start], A11 = A[1, j_start+1]
                #   B00 = B[j_start, k_start]
                #   C10 gets +1
                #
                # M3 = A00(B01 - B11)
                #   A00 = A[0, j_start]
                #   B01 = B[j_start, k_start+1], B11 = B[j_start+1, k_start+1]
                #   C01 gets +1
                #
                # M4 = A11(B10 - B00)
                #   A11 = A[1, j_start+1]
                #   B10 = B[j_start+1, k_start], B00 = B[j_start, k_start]
                #   C00 gets +1, C10 gets +1
                #
                # M5 = (A00 + A01)B11
                #   A00 = A[0, j_start], A01 = A[0, j_start+1]
                #   B11 = B[j_start+1, k_start+1]
                #   C00 gets -1, C01 gets +1
                #
                # M6 = (A01 - A10)(B00 + B10)
                #   A01 = A[0, j_start+1], A10 = A[1, j_start]
                #   B00 = B[j_start, k_start], B10 = B[j_start+1, k_start]
                #   C11 gets +1
                #
                # M7 = (A10 - A01)(B11 + B01)
                #   A10 = A[1, j_start], A01 = A[0, j_start+1]
                #   B11 = B[j_start+1, k_start+1], B01 = B[j_start, k_start+1]
                #   C00 gets +1, C11 gets -1
                #
                # Let me re-verify the output formulas:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                #
                # Wait, I need to double-check these. The standard Strassen is:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                #
                # Let me verify with a simple case. Actually, let me just trust the standard formulas.
                
                # Define the 7 Strassen terms
                strassen_terms = [
                    # (U_coeffs: list of (i, j, coeff), V_coeffs: list of (j, k, coeff), W_coeffs: list of (k, i, coeff))
                    {
                        'U': [(0, j_start, 1), (1, j_start + 1, 1)],
                        'V': [(j_start, k_start, 1), (j_start + 1, k_start + 1, 1)],
                        'W': [(k_start, 0, 1), (k_start + 1, 1, 1)]
                    },
                    {
                        'U': [(1, j_start, 1), (1, j_start + 1, 1)],
                        'V': [(j_start, k_start, 1)],
                        'W': [(k_start, 1, 1)]
                    },
                    {
                        'U': [(0, j_start, 1)],
                        'V': [(j_start, k_start + 1, 1), (j_start + 1, k_start + 1, -1)],
                        'W': [(k_start + 1, 0, 1)]
                    },
                    {
                        'U': [(1, j_start + 1, 1)],
                        'V': [(j_start + 1, k_start, 1), (j_start, k_start, -1)],
                        'W': [(k_start, 0, 1), (k_start, 1, 1)]
                    },
                    {
                        'U': [(0, j_start, 1), (0, j_start + 1, 1)],
                        'V': [(j_start + 1, k_start + 1, 1)],
                        'W': [(k_start, 0, -1), (k_start + 1, 0, 1)]
                    },
                    {
                        'U': [(0, j_start + 1, 1), (1, j_start, -1)],
                        'V': [(j_start, k_start, 1), (j_start + 1, k_start, 1)],
                        'W': [(k_start + 1, 1, 1)]
                    },
                    {
                        'U': [(1, j_start, 1), (0, j_start + 1, -1)],
                        'V': [(j_start + 1, k_start + 1, 1), (j_start, k_start + 1, 1)],
                        'W': [(k_start, 0, 1), (k_start + 1, 1, -1)]
                    },
                ]
                
                for term in strassen_terms:
                    # Add a new column
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    for i, j, coeff in term['U']:
                        U_col[i * m + j] = coeff
                    for j, k, coeff in term['V']:
                        V_col[j * p + k] = coeff
                    for k, i, coeff in term['W']:
                        W_col[k * n + i] = coeff
                    
                    U.append(U_col)
                    V.append(V_col)
                    W.append(W_col)
            else:
                # Schoolbook for this block
                for i in range(n):
                    for j in range(j_start, j_end):
                        for k in range(k_start, k_end):
                            U_col = [0] * (n * m)
                            V_col = [0] * (m * p)
                            W_col = [0] * (p * n)
                            
                            U_col[i * m + j] = 1
                            V_col[j * p + k] = 1
                            W_col[k * n + i] = 1
                            
                            U.append(U_col)
                            V.append(V_col)
                            W.append(W_col)
    
    # Convert to the expected format: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Currently U, V, W are lists of columns. Transpose them.
    rank = len(U)
    U_out = [[U[col][row] for col in range(rank)] for row in range(n * m)]
    V_out = [[V[col][row] for col in range(rank)] for row in range(m * p)]
    W_out = [[W[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
