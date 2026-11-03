import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition where we split the m=4 and p=6 dimensions into 2x2 blocks.
    # There are 2 blocks in m (heights) and 3 blocks in p (widths). Total 6 blocks of size 2x2x2.
    # For each 2x2x2 block, we use Strassen's algorithm (rank 7).
    # Total rank = 6 * 7 = 42.
    # This is better than schoolbook rank 2*4*6 = 48.

    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen factors for 2x2 matrix multiplication.
    # Indices in local 2x2x2 space: i,j,k in {0,1}
    # U_local[i*2+j, r], V_local[j*2+k, r], W_local[k*2+i, r]
    
    # Let's define the 7 factors for Strassen.
    # The standard Strassen factors (from memory, need to be precise):
    # C = A * B
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # We need to map these to the tensor factors.
    # For a given factor r, U[i,j] is the coefficient of A_ij, V[j,k] is coefficient of B_jk, W[k,i] is coefficient of C_ki (Wait, W index is k*n+i? Yes, W[k*n+i, r] is C[i,k] transposed? No, trace convention W[k*n+i, r] usually means the output element at position corresponding to row i, col k? Let's check the schoolbook code.
    # Schoolbook: W[k*n + i] = 1. C[i,k] = sum A[i,j] B[j,k]. So W[k*n+i] corresponds to C[i,k].
    # So W_local[k*2+i, r] is the coefficient for C[i,k] in the local 2x2 block.

    # Let's build the local U, V, W matrices for Strassen (2x4) U/V (4x7) and W (4x7).
    # Wait, local U is 2x2=4 rows, 7 cols. Local V is 2x2=4 rows, 7 cols. Local W is 2x2=4 rows, 7 cols.
    
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # M1: (A00 + A11)(B00 + B11) -> C00 += 1, C10 -= 1, C01 -= 1, C11 += 1? No.
    # C00 += M1, C01 += M5, C10 += M2, C11 += M6... wait.
    # Let's use the formulas directly.
    
    # M1: U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C10=1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Factor 1: M1
    # U: A00, A11 are 1.
    U_loc[0][0] += 1; U_loc[3][0] += 1
    # V: B00, B11 are 1.
    V_loc[0][0] += 1; V_loc[3][0] += 1
    # W: C00 += 1, C10? No, W index is k*2+i.
    # C00 (i=0,k=0) -> W[0] += 1.
    # C10 (i=0,k=1)? No, C[0,1] is i=0, k=1 -> W[1*2+0] = W[2] += 1?
    # Wait, C11 (i=1,k=1) -> W[1*2+1] = W[3] += 1?
    # Let's re-verify indices.
    # C[i,k]. W index = k*2 + i.
    # C00 (i=0,k=0) -> W[0]
    # C01 (i=0,k=1) -> W[2]
    # C10 (i=1,k=0) -> W[1]
    # C11 (i=1,k=1) -> W[3]
    
    # C00 = M1 ... => W[0] += 1
    # C01 = M3 + M5 => W[2] += 0 (for M1)
    # C10 = M2 + M4 => W[1] += 0 (for M1)
    # C11 = M1 - M2 + M3 + M6 => W[3] += 1
    W_loc[0][0] += 1
    W_loc[3][0] += 1
    
    # Factor 2: M2 = (A10 + A11) B00
    U_loc[2][1] += 1; U_loc[3][1] += 1
    V_loc[0][1] += 1
    W_loc[1][1] += 1  # C10 += M2
    W_loc[3][1] -= 1  # C11 -= M2
    
    # Factor 3: M3 = A00 (B01 - B11)
    U_loc[0][2] += 1
    V_loc[1][2] += 1
    V_loc[3][2] -= 1
    W_loc[2][2] += 1  # C01 += M3
    W_loc[3][2] += 1  # C11 += M3
    
    # Factor 4: M4 = A11 (B10 - B00)
    U_loc[3][3] += 1
    V_loc[2][3] += 1
    V_loc[0][3] -= 1
    W_loc[1][3] += 1  # C10 += M4
    W_loc[0][3] += 1  # C00 += M4
    
    # Factor 5: M5 = (A00 + A01) B11
    U_loc[0][4] += 1
    U_loc[1][4] += 1
    V_loc[3][4] += 1
    W_loc[0][4] -= 1  # C00 -= M5
    W_loc[2][4] += 1  # C01 += M5
    
    # Factor 6: M6 = (A10 - A00) (B00 + B01)
    U_loc[2][5] += 1
    U_loc[0][5] -= 1
    V_loc[0][5] += 1
    V_loc[1][5] += 1
    W_loc[3][5] += 1  # C11 += M6
    
    # Factor 7: M7 = (A01 - A11) (B10 + B11)
    U_loc[1][6] += 1
    U_loc[3][6] -= 1
    V_loc[2][6] += 1
    V_loc[3][6] += 1
    W_loc[0][6] += 1  # C00 += M7
    
    # Now, embed these into the global matrices.
    # Global dimensions: n=2, m=4, p=6.
    # Blocks: m is split into 2 blocks of size 2 (rows 0-1, 2-3).
    # p is split into 3 blocks of size 2 (cols 0-1, 2-3, 4-5).
    # So we have 2x3 = 6 blocks.
    # Let block_row = 0,1 and block_col = 0,1,2.
    # Local indices i_loc, j_loc, k_loc in {0,1}.
    # Global i = i_loc.
    # Global j = block_row * 2 + j_loc.
    # Global k = block_col * 2 + k_loc.
    
    for br in range(2):
        for bc in range(3):
            for r in range(7):
                col = (br * 3 + bc) * 7 + r
                
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_val = U_loc[i_loc * 2 + j_loc][r]
                        if u_val != 0:
                            i = i_loc
                            j = br * 2 + j_loc
                            U[i * 4 + j][col] += u_val
                        
                        for k_loc in range(2):
                            w_val = W_loc[k_loc * 2 + i_loc][r]
                            if w_val != 0:
                                k = bc * 2 + k_loc
                                W[k * 2 + i_loc][col] += w_val
                            
                    for j_loc in range(2):
                        for k_loc in range(2):
                            v_val = V_loc[j_loc * 2 + k_loc][r]
                            if v_val != 0:
                                j = br * 2 + j_loc
                                k = bc * 2 + k_loc
                                V[j * 6 + k][col] += v_val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
