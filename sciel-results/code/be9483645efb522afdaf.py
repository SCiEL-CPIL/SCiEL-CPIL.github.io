"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 exact tensor decomposition for 2x3 * 3x10 multiplication.
    Uses Strassen's algorithm on 2x2 sub-blocks (for j=0,1) plus schoolbook for j=2.
    5 blocks of 2 columns each -> 5 * (7 Strassen + 4 schoolbook for j=2) = 55 terms.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    # For each block of 2 columns (k_start, k_start+1)
    for b in range(5):
        k0 = b * 2
        k1 = b * 2 + 1
        
        # Strassen's algorithm for the 2x2 sub-block (i=0,1; j=0,1; k=k0,k1)
        # 7 multiplications
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1]) -- wait, need to be careful
        # Let me redo Strassen properly for C = A*B where A is 2x2, B is 2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # In our case:
        # A[i,j] where i in {0,1}, j in {0,1}
        # B[j,k] where j in {0,1}, k in {k0, k1}
        # C[i,k] where i in {0,1}, k in {k0, k1}
        
        # So B[0,k0] = B[0, k0], B[0,k1] = B[0, k1]
        #    B[1,k0] = B[1, k0], B[1,k1] = B[1, k1]
        
        # Strassen:
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])  -- NO, this is wrong
        # Standard Strassen for C = A*B:
        # M1 = (A11 - A22) * (B21 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = (A11 + A12) * B22
        # M4 = A11 * (B12 - B22)
        # M5 = A22 * (B21 - B11)
        # M6 = (A12 - A22) * (B21 + B22)
        # M7 = (A11 - A21) * (B11 + B12)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Using 1-indexed notation from Strassen, mapping to our 0-indexed:
        # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
        # B11 = B[0,k0], B12 = B[0,k1], B21 = B[1,k0], B22 = B[1,k1]
        
        # M1 = (A[0,0] - A[1,1]) * (B[1,k0] + B[1,k1])
        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        # M3 = (A[0,0] + A[0,1]) * B[1,k1]
        # M4 = A[0,0] * (B[0,k1] - B[1,k1])
        # M5 = A[1,1] * (B[1,k0] - B[0,k0])
        # M6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        # M7 = (A[0,0] - A[1,0]) * (B[0,k0] + B[0,k1])
        
        # C[0,k0] = M1 + M4 - M5 + M7
        # C[0,k1] = M3 + M5
        # C[1,k0] = M2 + M4
        # C[1,k1] = M1 - M2 + M3 + M6
        
        # Let's define the 7 rank-1 terms for Strassen
        # Each term has U, V, W entries
        
        col = b * 11  # starting column index for this block
        
        # Term 1: M1 = (A[0,0] - A[1,1]) * (B[1,k0] + B[1,k1])
        # U: A[0,0] -> row 0*3+0=0, coeff +1; A[1,1] -> row 1*3+1=4, coeff -1
        # V: B[1,k0] -> row 1*10+k0, coeff +1; B[1,k1] -> row 1*10+k1, coeff +1
        # W: C[0,k0] -> row k0*2+0, coeff +1; C[1,k1] -> row k1*2+1, coeff +1
        c = col
        U[0][c] = 1
        U[4][c] = -1
        V[10 + k0][c] = 1
        V[10 + k1][c] = 1
        W[k0 * 2 + 0][c] = 1
        W[k1 * 2 + 1][c] = 1
        
        # Term 2: M2 = (A[1,0] + A[1,1]) * B[0,k0]
        # U: A[1,0] -> row 1*3+0=3, coeff +1; A[1,1] -> row 4, coeff +1
        # V: B[0,k0] -> row 0*10+k0 = k0, coeff +1
        # W: C[1,k0] -> row k0*2+1, coeff +1
        c = col + 1
        U[3][c] = 1
        U[4][c] = 1
        V[k0][c] = 1
        W[k0 * 2 + 1][c] = 1
        
        # Term 3: M3 = (A[0,0] + A[0,1]) * B[1,k1]
        # U: A[0,0] -> row 0, coeff +1; A[0,1] -> row 1, coeff +1
        # V: B[1,k1] -> row 10+k1, coeff +1
        # W: C[0,k1] -> row k1*2+0, coeff +1
        c = col + 2
        U[0][c] = 1
        U[1][c] = 1
        V[10 + k1][c] = 1
        W[k1 * 2 + 0][c] = 1
        
        # Term 4: M4 = A[0,0] * (B[0,k1] - B[1,k1])
        # U: A[0,0] -> row 0, coeff +1
        # V: B[0,k1] -> row k1, coeff +1; B[1,k1] -> row 10+k1, coeff -1
        # W: C[0,k0] -> row k0*2+0, coeff +1; C[1,k0] -> row k0*2+1, coeff +1
        c = col + 3
        U[0][c] = 1
        V[k1][c] = 1
        V[10 + k1][c] = -1
        W[k0 * 2 + 0][c] = 1
        W[k0 * 2 + 1][c] = 1
        
        # Term 5: M5 = A[1,1] * (B[1,k0] - B[0,k0])
        # U: A[1,1] -> row 4, coeff +1
        # V: B[1,k0] -> row 10+k0, coeff +1; B[0,k0] -> row k0, coeff -1
        # W: C[0,k0] -> row k0*2+0, coeff -1; C[0,k1] -> row k1*2+0, coeff +1
        c = col + 4
        U[4][c] = 1
        V[10 + k0][c] = 1
        V[k0][c] = -1
        W[k0 * 2 + 0][c] = -1
        W[k1 * 2 + 0][c] = 1
        
        # Term 6: M6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        # U: A[0,1] -> row 1, coeff +1; A[1,1] -> row 4, coeff -1
        # V: B[1,k0] -> row 10+k0, coeff +1; B[1,k1] -> row 10+k1, coeff +1
        # W: C[1,k1] -> row k1*2+1, coeff +1
        c = col + 5
        U[1][c] = 1
        U[4][c] = -1
        V[10 + k0][c] = 1
        V[10 + k1][c] = 1
        W[k1 * 2 + 1][c] = 1
        
        # Term 7: M7 = (A[0,0] - A[1,0]) * (B[0,k0] + B[0,k1])
        # U: A[0,0] -> row 0, coeff +1; A[1,0] -> row 3, coeff -1
        # V: B[0,k0] -> row k0, coeff +1; B[0,k1] -> row k1, coeff +1
        # W: C[0,k0] -> row k0*2+0, coeff +1
        c = col + 6
        U[0][c] = 1
        U[3][c] = -1
        V[k0][c] = 1
        V[k1][c] = 1
        W[k0 * 2 + 0][c] = 1
        
        # Schoolbook for j=2 (A[i,2] * B[2,k]) for i in {0,1}, k in {k0,k1}
        # 4 terms
        # Term 8: A[0,2] * B[2,k0]
        c = col + 7
        U[2][c] = 1  # row 0*3+2=2
        V[20 + k0][c] = 1  # row 2*10+k0
        W[k0 * 2 + 0][c] = 1  # C[0,k0]
        
        # Term 9: A[0,2] * B[2,k1]
        c = col + 8
        U[2][c] = 1
        V[20 + k1][c] = 1
        W[k1 * 2 + 0][c] = 1  # C[0,k1]
        
        # Term 10: A[1,2] * B[2,k0]
        c = col + 9
        U[5][c] = 1  # row 1*3+2=5
        V[20 + k0][c] = 1
        W[k0 * 2 + 1][c] = 1  # C[1,k0]
        
        # Term 11: A[1,2] * B[2,k1]
        c = col + 10
        U[5][c] = 1
        V[20 + k1][c] = 1
        W[k1 * 2 + 1][c] = 1  # C[1,k1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
