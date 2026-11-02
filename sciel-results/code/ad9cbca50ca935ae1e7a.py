import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Strategy: Use Strassen-based block decomposition to reduce rank below schoolbook (112)
    # Split 4x4 * 4x7 into 2x2 blocks
    # A (4x4) split into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B (4x7) split into: B = [[B00, B01, B02], [B10, B11, B12]]
    #   where B00, B10 are 2x2; B01, B11 are 2x2; B02, B12 are 2x3
    # C (4x7) = [[C00, C01, C02], [C10, C11, C12]]
    #   C00 = A00*B00 + A01*B10  (2x2)
    #   C01 = A00*B01 + A01*B11  (2x2)
    #   C02 = A00*B02 + A01*B12  (2x3)
    #   C10 = A10*B00 + A11*B10  (2x2)
    #   C11 = A10*B01 + A11*B11  (2x2)
    #   C12 = A10*B02 + A11*B12  (2x3)
    
    # For 2x2 * 2x2: use Strassen (rank 7) instead of schoolbook (rank 4)
    # For 2x2 * 2x3: need a decomposition. Let's use rank 8 (schoolbook is 6, but we can do better)
    # Actually, 2x2 * 2x3 schoolbook is rank 6. Let's try to find a lower rank.
    # The tensor for 2x2 * 2x3 has 6 entries in output. Minimum rank is at least 6.
    # Let's use schoolbook for 2x2 * 2x3 (rank 6) for now.
    
    # Total rank estimate:
    # 4 blocks of 2x2*2x2 using Strassen: 4 * 7 = 28
    # 2 blocks of 2x2*2x3 using schoolbook: 2 * 6 = 12
    # Total: 40
    
    # But we need to be careful about how we combine these.
    # Each Strassen product for 2x2*2x2 involves 7 scalar multiplications.
    # Each schoolbook product for 2x2*2x3 involves 6 scalar multiplications.
    
    # Let's implement this step by step.
    
    # First, let's define helper functions for Strassen and 2x2*2x3 schoolbook.
    
    def strassen_2x2x2():
        """Return U, V, W matrices for 2x2 * 2x2 using Strassen (rank 7)."""
        # Indices: A is 2x2 (indices 0,1,2,3), B is 2x2 (indices 0,1,2,3), C is 2x2 (indices 0,1,2,3)
        # U has 4 rows (A indices), V has 4 rows (B indices), W has 4 rows (C indices)
        # Each has 7 columns (7 products)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for [[a,b],[c,d]] * [[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # A indices: a=0, b=1, c=2, d=3
        # B indices: e=0, f=1, g=2, h=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2 = (c+d)e
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # C10
        W[3, 1] = -1  # C11
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[3, 2] = -1  # h
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g
        V[0, 3] = -1  # e
        W[0, 3] = -1  # C00
        W[2, 3] = 1  # C10
        
        # M5 = (a+b)h
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b
        V[3, 4] = 1  # h
        W[0, 4] = -1  # C00
        W[1, 4] = 1  # C01
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1  # c
        U[0, 5] = -1  # a
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        W[3, 5] = -1  # C11
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1  # b
        U[3, 6] = -1  # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        W[0, 6] = 1  # C00
        W[1, 6] = 0  # C01 (no contribution)
        
        return U, V, W
    
    def schoolbook_2x2x3():
        """Return U, V, W matrices for 2x2 * 2x3 using schoolbook (rank 6)."""
        # A is 2x2 (indices 0,1,2,3), B is 2x3 (indices 0,1,2,3,4,5), C is 2x3 (indices 0,1,2,3,4,5)
        # U has 4 rows, V has 6 rows, W has 6 rows, each with 6 columns
        
        U = np.zeros((4, 6))
        V = np.zeros((6, 6))
        W = np.zeros((6, 6))
        
        # Products:
        # P1 = a*e -> C00
        # P2 = a*f -> C01
        # P3 = a*g -> C02
        # P4 = b*h -> C00
        # P5 = b*i -> C01
        # P6 = b*j -> C02
        # P7 = c*e -> C10
        # P8 = c*f -> C11
        # P9 = c*g -> C12
        # P10 = d*h -> C10
        # P11 = d*i -> C11
        # P12 = d*j -> C12
        
        # Wait, that's 12 products for schoolbook. Let me reconsider.
        # 2x2 * 2x3: A has 4 entries, B has 6 entries, C has 6 entries.
        # Schoolbook: for each C[i,j], sum over k of A[i,k]*B[k,j]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # So we need 12 scalar multiplications for schoolbook.
        # But we can potentially do better. Let's use rank 12 for now.
        
        # Actually, let's just use the full schoolbook with 12 products.
        # U: 4 rows (A indices), V: 6 rows (B indices), W: 6 rows (C indices), 12 columns
        
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        
        col = 0
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        U[0, col] = 1; V[0, col] = 1; W[0, col] = 1; col += 1  # A00 * B00
        U[1, col] = 1; V[3, col] = 1; W[0, col] = 1; col += 1  # A01 * B10
        
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        U[0, col] = 1; V[1, col] = 1; W[1, col] = 1; col += 1  # A00 * B01
        U[1, col] = 1; V[4, col] = 1; W[1, col] = 1; col += 1  # A01 * B11
        
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        U[0, col] = 1; V[2, col] = 1; W[2, col] = 1; col += 1  # A00 * B02
        U[1, col] = 1; V[5, col] = 1; W[2, col] = 1; col += 1  # A01 * B12
        
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U[2, col] = 1; V[0, col] = 1; W[3, col] = 1; col += 1  # A10 * B00
        U[3, col] = 1; V[3, col] = 1; W[3, col] = 1; col += 1  # A11 * B10
        
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        U[2, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1  # A10 * B01
        U[3, col] = 1; V[4, col] = 1; W[4, col] = 1; col += 1  # A11 * B11
        
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        U[2, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1  # A10 * B02
        U[3, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1  # A11 * B12
        
        return U, V, W
    
    # Now combine these block decompositions.
    # We have 4 blocks of 2x2*2x2 (using Strassen, rank 7 each) and 2 blocks of 2x2*2x3 (using schoolbook, rank 12 each).
    # Total rank: 4*7 + 2*12 = 28 + 24 = 52
    
    # But we need to map the block indices to the global indices.
    # Let's build the full U, V, W matrices.
    
    total_rank = 4 * 7 + 2 * 12  # 52
    U_full = np.zeros((n * m, total_rank))  # 16 x 52
    V_full = np.zeros((m * p, total_rank))  # 28 x 52
    W_full = np.zeros((p * n, total_rank))  # 28 x 52
    
    col_offset = 0
    
    # Block (0,0): A00 * B00 + A01 * B10 -> C00 (2x2)
    # A00 is rows 0-1, cols 0-1 of A -> global A indices 0,1,4,5
    # B00 is rows 0-1, cols 0-1 of B -> global B indices 0,1,7,8
    # C00 is rows 0-1, cols 0-1 of C -> global C indices 0,1,4,5
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    # Block (0,0): A00 * B00 + A01 * B10 -> C00 (2x2)
    # A00: rows 0-1, cols 0-1 -> A indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B00: rows 0-1, cols 0-1 -> B indices: B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # C00: rows 0-1, cols 0-1 -> C[i,k] at k*4+i: C[0,0]=0, C[0,1]=4, C[1,0]=1, C[1,1]=5
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [0, 1, 4, 5]
    b_map = [0, 1, 7, 8]
    # Strassen local: C00=0, C01=1, C10=2, C11=3
    # Global: C00->0*4+0=0, C01->1*4+0=4, C10->0*4+1=1, C11->1*4+1=5
    c_map = [0, 4, 1, 5]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,1): A00 * B01 + A01 * B11 -> C01 (2x2)
    # A00: global 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> B[0,2]=2, B[0,3]=3, B[1,2]=9, B[1,3]=10
    # C01: rows 0-1, cols 2-3 -> C[0,2]=2*4+0=8, C[0,3]=3*4+0=12, C[1,2]=2*4+1=9, C[1,3]=3*4+1=13
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [0, 1, 4, 5]
    b_map = [2, 3, 9, 10]
    c_map = [8, 12, 9, 13]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,0): A10 * B00 + A11 * B10 -> C10 (2x2)
    # A10: rows 2-3, cols 0-1 -> A[2,0]=8, A[2,1]=9, A[3,0]=12, A[3,1]=13
    # B00: global 0,1,7,8
    # C10: rows 2-3, cols 0-1 -> C[2,0]=0*4+2=2, C[2,1]=1*4+2=6, C[3,0]=0*4+3=3, C[3,1]=1*4+3=7
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [8, 9, 12, 13]
    b_map = [0, 1, 7, 8]
    c_map = [2, 6, 3, 7]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,1): A10 * B01 + A11 * B11 -> C11 (2x2)
    # A10: global 8,9,12,13
    # B01: global 2,3,9,10
    # C11: rows 2-3, cols 2-3 -> C[2,2]=2*4+2=10, C[2,3]=3*4+2=14, C[3,2]=2*4+3=11, C[3,3]=3*4+3=15
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [8, 9, 12, 13]
    b_map = [2, 3, 9, 10]
    c_map = [10, 14, 11, 15]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,2): A00 * B02 + A01 * B12 -> C02 (2x3)
    # A00: global 0,1,4,5
    # B02: rows 0-1, cols 4-6 -> B[0,4]=4, B[0,5]=5, B[0,6]=6, B[1,4]=11, B[1,5]=12, B[1,6]=13
    # C02: rows 0-1, cols 4-6 -> C[0,4]=4*4+0=16, C[0,5]=5*4+0=20, C[0,6]=6*4+0=24, C[1,4]=4*4+1=17, C[1,5]=5*4+1=21, C[1,6]=6*4+1=25
    
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    
    a_map = [0, 1, 4, 5]
    b_map = [4, 5, 6, 11, 12, 13]
    # Schoolbook local C indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    # Global: C[0,4]=16, C[0,5]=20, C[0,6]=24, C[1,4]=17, C[1,5]=21, C[1,6]=25
    c_map = [16, 20, 24, 17, 21, 25]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Block (1,2): A10 * B02 + A11 * B12 -> C12 (2x3)
    # A10: global 8,9,12,13
    # B02: global 4,5,6,11,12,13
    # C12: rows 2-3, cols 4-6 -> C[2,4]=4*4+2=18, C[2,5]=5*4+2=22, C[2,6]=6*4+2=26, C[3,4]=4*4+3=19, C[3,5]=5*4+3=23, C[3,6]=6*4+3=27
    
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    
    a_map = [8, 9, 12, 13]
    b_map = [4, 5, 6, 11, 12, 13]
    c_map = [18, 22, 26, 19, 23, 27]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Verify the decomposition exhaustively (n*m*p = 112 is small)
    def verify(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    for c in range(rank):
                        val += ui[c] * vj[c] * wk[c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
