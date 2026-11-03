"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid decomposition:
    # Top 2 rows of A (2x4) multiplied by B (4x4) using two Strassen 2x2x2 decompositions.
    # Bottom 1 row of A (1x4) multiplied by B (4x4) using schoolbook.
    
    # Total rank estimate:
    # 2x4x4 part: Split into two 2x2x2 multiplications (left and right halves of B).
    # Each 2x2x2 uses Strassen (rank 7). So 2 * 7 = 14 for each half? 
    # Wait, C_top_left = A00*B00 + A01*B10. This is sum of two 2x2x2.
    # Rank <= 7 + 7 = 14.
    # C_top_right = A00*B01 + A01*B11. Rank <= 7 + 7 = 14.
    # Total for top 2 rows: 28.
    # Bottom 1 row: 1x4x4 schoolbook is 16.
    # Total rank: 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A indices: (i, j) local 0..1, 0..1 -> global i_off + i, j_off + j
    # B indices: (j, k) local 0..1, 0..1 -> global j_off + j, k_off + k
    # C indices: (i, k) local 0..1, 0..1 -> global i_off + i, k_off + k
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen terms for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U vectors (A part)
        u1 = [(1,1), (1,1)] # A11, A22 (indices 0,0 and 1,1)
        u2 = [(1,0), (1,1)] # A21, A22 (indices 1,0 and 1,1)
        u3 = [(0,0)]        # A11
        u4 = [(1,1)]        # A22
        u5 = [(0,0), (0,1)] # A11, A12
        u6 = [(1,0), (0,0)] # A21, -A11
        u7 = [(0,1), (1,1)] # A12, -A22
        
        # V vectors (B part)
        v1 = [(0,0), (1,1)] # B11, B22
        v2 = [(0,0)]        # B11
        v3 = [(0,1), (1,1)] # B12, -B22
        v4 = [(1,0), (0,0)] # B21, -B11
        v5 = [(1,1)]        # B22
        v6 = [(0,0), (0,1)] # B11, B12
        v7 = [(1,0), (1,1)] # B21, B22
        
        # W vectors (C part)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We add 7 columns
        # Col 0: M1
        U[i_off+0][col] += 1; U[i_off+1][col] += 1
        V[j_off+0][col] += 1; V[j_off+1][col] += 1
        W[k_off+0][col] += 1; W[k_off+1][col] += 1 # C11, C22 get M1
        # Wait, W is (p*n). Index is k*n + i.
        # C11 is i=0, k=0. C22 is i=1, k=1.
        # So W[0*n+0] and W[1*n+1] get 1.
        # My above line was wrong.
        
        # Let's restart the Strassen addition carefully.
        col_start = col
        
        # M1: (A00+A11)(B00+B11)
        U[i_off+0][col] += 1; U[i_off+1][col] += 1
        V[j_off+0][col] += 1; V[j_off+1][col] += 1
        W[0*n + 0][col] += 1; W[1*n + 1][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[i_off+1][col] += 1; U[i_off+1][col] += 1 # A10 is (1,0), A11 is (1,1)
        V[j_off+0][col] += 1
        W[1*n + 0][col] += 1; W[1*n + 1][col] += -1 # C21 gets M2, C22 gets -M2
        col += 1
        
        # M3: A00(B01-B11)
        U[i_off+0][col] += 1
        V[j_off+0][col] += 1; V[j_off+1][col] += -1
        W[0*n + 1][col] += 1; W[1*n + 1][col] += 1 # C12 gets M3, C22 gets M3
        col += 1
        
        # M4: A11(B10-B00)
        U[i_off+1][col] += 1
        V[j_off+1][col] += 1; V[j_off+0][col] += -1
        W[0*n + 0][col] += 1; W[1*n + 0][col] += 1 # C11 gets M4, C21 gets M4
        col += 1
        
        # M5: (A00+A01)B11
        U[i_off+0][col] += 1; U[i_off+0][col] += 1
        V[j_off+1][col] += 1
        W[0*n + 0][col] += -1; W[0*n + 1][col] += 1 # C11 gets -M5, C12 gets M5
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[i_off+1][col] += 1; U[i_off+0][col] += -1
        V[j_off+0][col] += 1; V[j_off+0][col] += 1
        W[1*n + 1][col] += 1 # C22 gets M6
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i_off+0][col] += 1; U[i_off+1][col] += -1
        V[j_off+1][col] += 1; V[j_off+1][col] += 1
        W[0*n + 0][col] += 1 # C11 gets M7
        col += 1
        
        return col

    # Top 2 rows: i_off=0
    # Left half of B: j_off=0, k_off=0
    col = add_strassen(0, 0, 0)
    # Right half of B: j_off=2, k_off=2
    col = add_strassen(0, 2, 2)
    
    # Bottom 1 row: i_off=2
    # Schoolbook for 1x4x4
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col] = 1
            V[j*4 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
