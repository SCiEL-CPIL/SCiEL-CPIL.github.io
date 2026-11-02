# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    
    # Rank 18 decomposition:
    # Block 1: Cols 0,1 (2x2) -> Strassen (Rank 7)
    # Block 2: Col 2 (2x1) -> Schoolbook (Rank 4)
    # Block 3: Cols 3,4 (2x2) -> Strassen (Rank 7)
    
    total_rank = 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a Strassen block for A(2x2) * B(2x2) -> C(2x2)
    # B columns are c0, c1
    def add_strassen_block(start_rank, c0, c1):
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)(B00+B01)
        # P3 = (A00+A01)(B10+B11)
        # P4 = A00(B00-B11)
        # P5 = A11(B10-B00)
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P3 + P6
        
        r = start_rank
        
        # P1
        U[0][r] = 1; U[3][r] = 1
        V[0*p+c0][r] = 1; V[1*p+c1][r] = 1
        W[c0*n+0][r] = 1; W[c1*n+1][r] = 1
        r += 1
        
        # P2
        U[2][r] = 1; U[3][r] = 1
        V[0*p+c0][r] = 1; V[0*p+c1][r] = 1
        W[c0*n+1][r] = 1; W[c1*n+1][r] = 1
        r += 1
        
        # P3
        U[0][r] = 1; U[1][r] = 1
        V[1*p+c0][r] = 1; V[1*p+c1][r] = 1
        W[c0*n+0][r] = 1; W[c1*n+0][r] = 1
        r += 1
        
        # P4
        U[0][r] = 1
        V[0*p+c0][r] = 1; V[1*p+c1][r] = -1
        W[c0*n+1][r] = 1; W[c1*n+1][r] = 1
        r += 1
        
        # P5
        U[3][r] = 1
        V[1*p+c0][r] = 1; V[0*p+c0][r] = -1
        W[c0*n+0][r] = 1; W[c1*n+0][r] = 1
        r += 1
        
        # P6
        U[2][r] = 1; U[0][r] = -1
        V[0*p+c0][r] = 1; V[0*p+c1][r] = 1
        W[c0*n+1][r] = 1; W[c1*n+1][r] = 1
        r += 1
        
        # P7
        U[1][r] = 1; U[3][r] = -1
        V[1*p+c0][r] = 1; V[1*p+c1][r] = 1
        W[c0*n+0][r] = 1; W[c1*n+0][r] = 1
        r += 1
        
        return r

    # Helper to add a Schoolbook block for A(2x2) * B(2x1) -> C(2x1)
    def add_schoolbook_block(start_rank, c0):
        r = start_rank
        # C00 = A00*B00
        U[0][r] = 1; V[0*p+c0][r] = 1; W[c0*n+0][r] = 1
        r += 1
        # C01 = A01*B10 (Wait, C is 2x1, so only C00 and C10? No, C is 2x1, indices C0, C1)
        # C[i, c0] = A[i,0]B[0,c0] + A[i,1]B[1,c0]
        # C0,c0 = A00*B0,c0 + A01*B1,c0
        # C1,c0 = A10*B0,c0 + A11*B1,c0
        
        # Term 1: A00 * B0,c0 -> C0,c0
        # Already did C00? No, I did U[0] A00, V[0] B0. W[0] C0.
        # Let's restart the schoolbook helper to be clear.
        
        # A00 * B0,c0 -> C0,c0
        U[0][r] = 1; V[0*p+c0][r] = 1; W[c0*n+0][r] = 1
        r += 1
        # A01 * B1,c0 -> C0,c0
        U[1][r] = 1; V[1*p+c0][r] = 1; W[c0*n+0][r] = 1
        r += 1
        # A10 * B0,c0 -> C1,c0
        U[2][r] = 1; V[0*p+c0][r] = 1; W[c0*n+1][r] = 1
        r += 1
        # A11 * B1,c0 -> C1,c0
        U[3][r] = 1; V[1*p+c0][r] = 1; W[c0*n+1][r] = 1
        r += 1
        
        return r

    current_rank = 0
    current_rank = add_strassen_block(current_rank, 0, 1)
    current_rank = add_schoolbook_block(current_rank, 2)
    current_rank = add_strassen_block(current_rank, 3, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
