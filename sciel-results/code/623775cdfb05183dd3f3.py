"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Strategy: Split m=3 into m1=2, m2=1. Split p=10 into 5 blocks of 2.
    # For each block: Strassen (rank 7) for 2x2x2 + 4 terms for outer product = 11
    # Total: 5 * 11 = 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    col = 0
    for b in range(5):
        # Strassen 7 terms for A_top (2x2) * B_top (2x2)
        # A_top indices in U: A[i,j] -> i*m+j, i in {0,1}, j in {0,1} => 0,1,3,4
        # B_top indices in V: B[j,k] -> j*p+k, j in {0,1}, k in {2b,2b+1} => 2b,2b+1,10+2b,10+2b+1
        # C block indices in W: C[i,k] -> k*n+i, k in {2b,2b+1}, i in {0,1} => 4b,4b+1,4b+2,4b+3
        
        # M1 = (A00+A11)*(B00+B11)
        c = col; col += 1
        U[0][c] = 1; U[4][c] = 1
        V[2*b][c] = 1; V[10+2*b+1][c] = 1
        W[4*b][c] = 1; W[4*b+3][c] = 1
        
        # M2 = (A10+A11)*B00
        c = col; col += 1
        U[3][c] = 1; U[4][c] = 1
        V[2*b][c] = 1
        W[4*b+1][c] = 1
        
        # M3 = A00*(B01-B11)
        c = col; col += 1
        U[0][c] = 1
        V[2*b+1][c] = 1; V[10+2*b+1][c] = -1
        W[4*b+2][c] = 1
        
        # M4 = A11*(B10-B00)
        c = col; col += 1
        U[4][c] = 1
        V[10+2*b][c] = 1; V[2*b][c] = -1
        W[4*b][c] = 1; W[4*b+1][c] = 1
        
        # M5 = (A00+A01)*B11
        c = col; col += 1
        U[0][c] = 1; U[1][c] = 1
        V[10+2*b+1][c] = 1
        W[4*b][c] = -1; W[4*b+2][c] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        c = col; col += 1
        U[3][c] = 1; U[0][c] = -1
        V[2*b][c] = 1; V[2*b+1][c] = 1
        W[4*b+3][c] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        c = col; col += 1
        U[1][c] = 1; U[4][c] = -1
        V[10+2*b][c] = 1; V[10+2*b+1][c] = 1
        W[4*b][c] = 1
        
        # Outer product: A_bot (2x1) * B_bot (1x2)
        # A_bot indices: A[i,2] -> i*m+2 => 2, 5
        # B_bot indices: B[2,k] -> 2*p+k => 20+2b, 20+2b+1
        # 4 terms:
        c = col; col += 1
        U[2][c] = 1; V[20+2*b][c] = 1; W[4*b][c] = 1
        c = col; col += 1
        U[2][c] = 1; V[20+2*b+1][c] = 1; W[4*b+2][c] = 1
        c = col; col += 1
        U[5][c] = 1; V[20+2*b][c] = 1; W[4*b+1][c] = 1
        c = col; col += 1
        U[5][c] = 1; V[20+2*b+1][c] = 1; W[4*b+3][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
