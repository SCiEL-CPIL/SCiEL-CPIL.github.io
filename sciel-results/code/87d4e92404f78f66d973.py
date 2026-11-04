"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Split middle dimension: m = 2 + 1
    # Part 1: A1 (2x2) * B1 (2x10) -> split into 5 blocks of 2x2*2x2, use Strassen rank 7 each
    # Part 2: A2 (2x1) * B2 (1x10) -> schoolbook rank 20
    # Total rank: 5*7 + 20 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # Strassen for 2x2x2 blocks
    # For block b (b=0..4), columns 2b, 2b+1 of the 2x10 B1
    # A1 rows: 0,1 (first 2 rows of A, which are the first 2 of the 3 middle dims)
    # B1 rows: 0,1 (first 2 rows of B)
    # Output: C columns 2b, 2b+1, rows 0,1
    
    for b in range(5):
        # Strassen 7 multiplications for 2x2 * 2x2
        # U indices: i*m + j where i in {0,1}, j in {0,1} -> indices 0,1,2,3
        # V indices: j*p + k where j in {0,1}, k in {2b, 2b+1} -> indices 0+2b, 0+2b+1, 1*10+2b, 1*10+2b+1
        # W indices: k*n + i where k in {2b, 2b+1}, i in {0,1} -> indices 2b*2+0, 2b*2+1, (2b+1)*2+0, (2b+1)*2+1
        
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1 = (P11+P22)*(Q11+Q22)
        # U: P11=(i=0,j=0) idx 0, P22=(i=1,j=1) idx 3
        # V: Q11=(j=0,k=k1) idx k1, Q22=(j=1,k=k2) idx 10+k2
        # W: R11=(k=k1,i=0) idx 2*k1, R22=(k=k2,i=1) idx 2*k2+1
        U[0][col] = 1; U[3][col] = 1
        V[k1][col] = 1; V[10 + k2][col] = 1
        W[2 * k1][col] = 1; W[2 * k2 + 1][col] = 1
        col += 1
        
        # M2 = (P21+P22)*Q11
        # U: P21=(i=1,j=0) idx 2, P22=(i=1,j=1) idx 3
        # V: Q11=(j=0,k=k1) idx k1
        # W: R21=(k=k1,i=1) idx 2*k1+1
        U[2][col] = 1; U[3][col] = 1
        V[k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # M3 = P11*(Q12-Q22)
        # U: P11=(i=0,j=0) idx 0
        # V: Q12=(j=0,k=k2) idx k2, Q22=(j=1,k=k2) idx 10+k2
        # W: R12=(k=k2,i=0) idx 2*k2
        U[0][col] = 1
        V[k2][col] = 1; V[10 + k2][col] = -1
        W[2 * k2][col] = 1
        col += 1
        
        # M4 = P22*(Q21-Q11)
        # U: P22=(i=1,j=1) idx 3
        # V: Q21=(j=1,k=k1) idx 10+k1, Q11=(j=0,k=k1) idx k1
        # W: R11=(k=k1,i=0) idx 2*k1, R21=(k=k1,i=1) idx 2*k1+1
        U[3][col] = 1
        V[10 + k1][col] = 1; V[k1][col] = -1
        W[2 * k1][col] = 1; W[2 * k1 + 1][col] = 1
        col += 1
        
        # M5 = (P11+P12)*Q22
        # U: P11=(i=0,j=0) idx 0, P12=(i=0,j=1) idx 1
        # V: Q22=(j=1,k=k2) idx 10+k2
        # W: R11=(k=k1,i=0) idx 2*k1, R12=(k=k2,i=0) idx 2*k2
        U[0][col] = 1; U[1][col] = 1
        V[10 + k2][col] = 1
        W[2 * k1][col] = -1; W[2 * k2][col] = 1
        col += 1
        
        # M6 = (P21-P11)*(Q11+Q12)
        # U: P21=(i=1,j=0) idx 2, P11=(i=0,j=0) idx 0
        # V: Q11=(j=0,k=k1) idx k1, Q12=(j=0,k=k2) idx k2
        # W: R22=(k=k2,i=1) idx 2*k2+1
        U[2][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[k2][col] = 1
        W[2 * k2 + 1][col] = 1
        col += 1
        
        # M7 = (P12-P22)*(Q21+Q22)
        # U: P12=(i=0,j=1) idx 1, P22=(i=1,j=1) idx 3
        # V: Q21=(j=1,k=k1) idx 10+k1, Q22=(j=1,k=k2) idx 10+k2
        # W: R11=(k=k1,i=0) idx 2*k1, R22=(k=k2,i=1) idx 2*k2+1
        U[1][col] = 1; U[3][col] = -1
        V[10 + k1][col] = 1; V[10 + k2][col] = 1
        W[2 * k1][col] = 1; W[2 * k2 + 1][col] = 1
        col += 1
    
    # Schoolbook for 2x1x10 part
    # A2: rows 0,1 of A, column 2 (middle dim index 2) -> U indices: i*m+2 = i*3+2 for i in {0,1} -> 2, 5
    # B2: row 2 of B, columns 0..9 -> V indices: 2*10+k = 20+k for k in 0..9
    # C: rows 0,1, all 10 columns -> W indices: k*n+i = 2*k+i for k in 0..9, i in {0,1}
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0..9
            U[i * m + 2][col] = 1  # i*3+2
            V[2 * p + k][col] = 1  # 20+k
            W[k * n + i][col] = 1  # 2*k+i
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
