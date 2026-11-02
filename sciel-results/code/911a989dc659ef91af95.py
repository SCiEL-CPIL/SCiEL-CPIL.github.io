"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50
    
    col = 0
    
    # Strassen blocks: 4 disjoint 2x2x2 sub-tensors
    # j-pair (0,1), k-pairs: (0,1), (2,3), (4,5), (6,7)
    strassen_k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for (k1, k2) in strassen_k_pairs:
        # Factor 1: M1 = (A11+A22)(B11+B22)
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[9 + k2][col] = 1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        W[2*k2][col] = 1; W[2*k2+1][col] = 1
        col += 1
        
        # Factor 2: M2 = (A21+A22)B11
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # Factor 3: M3 = A11(B12-B22)
        U[0][col] = 1
        V[k2][col] = 1; V[9+k2][col] = -1
        W[2*k2][col] = 1; W[2*k2+1][col] = 1
        col += 1
        
        # Factor 4: M4 = A22(B21-B11)
        U[4][col] = 1
        V[9+k1][col] = 1; V[k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # Factor 5: M5 = (A11+A12)B22
        U[0][col] = 1; U[1][col] = 1
        V[9+k2][col] = 1
        W[2*k2][col] = -1; W[2*k2+1][col] = 1
        col += 1
        
        # Factor 6: M6 = (A21-A11)(B11+B12)
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[k2][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # Factor 7: M7 = (A12-A22)(B21+B22)
        U[1][col] = 1; U[4][col] = -1
        V[9+k1][col] = 1; V[9+k2][col] = 1
        W[2*k1][col] = 1; W[2*k2][col] = 1
        col += 1
    
    # Schoolbook for j=2, all k, all i: 18 terms
    for i in range(n):
        for k in range(p):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # Schoolbook for j in {0,1}, k=8, all i: 4 terms
    for i in range(n):
        for j in [0, 1]:
            U[i*m+j][col] = 1
            V[j*p+8][col] = 1
            W[8*n+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
