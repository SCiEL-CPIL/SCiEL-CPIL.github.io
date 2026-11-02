"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 37 decomposition:
    # Split m=3 into m1=2, m2=1.
    # Part 1: [2, 2, 10] using Strassen on 5 blocks of [2, 2, 2]. Rank 5*7=35.
    # Part 2: [2, 1, 10] using outer product. Rank 2.
    
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: [2, 2, 10]
    # A is 2x2 (rows 0,1 of n; cols 0,1 of m)
    # B is 2x10 (rows 0,1 of m; cols 0..9 of p)
    # Split B into 5 blocks of 2x2.
    for block_idx in range(5):
        k_start = block_idx * 2
        k1 = k_start
        k2 = k_start + 1
        
        # Strassen M1 = (A11 + A22)(B11 + B22)
        # U: A11 (0,0), A22 (1,1)
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        # V: B11 (0, k1), B22 (1, k2)
        V[0 * p + k1][col] += 1
        V[1 * p + k2][col] += 1
        # W: C11 (k1, 0) += 1, C22 (k2, 1) += 1
        W[k1 * n + 0][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # Strassen M2 = (A21 + A22)B11
        # U: A21 (1,0), A22 (1,1)
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        # V: B11 (0, k1)
        V[0 * p + k1][col] += 1
        # W: C21 (k1, 1) += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # Strassen M3 = A11(B21 - B22)
        # U: A11 (0,0)
        U[0 * m + 0][col] += 1
        # V: B21 (1, k1), B22 (1, k2)
        V[1 * p + k1][col] += 1
        V[1 * p + k2][col] -= 1
        # W: C12 (k2, 0) += 1
        W[k2 * n + 0][col] += 1
        col += 1
        
        # Strassen M4 = A22(B12 - B11)
        # U: A22 (1,1)
        U[1 * m + 1][col] += 1
        # V: B12 (0, k2), B11 (0, k1)
        V[0 * p + k2][col] += 1
        V[0 * p + k1][col] -= 1
        # W: C11 (k1, 0) += 1, C21 (k1, 1) += 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # Strassen M5 = (A11 + A12)B22
        # U: A11 (0,0), A12 (0,1)
        U[0 * m + 0][col] += 1
        U[0 * m + 1][col] += 1
        # V: B22 (1, k2)
        V[1 * p + k2][col] += 1
        # W: C11 (k1, 0) -= 1, C12 (k2, 0) += 1
        W[k1 * n + 0][col] -= 1
        W[k2 * n + 0][col] += 1
        col += 1
        
        # Strassen M6 = (A21 - A11)(B11 + B12)
        # U: A21 (1,0), A11 (0,0)
        U[1 * m + 0][col] += 1
        U[0 * m + 0][col] -= 1
        # V: B11 (0, k1), B12 (0, k2)
        V[0 * p + k1][col] += 1
        V[0 * p + k2][col] += 1
        # W: C22 (k2, 1) += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # Strassen M7 = (A12 - A22)(B21 + B22)
        # U: A12 (0,1), A22 (1,1)
        U[0 * m + 1][col] += 1
        U[1 * m + 1][col] -= 1
        # V: B21 (1, k1), B22 (1, k2)
        V[1 * p + k1][col] += 1
        V[1 * p + k2][col] += 1
        # W: C11 (k1, 0) += 1, C22 (k2, 1) += 1
        W[k1 * n + 0][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1

    # Part 2: [2, 1, 10]
    # A is 2x1 (rows 0,1 of n; col 2 of m)
    # B is 1x10 (row 2 of m; cols 0..9 of p)
    # C is 2x10
    # T[i, 2, k] = delta_{2,k}? No, j=2.
    # T[i, j, k] = delta_{jk}.
    # So T[i, 2, k] = 1 if k=2, else 0.
    # This means C[i, 2] = A[i, 2] * B[2, 2].
    # C[i, k] = 0 for k != 2.
    
    # Rank 2 decomposition for this part.
    # Term 1: i=0
    U[0 * m + 2][col] += 1
    V[2 * p + 2][col] += 1
    W[2 * n + 0][col] += 1
    col += 1
    
    # Term 2: i=1
    U[1 * m + 2][col] += 1
    V[2 * p + 2][col] += 1
    W[2 * n + 1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
