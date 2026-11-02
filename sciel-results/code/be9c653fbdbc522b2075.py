"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 43: Strassen for k=0,1 (7 terms each) + trivial for k=2..10 (9 terms each)
    rank = 7 * 2 + 9 * (p - 2)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen for k=0 and k=1
    for k in range(2):
        # Term 1: (A11+A22)(B11+B22) -> C11 += , C22 +=
        U[0][col] += 1.0
        U[3][col] += 1.0
        V[k][col] += 1.0
        V[11 + k][col] += 1.0
        W[k * n + 0][col] += 1.0
        W[k * n + 1][col] += 1.0
        col += 1
        
        # Term 2: (A21+A22)B11 -> C21 +=
        U[2][col] += 1.0
        U[3][col] += 1.0
        V[k][col] += 1.0
        W[k * n + 1][col] += 1.0
        col += 1
        
        # Term 3: A11(B21-B11) -> C12 -=
        U[0][col] += 1.0
        V[11 + k][col] += 1.0
        V[k][col] -= 1.0
        W[k * n + 0][col] -= 1.0
        col += 1
        
        # Term 4: A22(B12-B22) -> C21 -=
        U[3][col] += 1.0
        V[10 + k][col] += 1.0
        V[11 + k][col] -= 1.0
        W[k * n + 1][col] -= 1.0
        col += 1
        
        # Term 5: (A11+A12)B22 -> C12 +=
        U[0][col] += 1.0
        U[1][col] += 1.0
        V[11 + k][col] += 1.0
        W[k * n + 0][col] += 1.0
        col += 1
        
        # Term 6: (A21-A11)(B11+B12) -> C21 +=
        U[2][col] += 1.0
        U[0][col] -= 1.0
        V[k][col] += 1.0
        V[10 + k][col] += 1.0
        W[k * n + 1][col] += 1.0
        col += 1
        
        # Term 7: (A12-A22)(B21+B22) -> C12 +=
        U[1][col] += 1.0
        U[3][col] -= 1.0
        V[11 + k][col] += 1.0
        V[10 + k][col] += 1.0
        W[k * n + 0][col] += 1.0
        col += 1
    
    # Trivial for k=2..10
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
