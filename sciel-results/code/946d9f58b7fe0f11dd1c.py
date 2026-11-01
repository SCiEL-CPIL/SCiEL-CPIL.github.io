"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm on 8 blocks of 2x2 matrix multiplication
    # A is 2x2, B is 2x16. Split B into 8 blocks of 2x2.
    # Each block requires 7 multiplications.
    # Total rank = 8 * 7 = 56.
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2):
    # U index: i * m + j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Indices for B (2x16):
    # V index: j * p + k
    # For block r, k=2r, 2r+1
    # j=0, k=2r -> 2r
    # j=0, k=2r+1 -> 2r+1
    # j=1, k=2r -> 16 + 2r
    # j=1, k=2r+1 -> 16 + 2r + 1
    
    # Indices for C (16x2 output, but stored as 2x16? No, W is p*n = 16*2 = 32)
    # W index: k * n + i
    # For block r, C columns are 2r, 2r+1.
    # C_{0, 2r} -> k=2r, i=0 -> 2*(2r) + 0 = 4r
    # C_{1, 2r} -> k=2r, i=1 -> 2*(2r) + 1 = 4r + 1
    # C_{0, 2r+1} -> k=2r+1, i=0 -> 2*(2r+1) + 0 = 4r + 2
    # C_{1, 2r+1} -> k=2r+1, i=1 -> 2*(2r+1) + 1 = 4r + 3
    
    for r in range(8):
        base_col = 7 * r
        
        # V indices for this block
        v00 = 2 * r
        v01 = 2 * r + 1
        v10 = 16 + 2 * r
        v11 = 16 + 2 * r + 1
        
        # W indices for this block
        w00 = 4 * r
        w10 = 4 * r + 1
        w01 = 4 * r + 2
        w11 = 4 * r + 3
        
        # M1 = (A00 + A11)(B00 + B11)
        col = base_col
        U[0][col] += 1
        U[3][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        
        # M2 = (A10 + A11)B00
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        
        # M3 = A00(B01 - B11)
        col = base_col + 2
        U[0][col] += 1
        V[v01][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        W[w10][col] += 1
        
        # M4 = A11(B10 - B00)
        col = base_col + 3
        U[3][col] += 1
        V[v10][col] += 1
        V[v00][col] -= 1
        W[w00][col] -= 1
        W[w01][col] += 1
        
        # M5 = (A00 + A01)B11
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        col = base_col + 5
        U[2][col] += 1
        U[0][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        col = base_col + 6
        U[1][col] += 1
        U[3][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
