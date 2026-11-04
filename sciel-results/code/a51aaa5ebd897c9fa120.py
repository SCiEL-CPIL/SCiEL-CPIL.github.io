"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll use Strassen's algorithm for 2x2x2 (rank 7) to build up to 4x4x4 (rank 28)
    # Then handle 9 columns as 2 blocks of 4 + 1 column of 1
    # Total rank: 28 + 28 + 16 = 72
    
    rank = 72
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper: add Strassen 2x2x2 decomposition
    def add_strassen_2x2(U, V, W, a_off, b_off, c_off, col_start):
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_off + 0][c] = 1
        U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1   # C11
        W[c_off + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_off + 2][c] = 1
        U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1
        W[c_off + 2][c] = 1   # C21
        W[c_off + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_off + 0][c] = 1
        V[b_off + 2][c] = 1
        V[b_off + 3][c] = -1
        W[c_off + 1][c] = 1   # C12
        W[c_off + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_off + 3][c] = 1
        V[b_off + 1][c] = 1
        V[b_off + 0][c] = -1
        W[c_off + 0][c] = 1   # C11
        W[c_off + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_off + 0][c] = 1
        U[a_off + 1][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = -1  # C11
        W[c_off + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_off + 2][c] = 1
        U[a_off + 0][c] = -1
        V[b_off + 0][c] = 1
        V[b_off + 1][c] = 1
        W[c_off + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_off + 1][c] = 1
        U[a_off + 3][c] = -1
        V[b_off + 2][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1   # C11
        c += 1
        
        return c
    
    # Helper: add 4x4x4 Strassen-based decomposition (rank 28)
    def add_4x4x4_strassen(U, V, W, a_off, b_off, c_off, col_start):
        c = col_start
        
        # For 4x4 matrix A stored with row index = i*4 + j
        # A[i,j] is at U[a_off + i*4 + j]
        
        # Block structure:
        # A11 = A[0:2, 0:2] -> U indices: a_off+0, a_off+1, a_off+4, a_off+5
        # A12 = A[0:2, 2:4] -> U indices: a_off+2, a_off+3, a_off+6, a_off+7
        # A21 = A[2:4, 0:2] -> U indices: a_off+8, a_off+9, a_off+12, a_off+13
        # A22 = A[2:4, 2:4] -> U indices: a_off+10, a_off+11, a_off+14, a_off+15
        
        # Similarly for B (4x4) and C (4x4)
        
        # We need to compute C = A*B using 7 Strassen products, each being a 2x2 matrix product
        # Each 2x2 product uses schoolbook (4 scalar mults), so 7*4 = 28 total
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22: 
        #   (0,0): A11[0,0] + A22[0,0] = A[0,0] + A[2,2] -> U[a_off+0] + U[a_off+10]
        #   (0,1): A11[0,1] + A22[0,1] = A[0,1] + A[2,3] -> U[a_off+1] + U[a_off+11]
        #   (1,0): A11[1,0] + A22[1,0] = A[1,0] + A[3,2] -> U[a_off+4] + U[a_off+14]
        #   (1,1): A11[1,1] + A22[1,1] = A[1,1] + A[3,3] -> U[a_off+5] + U[a_off+15]
        
        # B11+B22:
        #   (0,0): B[0,0] + B[2,2] -> V[b_off+0] + V[b_off+10]
        #   (0,1): B[0,1] + B[2,3] -> V[b_off+1] + V[b_off+11]
        #   (1,0): B[1,0] + B[3,2] -> V[b_off+4] + V[b_off+14]
        #   (1,1): B[1,1] + B[3,3] -> V[b_off+5] + V[b_off+15]
        
        # For schoolbook 2x2x2, we need 4 scalar multiplications
        # Let's define the 4 products for M1:
        # P1_1 = (A11+A22)[0,0] * (B11+B22)[0,0]
        # P1_2 = (A11+A22)[0,1] * (B11+B22)[1,0]
        # P1_3 = (A11+A22)[1,0] * (B11+B22)[0,1]
        # P1_4 = (A11+A22)[1,1] * (B11+B22)[1,1]
        
        # Wait, this is getting complex. Let me use a cleaner approach.
        
        # Actually, let me just use schoolbook for 4x4x4 which is rank 16, but that's worse.
        # Let me try a different approach: use Strassen recursively.
        
        # For 4x4x4, partition into 2x2 blocks.
        # Use Strassen to get 7 products of 2x2 matrices.
        # Each 2x2x2 product can be done with Strassen (rank 7) or schoolbook (rank 4).
        
        # If we use schoolbook for each 2x2x2: 7 * 4 = 28
        # If we use Strassen for each 2x2x2: 7 * 7 = 49
        
        # Let's use schoolbook for the inner products: rank 28
        
        # For each of the 7 Strassen products, we compute a 2x2 matrix product using schoolbook.
        
        # Let me define a helper that adds a schoolbook 2x2x2 decomposition
        def add_schoolbook_2x2(U, V, W, a_off, b_off, c_off, col_start):
            c = col_start
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            
            # Product 1: A[0,0]*B[0,0]
            U[a_off + 0][c] = 1
            V[b_off + 0][c] = 1
            W[c_off + 0][c] = 1
            c += 1
            
            # Product 2: A[0,1]*B[1,0]
            U[a_off + 1][c] = 1
            V[b_off + 2][c] = 1
            W[c_off + 0][c] = 1
            c += 1
            
            # Product 3: A[0,0]*B[0,1]
            U[a_off + 0][c] = 1
            V[b_off + 1][c] = 1
            W[c_off + 1][c] = 1
            c += 1
            
            # Product 4: A[0,1]*B[1,1]
            U[a_off + 1][c] = 1
            V[b_off + 3][c] = 1
            W[c_off + 1][c] = 1
            c += 1
            
            # Product 5: A[1,0]*B[0,0]
            U[a_off + 2][c] = 1
            V[b_off + 0][c] = 1
            W[c_off + 2][c] = 1
            c += 1
            
            # Product 6: A[1,1]*B[1,0]
            U[a_off + 3][c] = 1
            V[b_off + 2][c] = 1
            W[c_off + 2][c] = 1
            c += 1
            
            # Product 7: A[1,0]*B[0,1]
            U[a_off + 2][c] = 1
            V[b_off + 1][c] = 1
            W[c_off + 3][c] = 1
            c += 1
            
            # Product 8: A[1,1]*B[1,1]
            U[a_off + 3][c] = 1
            V[b_off + 3][c] = 1
            W[c_off + 3][c] = 1
            c += 1
            
            return c
        
        # Now for each Strassen product, we need to compute a 2x2 matrix product.
        # The 7 Strassen products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each, we need to identify the 2x2 submatrices.
        
        # This is getting very complex. Let me just use schoolbook for the entire 4x4x4.
        # Rank = 16 for 4x4x4, so for 9 columns: 16*2 + 16 = 48? No, that's not right.
        
        # Actually, for 4x4x9, if we use schoolbook for each column: 4*4*9 = 144
        # If we use Strassen for 4x4x4 (rank 28) for 2 blocks + schoolbook for 1 column: 28+28+16 = 72
        
        # Let me just implement schoolbook for the whole thing to be safe.
        return c
    
    # Given the complexity, let me just use schoolbook for the entire 4x4x9 multiplication
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to lists for the return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
