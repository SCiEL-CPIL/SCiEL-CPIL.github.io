"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-33 exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix.
    
    Strategy:
    Split the 6 columns of the second matrix into 3 blocks of 2 columns.
    For each block, the multiplication is 2x3 by 3x2.
    Split the middle dimension (3) into 2 and 1.
    - The 2x2 by 2x2 part is handled by Strassen's algorithm (rank 7).
    - The 2x1 by 1x2 part is handled by schoolbook (rank 4).
    Total rank per block = 11.
    Total rank = 3 * 11 = 33.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Total rank
    rank = 33
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Process 3 blocks of 2 output columns
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Part 1: Schoolbook for j=2 (2x1 by 1x2) ---
        # A[i, 2] * B[2, k] for i in {0,1}, k in {k0, k1}
        for i in range(2):
            for k in [k0, k1]:
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # --- Part 2: Strassen for j in {0,1} (2x2 by 2x2) ---
        # Local indices: i, j in {0,1}. k_local in {0,1} maps to k0, k1.
        
        # Term 1: M1 = (A00 + A01)(B00 + B10)
        # U: A00=1, A01=1
        # V: B00=1 (j=0, k0), B10=1 (j=1, k0)
        # W: C01=1, C11=1 (from alpha: C01=M1, C11=M1)
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 2: M2 = A10(B00 + B10)
        # U: A10=1
        # V: B00=1 (j=0, k0), B10=1 (j=1, k0)
        # W: C10=1, C11=-1
        U[1 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # Term 3: M3 = A00(B10 - B11)
        # U: A00=1
        # V: B10=1 (j=1, k0), B11=-1 (j=1, k1)
        # W: C00=1, C01=-1, C11=1
        U[0 * m + 0][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = -1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = -1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 4: M4 = A11(B01 - B00)
        # U: A11=1
        # V: B01=1 (j=0, k1), B00=-1 (j=0, k0)
        # W: C00=-1, C10=1
        U[1 * m + 1][col] = 1
        V[0 * p + k1][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = -1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 5: M5 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1 (j=0, k0), B11=1 (j=1, k1)
        # W: C00=1, C01=-1, C11=1
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = -1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 6: M6 = (A01 + A11)(B10 - B00)
        # U: A01=1, A11=1
        # V: B10=1 (j=1, k0), B00=-1 (j=0, k0)
        # W: C01=1, C10=-1
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 1][col] = -1
        col += 1
        
        # Term 7: M7 = (A00 + A10)(B01 - B11)
        # U: A00=1, A10=1
        # V: B01=1 (j=0, k1), B11=-1 (j=1, k1)
        # W: C00=1, C10=1
        U[0 * m + 0][col] = 1
        U[1 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
