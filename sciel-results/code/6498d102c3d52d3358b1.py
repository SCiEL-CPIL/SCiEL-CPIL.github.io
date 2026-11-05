import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # Structure:
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x7) = [B1 (2x7); B2 (1x7)]
    # C = A1 B1 + A2 B2
    
    # Part 2: A2 B2 (2x1 x 7) -> Rank 14
    # Part 1: A1 B1 (2x2 x 7) -> Rank 25
    #   Part 1.2: A1 B12 (2x2 x 4) -> Rank 14 (Two Strassen 2x2x2)
    #   Part 1.1: A1 B11 (2x2 x 3) -> Rank 11 (One Strassen 2x2x2 + Schoolbook 2x2x1)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i0, i1: rows of A (0,1)
    # j0, j1: cols of A / rows of B (0,1)
    # k0, k1: cols of B (0,1)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        # W contribution to C00 and C11
        # C00 += M1, C11 += M1
        # W index is k*n+i
        # For M1, it contributes to C00 (k0,i0) and C11 (k1,i1)?
        # Wait, Strassen M1 = (A00+A11)(B00+B11).
        # This product is a scalar.
        # It contributes to C00 and C11.
        # C00 index: k0, i0.
        # C11 index: k1, i1.
        # So W[k0*n+i0] += 1, W[k1*n+i1] += 1.
        col += 1
        
        # M2 = (A10+A11)B00
        U[i1*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+i1][col] += 1
        # C10 += M2. Index k0, i1.
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0*m+j0][col] += 1
        V[j0*p+k1][col] += 1
        V[j1*p+k1][col] += -1
        W[k1*n+i0][col] += 1
        # C01 += M3. Index k1, i0.
        col += 1
        
        # M4 = A11(B10-B00)
        U[i1*m+j1][col] += 1
        V[j1*p+k0][col] += 1
        V[j0*p+k0][col] += -1
        W[k0*n+i1][col] += 1
        # C10 += M4. Index k0, i1.
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m+j0][col] += 1
        U[i0*m+j1][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i0][col] += -1 # C00 -= M5
        W[k1*n+i1][col] += 1   # C11 += M5? No.
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # M5 contributes to C00 (-1) and C01 (+1).
        # C00 index: k0, i0.
        # C01 index: k1, i0.
        W[k0*n+i0][col] += -1
        W[k1*n+i0][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[i1*m+j0][col] += 1
        U[i0*m+j0][col] += -1
        V[j0*p+k0][col] += 1
        V[j0*p+k1][col] += 1
        W[k1*n+i1][col] += 1
        # C11 += M6. Index k1, i1.
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        # C00 += M7. Index k0, i0.
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    # i0, i1: rows of A
    # j0, j1: cols of A / rows of B
    # k0: col of B
    def add_schoolbook_2x2x1(i0, i1, j0, j1, k0):
        nonlocal col
        # 4 terms
        for i in [i0, i1]:
            for j in [j0, j1]:
                U[i*m+j][col] += 1
                V[j*p+k0][col] += 1
                W[k0*n+i][col] += 1
                col += 1

    # Helper to add Schoolbook 2x1x7 block
    # i0, i1: rows of A
    # j0: col of A / row of B
    # k_start, k_end: cols of B
    def add_schoolbook_2x1x7(i0, i1, j0, k_start, k_end):
        nonlocal col
        for i in [i0, i1]:
            for k in range(k_start, k_end+1):
                U[i*m+j0][col] += 1
                V[j0*p+k][col] += 1
                W[k*n+i][col] += 1
                col += 1

    # Part 2: A2 B2
    # A2 is rows 0,1, col 2 of A.
    # B2 is row 2 of B, cols 0..6.
    add_schoolbook_2x1x7(0, 1, 2, 0, 6)
    
    # Part 1.2: A1 B12
    # A1 is rows 0,1, cols 0,1 of A.
    # B12 is rows 0,1 of B, cols 3..6.
    # Split into two 2x2x2 blocks.
    # Block 1: cols 3,4
    add_strassen(0, 1, 0, 1, 3, 4)
    # Block 2: cols 5,6
    add_strassen(0, 1, 0, 1, 5, 6)
    
    # Part 1.1: A1 B11
    # A1 is rows 0,1, cols 0,1 of A.
    # B11 is rows 0,1 of B, cols 0..2.
    # Split into 2x2x2 (cols 0,1) and 2x2x1 (col 2).
    # Block 1: cols 0,1
    add_strassen(0, 1, 0, 1, 0, 1)
    # Block 2: col 2
    add_schoolbook_2x2x1(0, 1, 0, 1, 2)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Verification
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check rank
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Check decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Should be 1 for all i,j,k
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    errors += 1
                    if errors < 5:
                        print(f"Error at {i},{j},{k}: {val}")
    print(f"Errors: {errors}")
