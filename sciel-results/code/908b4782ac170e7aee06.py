import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using block Strassen.
    # Rank is expected to be 50.
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for 2x2 by 2x2
    # A is 2x2 at (i0, j0), B is 2x2 at (j0, k0), C is 2x2 at (i0, k0)
    # Indices in U: i0..i0+1, j0..j0+1
    # Indices in V: j0..j0+1, k0..k0+1
    # Indices in W: k0..k0+1, i0..i0+1 (Note W index is k*n+i)
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        # (A11+A22)(B11+B22)
        # U: A11, A22
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        # V: B11, B22
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C11, C22? No, W contributes to all C entries.
        # C11 += M1, C22 += M1
        # W index for C11 is k0*n + i0
        # W index for C22 is (k0+1)*n + (i0+1)
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2
        # (A21+A22)B11
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        # C21 += M2
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M3
        # A11(B12-B22)
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        # C12 += M3
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4
        # A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        # C11 += M4, C21 += M4
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M5
        # (A11+A12)B22
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # C11 -= M5, C12 += M5
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6
        # (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        # C22 += M6
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M7
        # (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # C11 += M7, C22 -= M7? 
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # My M7 assignment: C11 += M7.
        W[k0*n + i0][col] = 1
        # Does M7 contribute to C22? No.
        col += 1

    # We need to handle 2x2 by 2x4 and 2x2 by 2x3 cases.
    # 2x2 by 2x4: Split into two 2x2 by 2x2.
    def add_2x2_by_2x4(i0, j0, k0):
        # B is 2x4 at (j0, k0)
        # Split B into (j0, k0) 2x2 and (j0, k0+2) 2x2
        add_strassen(i0, j0, k0)
        add_strassen(i0, j0, k0+2)

    # 2x2 by 2x3: Split into 2x2 by 2x2 and 2x2 by 2x1.
    def add_2x2_by_2x1(i0, j0, k0):
        nonlocal col
        # A is 2x2, B is 2x1.
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # 4 multiplications.
        # M1: A11 B1
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        # M2: A12 B2
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        # M3: A21 B1
        U[(i0+1)*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        # M4: A22 B2
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1

    def add_2x2_by_2x3(i0, j0, k0):
        # B is 2x3 at (j0, k0)
        # Split into 2x2 at (j0, k0) and 2x1 at (j0, k0+2)
        add_strassen(i0, j0, k0)
        add_2x2_by_2x1(i0, j0, k0+2)

    # Main construction
    # A is 2x4. Split into A_L (2x2) at (0,0) and A_R (2x2) at (0,2).
    # B is 4x7.
    # Split B into B_L (4x4) and B_R (4x3).
    # B_L is rows 0-3, cols 0-3.
    # B_R is rows 0-3, cols 4-6.
    
    # C = A_L B_L + A_R B_R
    
    # Process A_L B_L
    # A_L is 2x2 at (0,0).
    # B_L is 4x4. Split into B_LT (2x4) at (0,0) and B_LB (2x4) at (2,0).
    # A_L B_LT: A_L (2x2) by B_LT (2x4).
    add_2x2_by_2x4(0, 0, 0)
    # A_L B_LB: A_L (2x2) by B_LB (2x4).
    add_2x2_by_2x4(0, 2, 0)
    
    # Process A_R B_R
    # A_R is 2x2 at (0,2).
    # B_R is 4x3. Split into B_RT (2x3) at (0,4) and B_RB (2x3) at (2,4).
    # A_R B_RT: A_R (2x2) by B_RT (2x3).
    add_2x2_by_2x3(0, 2, 4)
    # A_R B_RB: A_R (2x2) by B_RB (2x3).
    add_2x2_by_2x3(0, 4, 4)
    
    # Check column count
    # 2x2 by 2x4: 14 cols.
    # 2x2 by 2x3: 11 cols.
    # Total: 14 + 14 + 11 + 11 = 50.
    
    if col != rank:
        # Fallback to schoolbook if construction failed
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
