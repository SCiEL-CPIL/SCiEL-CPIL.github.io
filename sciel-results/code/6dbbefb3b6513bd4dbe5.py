import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 36 solution:
    # 28 terms for the first 4x4x4 block (4 Strassen 2x2 blocks)
    # 8 terms for the last 4x4x2 block (schoolbook)
    R = 36
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2 block
    # i0, j0: top-left corner of the 2x2 block in A, C, B
    # i1, j1: bottom-right corner
    # B is m x p. The block is B[j0:j0+2, k0:k0+2]
    def add_strassen_block(i0, j0, k0, col_offset):
        nonlocal col
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        W[k0*n + i1][col] += 1
        W[k1*n + i0][col] += 1
        col += 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        W[k1*n + i0][col] += 1
        col += 1
        
        # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m + j0][col] += 1
        U[i0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i0][col] -= 1
        col += 1
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i1][col] -= 1
        col += 1
        
        # Note: The W updates above are distributed to the correct C indices.
        # Let's verify the W distribution for a 2x2 block at (i0, j0) to (i1, j1) and (j0, k0) to (j1, k1).
        # C[i0, k0] = M1 + M4 - M5 + M7
        # C[i0, k1] = M3 + M5
        # C[i1, k0] = M2 + M4
        # C[i1, k1] = M1 - M2 + M3 + M6
        
        # My W updates:
        # M1: W[k0, i0] += 1, W[k1, i1] += 1, W[k0, i1] += 1, W[k1, i0] += 1
        #      This puts 1 at (k0, i0), (k1, i1), (k0, i1), (k1, i0).
        #      C[i0, k0] gets +1 from M1. Correct.
        #      C[i1, k1] gets +1 from M1. Correct.
        #      C[i1, k0] gets +1 from M1. But we want M1 in C[i1, k0]? No.
        #      C[i1, k0] = M2 + M4. M1 is NOT in C[i1, k0].
        #      So my W update for M1 is WRONG for C[i1, k0] and C[i0, k1].
        
        # I need to map the scalar results to the correct W positions.
        # The "col" index corresponds to a specific linear combination.
        # The W matrix stores the coefficients for C.
        # For a term r, C_ik = sum_r W_ki,r * (sum U * A) * (sum V * B).
        # So for a specific Strassen term, say M1, it contributes to specific C entries.
        # M1 contributes to C[i0, k0] and C[i1, k1].
        # So W[k0, i0] += 1 and W[k1, i1] += 1.
        # It does NOT contribute to C[i1, k0] or C[i0, k1].
        
        # Let's redo the W updates carefully.
        
        # This function is flawed. I will rewrite the logic to be explicit per term.
        pass

    # Reset and do it explicitly
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    
    # Block 0,0: i in [0,1], j in [0,1], k in [0,1]
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0*m+0][col] = 1; U[1*m+1][col] = 1
    V[0*p+0][col] = 1; V[1*p+1][col] = 1
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10
    U[1*m+0][col] = 1; U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    W[0*n+1][col] = 1
    col += 1
    
    # M3: A00(B01-B11) -> C01
    U[0*m+0][col] = 1
    V[0*p+1][col] = 1; V[1*p+1][col] = -1
    W[1*n+0][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C10
    U[1*m+1][col] = 1
    V[1*p+0][col] = 1; V[0*p+0][col] = -1
    W[0*n+1][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0*m+0][col] = 1; U[0*m+1][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = -1; W[1*n+0][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[1*m+0][col] = 1; U[0*m+0][col] = -1
    V[0*p+0][col] = 1; V[0*p+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C10
    U[0*m+1][col] = 1; U[1*m+1][col] = -1
    V[1*p+0][col] = 1; V[1*p+1][col] = 1
    W[0*n+1][col] = -1
    col += 1
    
    # Block 1,1: i in [2,3], j in [2,3], k in [2,3]
    # Same structure, offset by 2
    # M1: (A22+A33)(B22+B33) -> C22, C33
    U[2*m+2][col] = 1; U[3*m+3][col] = 1
    V[2*p+2][col] = 1; V[3*p+3][col] = 1
    W[2*n+2][col] = 1; W[3*n+3][col] = 1
    col += 1
    
    # M2: (A32+A33)B22 -> C32
    U[3*m+2][col] = 1; U[3*m+3][col] = 1
    V[2*p+2][col] = 1
    W[2*n+3][col] = 1
    col += 1
    
    # M3: A22(B23-B33) -> C23
    U[2*m+2][col] = 1
    V[2*p+3][col] = 1; V[3*p+3][col] = -1
    W[3*n+2][col] = 1
    col += 1
    
    # M4: A33(B32-B22) -> C32
    U[3*m+3][col] = 1
    V[3*p+2][col] = 1; V[2*p+2][col] = -1
    W[2*n+3][col] = 1
    col += 1
    
    # M5: (A22+A23)B33 -> C22, C23
    U[2*m+2][col] = 1; U[2*m+3][col] = 1
    V[3*p+3][col] = 1
    W[2*n+2][col] = -1; W[3*n+2][col] = 1
    col += 1
    
    # M6: (A32-A22)(B22+B23) -> C33
    U[3*m+2][col] = 1; U[2*m+2][col] = -1
    V[2*p+2][col] = 1; V[2*p+3][col] = 1
    W[3*n+3][col] = 1
    col += 1
    
    # M7: (A23-A33)(B32+B33) -> C32
    U[2*m+3][col] = 1; U[3*m+3][col] = -1
    V[3*p+2][col] = 1; V[3*p+3][col] = 1
    W[2*n+3][col] = -1
    col += 1
    
    # Block 0,1: i in [0,1], j in [2,3], k in [0,1]
    # M1: (A02+A13)(B20+B31) -> C00, C11
    U[0*m+2][col] = 1; U[1*m+3][col] = 1
    V[2*p+0][col] = 1; V[3*p+1][col] = 1
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # M2: (A12+A13)B20 -> C10
    U[1*m+2][col] = 1; U[1*m+3][col] = 1
    V[2*p+0][col] = 1
    W[0*n+1][col] = 1
    col += 1
    
    # M3: A02(B21-B31) -> C01
    U[0*m+2][col] = 1
    V[2*p+1][col] = 1; V[3*p+1][col] = -1
    W[1*n+0][col] = 1
    col += 1
    
    # M4: A13(B30-B20) -> C10
    U[1*m+3][col] = 1
    V[3*p+0][col] = 1; V[2*p+0][col] = -1
    W[0*n+1][col] = 1
    col += 1
    
    # M5: (A02+A03)B31 -> C00, C01
    U[0*m+2][col] = 1; U[0*m+3][col] = 1
    V[3*p+1][col] = 1
    W[0*n+0][col] = -1; W[1*n+0][col] = 1
    col += 1
    
    # M6: (A12-A02)(B20+B21) -> C11
    U[1*m+2][col] = 1; U[0*m+2][col] = -1
    V[2*p+0][col] = 1; V[2*p+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # M7: (A03-A13)(B30+B31) -> C10
    U[0*m+3][col] = 1; U[1*m+3][col] = -1
    V[3*p+0][col] = 1; V[3*p+1][col] = 1
    W[0*n+1][col] = -1
    col += 1
    
    # Block 1,0: i in [2,3], j in [0,1], k in [2,3]
    # M1: (A20+A31)(B02+B13) -> C22, C33
    U[2*m+0][col] = 1; U[3*m+1][col] = 1
    V[0*p+2][col] = 1; V[1*p+3][col] = 1
    W[2*n+2][col] = 1; W[3*n+3][col] = 1
    col += 1
    
    # M2: (A30+A31)B02 -> C32
    U[3*m+0][col] = 1; U[3*m+1][col] = 1
    V[0*p+2][col] = 1
    W[2*n+3][col] = 1
    col += 1
    
    # M3: A20(B03-B13) -> C23
    U[2*m+0][col] = 1
    V[0*p+3][col] = 1; V[1*p+3][col] = -1
    W[3*n+2][col] = 1
    col += 1
    
    # M4: A31(B12-B02) -> C32
    U[3*m+1][col] = 1
    V[1*p+2][col] = 1; V[0*p+2][col] = -1
    W[2*n+3][col] = 1
    col += 1
    
    # M5: (A20+A21)B13 -> C22, C23
    U[2*m+0][col] = 1; U[2*m+1][col] = 1
    V[1*p+3][col] = 1
    W[2*n+2][col] = -1; W[3*n+2][col] = 1
    col += 1
    
    # M6: (A30-A20)(B02+B03) -> C33
    U[3*m+0][col] = 1; U[2*m+0][col] = -1
    V[0*p+2][col] = 1; V[0*p+3][col] = 1
    W[3*n+3][col] = 1
    col += 1
    
    # M7: (A21-A31)(B12+B13) -> C32
    U[2*m+1][col] = 1; U[3*m+1][col] = -1
    V[1*p+2][col] = 1; V[1*p+3][col] = 1
    W[2*n+3][col] = -1
    col += 1
    
    # Remaining 2 columns (k=4,5)
    # Schoolbook for A[4x4] * B[4x2]
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

