import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 57 decomposition
    # Part 1: 2x2x6 using 3 Strassen blocks (rank 21)
    # Part 2: 2x3x6 using 2 schoolbook blocks (rank 36)
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B columns into 3 pairs: (0,1), (2,3), (4,5)
    for c in range(3):
        # B rows 0,1. B cols 2c, 2c+1
        # Strassen for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the terms manually
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1) -> mapped to global (0, 2c), (0, 2c+1), (1, 2c), (1, 2c+1)
        
        # Term 1: M1
        # A: 00, 11 (coeff 1)
        # B: 00, 11 (coeff 1)
        # C: 00 (1), 11 (1)
        U[0*2+0][col] += 1 # A00
        U[1*2+1][col] += 1 # A11
        V[0*p+2*c][col] += 1 # B00
        V[1*p+2*c+1][col] += 1 # B11
        W[2*c*n+0][col] += 1 # C00
        W[(2*c+1)*n+1][col] += 1 # C11
        col += 1
        
        # Term 2: M2
        # A: 10, 11 (coeff 1)
        # B: 00 (coeff 1)
        # C: 10 (1), 11 (-1)
        U[1*2+0][col] += 1 # A10
        U[1*2+1][col] += 1 # A11
        V[0*p+2*c][col] += 1 # B00
        W[2*c*n+1][col] += 1 # C10
        W[(2*c+1)*n+1][col] += -1 # C11
        col += 1
        
        # Term 3: M3
        # A: 00 (coeff 1)
        # B: 10 (1), 11 (-1)
        # C: 01 (1), 11 (1)
        U[0*2+0][col] += 1 # A00
        V[1*p+2*c][col] += 1 # B10
        V[1*p+2*c+1][col] += -1 # B11
        W[(2*c+1)*n+0][col] += 1 # C01
        W[(2*c+1)*n+1][col] += 1 # C11
        col += 1
        
        # Term 4: M4
        # A: 11 (coeff 1)
        # B: 01 (1), 00 (-1)
        # C: 00 (1), 10 (1)
        U[1*2+1][col] += 1 # A11
        V[0*p+2*c+1][col] += 1 # B01
        V[0*p+2*c][col] += -1 # B00
        W[2*c*n+0][col] += 1 # C00
        W[2*c*n+1][col] += 1 # C10
        col += 1
        
        # Term 5: M5
        # A: 00 (1), 10 (1)
        # B: 11 (1)
        # C: 00 (-1), 01 (1)
        U[0*2+0][col] += 1 # A00
        U[1*2+0][col] += 1 # A10
        V[1*p+2*c+1][col] += 1 # B11
        W[2*c*n+0][col] += -1 # C00
        W[(2*c+1)*n+0][col] += 1 # C01
        col += 1
        
        # Term 6: M6
        # A: 11 (1), 00 (-1)
        # B: 00 (1), 10 (1)
        # C: 11 (1)
        U[1*2+1][col] += 1 # A11
        U[0*2+0][col] += -1 # A00
        V[0*p+2*c][col] += 1 # B00
        V[1*p+2*c][col] += 1 # B10
        W[(2*c+1)*n+1][col] += 1 # C11
        col += 1
        
        # Term 7: M7
        # A: 01 (1), 11 (-1)
        # B: 11 (1), 01 (1)
        # C: 00 (1)
        U[0*2+1][col] += 1 # A01
        U[1*2+1][col] += -1 # A11
        V[1*p+2*c+1][col] += 1 # B11
        V[0*p+2*c+1][col] += 1 # B01
        W[2*c*n+0][col] += 1 # C00
        col += 1

    # Part 2: A[0:2, 2:5] * B[2:5, 0:6]
    # Schoolbook for 2x3x6
    # A cols 2,3,4. B rows 2,3,4.
    for i in range(2):
        for j in range(3):
            a_j = j + 2 # global col index in A
            for k in range(6):
                b_k = k # global row index in B? No, B rows are 2,3,4.
                # Wait, B is 5x6. Rows 0..4.
                # In Part 2, we use B rows 2,3,4.
                # So local row 0,1,2 corresponds to global row 2,3,4.
                # The loop above for Part 1 used B rows 0,1.
                # So for Part 2, B rows are 2,3,4.
                # j_local is 0,1,2. j_global = j_local + 2.
                # k is 0..5.
                
                # Term: A[i, j_global] * B[j_global, k] -> C[i, k]
                U[i*m + a_j][col] = 1
                V[a_j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
