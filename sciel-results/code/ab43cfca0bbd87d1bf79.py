"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x6.
    # Split the 2x6 matrix into three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6].
    # Apply Strassen to each 2x2x2 block: 7 multiplications each -> 21 total.
    #
    # Strassen for C = A * B where A, B are 2x2:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # U has 4 rows (i*m+j for i in {0,1}, j in {0,1}), 21 columns.
    # V has 12 rows (j*p+k for j in {0,1}, k in {0..5}), 21 columns.
    # W has 12 rows (k*n+i for k in {0..5}, i in {0,1}), 21 columns.
    #
    # Convention: W[k*n + i, r]
    
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows
    
    # Strassen coefficients:
    # For each 2x2x2 block b (b=0,1,2), columns 7*b .. 7*b+6
    # A indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B indices for block b: B[0,0]=V[0+2*b], B[0,1]=V[1+2*b], B[1,0]=V[2+2*b], B[1,1]=V[3+2*b]
    #   i.e., V[j*p + k] where j in {0,1}, k in {2b, 2b+1}
    # C output for block b: C[0,0]=W[0*2+0]=W[0], C[0,1]=W[1*2+0]=W[2]... wait
    # W[k*n + i]: for block b, k in {2b, 2b+1}, i in {0,1}
    #   C[0,0] -> k=2b, i=0 -> W[2b*2 + 0] = W[4b]
    #   C[0,1] -> k=2b+1, i=0 -> W[(2b+1)*2 + 0] = W[4b+2]
    #   C[1,0] -> k=2b, i=1 -> W[4b+1]
    #   C[1,1] -> k=2b+1, i=1 -> W[4b+3]
    
    # Strassen mult factors:
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U coefficients for each (i,j) row across 7 Strassen factors per block:
    # Row 0 (i=0,j=0): A11 -> M1:1, M3:1, M5:1
    # Row 1 (i=0,j=1): A12 -> M5:1, M7:1
    # Row 2 (i=1,j=0): A21 -> M2:1, M6:1
    # Row 3 (i=1,j=1): A22 -> M1:1, M2:1, M4:1, M7:-1
    
    # V coefficients for each (j,k) row:
    # For block b, B11=V[2b], B12=V[2b+1], B21=V[2b+2], B22=V[2b+3]
    # M1: B11+B22 -> V[2b]:1, V[2b+3]:1
    # M2: B11 -> V[2b]:1
    # M3: B12-B22 -> V[2b+1]:1, V[2b+3]:-1
    # M4: B21-B11 -> V[2b+2]:1, V[2b]:-1
    # M5: B22 -> V[2b+3]:1
    # M6: B11+B12 -> V[2b]:1, V[2b+1]:1
    # M7: B21+B22 -> V[2b+2]:1, V[2b+3]:1
    
    # W coefficients for each (k,i) row:
    # For block b: C11=W[4b], C12=W[4b+2], C21=W[4b+1], C22=W[4b+3]
    # C11 = M1 + M4 - M5 + M7 -> W[4b]: M1:1, M4:1, M5:-1, M7:1
    # C12 = M3 + M5 -> W[4b+2]: M3:1, M5:1
    # C21 = M2 + M4 -> W[4b+1]: M2:1, M4:1
    # C22 = M1 - M2 + M3 + M6 -> W[4b+3]: M1:1, M2:-1, M3:1, M6:1
    
    for b in range(3):
        base = 7 * b
        # U: same for all blocks
        # Row 0 (A11): M1, M3, M5
        U[0][base + 0] = 1   # M1
        U[0][base + 2] = 1   # M3
        U[0][base + 4] = 1   # M5
        # Row 1 (A12): M5, M7
        U[1][base + 4] = 1   # M5
        U[1][base + 6] = 1   # M7
        # Row 2 (A21): M2, M6
        U[2][base + 1] = 1   # M2
        U[2][base + 5] = 1   # M6
        # Row 3 (A22): M1, M2, M4, M7
        U[3][base + 0] = 1   # M1
        U[3][base + 1] = 1   # M2
        U[3][base + 3] = 1   # M4
        U[3][base + 6] = -1  # M7
        
        # V:
        # V[2b] = B11: M1:1, M2:1, M4:-1, M6:1
        V[2*b + 0][base + 0] = 1   # M1
        V[2*b + 0][base + 1] = 1   # M2
        V[2*b + 0][base + 3] = -1  # M4
        V[2*b + 0][base + 5] = 1   # M6
        # V[2b+1] = B12: M3:1, M6:1
        V[2*b + 1][base + 2] = 1   # M3
        V[2*b + 1][base + 5] = 1   # M6
        # V[2b+2] = B21: M4:1, M7:1
        V[2*b + 2][base + 3] = 1   # M4
        V[2*b + 2][base + 6] = 1   # M7
        # V[2b+3] = B22: M1:1, M3:-1, M5:1, M7:1
        V[2*b + 3][base + 0] = 1   # M1
        V[2*b + 3][base + 2] = -1  # M3
        V[2*b + 3][base + 4] = 1   # M5
        V[2*b + 3][base + 6] = 1   # M7
        
        # W:
        # W[4b] = C11: M1:1, M4:1, M5:-1, M7:1
        W[4*b + 0][base + 0] = 1   # M1
        W[4*b + 0][base + 3] = 1   # M4
        W[4*b + 0][base + 4] = -1  # M5
        W[4*b + 0][base + 6] = 1   # M7
        # W[4b+1] = C21: M2:1, M4:1
        W[4*b + 1][base + 1] = 1   # M2
        W[4*b + 1][base + 3] = 1   # M4
        # W[4b+2] = C12: M3:1, M5:1
        W[4*b + 2][base + 2] = 1   # M3
        W[4*b + 2][base + 4] = 1   # M5
        # W[4b+3] = C22: M1:1, M2:-1, M3:1, M6:1
        W[4*b + 3][base + 0] = 1   # M1
        W[4*b + 3][base + 1] = -1  # M2
        W[4*b + 3][base + 2] = 1   # M3
        W[4*b + 3][base + 5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
