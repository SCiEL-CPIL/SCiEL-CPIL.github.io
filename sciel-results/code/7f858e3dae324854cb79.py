import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = 2, 3, 10
    rank = 45  # 35 (Strassen for 2x2x10) + 10 (2x1x10)
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    # Part 1: j=2 (2x1x10 subproblem, rank 10)
    # Columns 0-9
    for k in range(p):
        col = k
        # U: A[i,2] for i=0,1 -> indices 2, 3
        U[2][col] = 1  # A[0,2]
        U[3][col] = 1  # A[1,2]
        # V: B[2,k] -> index 2*10 + k = 20+k
        V[20 + k][col] = 1
        # W: C[k,i] for i=0,1 -> indices k*2+0, k*2+1
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
    
    # Part 2: j=0,1 (2x2x10 subproblem, Strassen rank 7 per 2x2 block)
    # For each pair of k values (k1, k2) where k1 = 2t, k2 = 2t+1, t=0..4
    # We use Strassen's 7 products for 2x2 matrix multiplication
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] (indices 0,1,2,3 in U)
    # B_sub depends on k values
    
    # Strassen 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices for A: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    strassen_cols = []
    for t in range(5):  # t = 0..4, k1=2t, k2=2t+1
        k1 = 2 * t
        k2 = 2 * t + 1
        base_col = 10 + 7 * t  # columns 10-44
        
        # M1: (A00+A11)*(B[0,k1]+B[0,k2]+B[1,k1]+B[1,k2])
        col = base_col + 0
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[0 * p + k1][col] = 1   # B[0,k1]
        V[0 * p + k2][col] = 1   # B[0,k2]
        V[1 * p + k1][col] = 1   # B[1,k1]
        V[1 * p + k2][col] = 1   # B[1,k2]
        # W: C00 gets +M1, C11 gets +M1
        # C[k,i]: k from {k1,k2}, i from {0,1}
        # C00 = C[k1,0] or C[k1,1]? 
        # Wait - we need to be careful. The 2x2 block multiplication gives us
        # C_sub[i,j] for the 2x2 block, but here we're computing 2x2 * 2x2 -> 2x2
        # where the 2x2 B block has columns k1,k2 and rows 0,1
        # Result C[i, k1] and C[i, k2] for i=0,1
        
        # Actually, let me reconsider. For 2x2x2:
        # C[i,j] = sum over the middle index
        # Here middle index has 2 values (k1,k2), so C is 2x2
        # C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
        # C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2]
        # C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1]
        # C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2]
        
        # So the 2x2 output has entries at W indices:
        # C[0,k1] -> k1*n+0 = k1*2
        # C[0,k2] -> k2*n+0 = k2*2
        # C[1,k1] -> k1*n+1 = k1*2+1
        # C[1,k2] -> k2*n+1 = k2*2+1
        
        # Strassen:
        # C[0,k1] = M1 + M4 - M5 + M7
        # C[0,k2] = M3 + M5
        # C[1,k1] = M2 + M4
        # C[1,k2] = M1 - M2 + M3 + M6
        
        W[k1 * 2 + 0][col] += 1    # C[0,k1] += M1
        W[k2 * 2 + 1][col] += 1    # C[1,k2] += M1
        
        # M2: (A10+A11)*B[0,k1]
        col = base_col + 1
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[0 * p + k1][col] = 1   # B[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1] += M2
        W[k2 * 2 + 1][col] += -1 # C[1,k2] += -M2
        
        # M3: A[0,0]*(B[0,k2]-B[1,k2])
        col = base_col + 2
        U[0][col] = 1
        V[0 * p + k2][col] = 1   # B[0,k2]
        V[1 * p + k2][col] = -1  # -B[1,k2]
        W[k2 * 2 + 0][col] += 1  # C[0,k2] += M3
        W[k2 * 2 + 1][col] += 1  # C[1,k2] += M3
        
        # M4: A[1,1]*(B[1,k1]-B[0,k1])
        col = base_col + 3
        U[3][col] = 1
        V[1 * p + k1][col] = 1   # B[1,k1]
        V[0 * p + k1][col] = -1  # -B[0,k1]
        W[k1 * 2 + 0][col] += 1  # C[0,k1] += M4
        W[k1 * 2 + 1][col] += 1  # C[1,k1] += M4
        
        # M5: (A[0,0]+A[0,1])*B[1,k2]
        col = base_col + 4
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        V[1 * p + k2][col] = 1  # B[1,k2]
        W[k1 * 2 + 0][col] += -1 # C[0,k1] += -M5
        W[k2 * 2 + 0][col] += 1  # C[0,k2] += M5
        
        # M6: (A[1,0]-A[0,0])*(B[0,k1]+B[0,k2])
        col = base_col + 5
        U[2][col] = 1   # A[1,0]
        U[0][col] = -1  # -A[0,0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        V[0 * p + k2][col] = 1  # B[0,k2]
        W[k2 * 2 + 1][col] += 1 # C[1,k2] += M6
        
        # M7: (A[0,1]-A[1,1])*(B[1,k1]+B[1,k2])
        col = base_col + 6
        U[1][col] = 1   # A[0,1]
        U[3][col] = -1  # -A[1,1]
        V[1 * p + k1][col] = 1  # B[1,k1]
        V[1 * p + k2][col] = 1  # B[1,k2]
        W[k1 * 2 + 0][col] += 1 # C[0,k1] += M7
    
    return {"U": U, "V": V, "W": W}
