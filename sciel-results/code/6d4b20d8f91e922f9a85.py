"""Find a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Rank-55 construction: 5*7 (Strassen for 2x2x10) + 2*1*10 (for column 2 of A)
    # Strassen part handles A columns 0,1 and B rows 0,1 -> C = A[:,:2] @ B[:2,:]
    # Remaining: A[:,2] @ B[2,:] which is 2x1x10, rank 20
    
    strassen_rank = 5 * 7  # 35
    remaining_rank = 2 * 1 * 10  # 20
    rank = strassen_rank + remaining_rank  # 55
    
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 55)
    V = [[0] * rank for _ in range(m * p)]   # shape (30, 55)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 55)
    
    # Strassen part: for each pair of B-columns (k, k+1) where k in {0,2,4,6,8}
    # A submatrix: rows 0,1; columns 0,1
    # B submatrix: rows 0,1; columns k,k+1
    # Output: C rows 0,1; columns k,k+1
    
    # Strassen variables for each pair (a,b,c,d) from A and (e,f,g,h) from B:
    # A = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B = [[B[0,k], B[0,k+1]], [B[1,k], B[1,k+1]]]
    # So a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,k], f=B[0,k+1], g=B[1,k], h=B[1,k+1]
    
    # m1 = (a+b)*(e+f)
    # m2 = a*(g-h)
    # m3 = (c+d)*e
    # m4 = d*(h-g)
    # m5 = (a+c)*(f+h)
    # m6 = (b-d)*(g+h)
    # m7 = (a-c)*(e+g)
    
    # C[0,k]   = m1 + m2 + m4 - m7
    # C[1,k]   = m3 + m7 - m4 - m6
    # C[0,k+1] = m1 - m3 - m5 + m7
    # C[1,k+1] = m2 + m5 + m6 - m1
    
    for pair_idx in range(5):
        k0 = 2 * pair_idx  # first column index in B for this pair
        k1 = k0 + 1        # second column index in B for this pair
        base_col = pair_idx * 7
        
        # m1 = (A[0,0]+A[0,1]) * (B[0,k0]+B[0,k1])
        # U entries: A[0,0] at U[0*3+0], A[0,1] at U[0*3+1]
        # V entries: B[0,k0] at V[0*10+k0], B[0,k1] at V[0*10+k1]
        # W entries: C[0,k0] at W[k0*2+0], C[0,k1] at W[k1*2+0], C[1,k0] at W[k0*2+1], C[1,k1] at W[k1*2+1]
        
        # m1 contributes to C[0,k0] (+1), C[0,k1] (+1)
        U[0][base_col] = 1    # A[0,0]
        U[1][base_col] = 1    # A[0,1]
        V[k0][base_col] = 1   # B[0,k0]
        V[k1][base_col] = 1   # B[0,k1]
        W[k0 * 2 + 0][base_col] = 1   # C[0,k0] coeff +1
        W[k1 * 2 + 0][base_col] = 1   # C[0,k1] coeff +1
        
        # m2 = A[0,0] * (B[1,k0] - B[1,k1])
        # contributes to C[0,k0] (+1)
        U[0][base_col + 1] = 1
        V[10 + k0][base_col + 1] = 1    # B[1,k0] at V[1*10+k0]
        V[10 + k1][base_col + 1] = -1   # B[1,k1] at V[1*10+k1]
        W[k0 * 2 + 0][base_col + 1] = 1   # C[0,k0] +1
        
        # m3 = (A[1,0] + A[1,1]) * B[0,k0]
        # contributes to C[1,k0] (+1), C[0,k1] (-1)
        U[3][base_col + 2] = 1    # A[1,0] at U[1*3+0]
        U[4][base_col + 2] = 1    # A[1,1] at U[1*3+1]
        V[0 + k0][base_col + 2] = 1   # B[0,k0]
        W[k0 * 2 + 1][base_col + 2] = 1   # C[1,k0] +1
        W[k1 * 2 + 0][base_col + 2] = -1  # C[0,k1] -1
        
        # m4 = A[1,1] * (B[1,k1] - B[1,k0])
        # contributes to C[0,k0] (+1), C[1,k0] (-1)
        U[4][base_col + 3] = 1    # A[1,1]
        V[10 + k1][base_col + 3] = 1   # B[1,k1]
        V[10 + k0][base_col + 3] = -1  # B[1,k0]
        W[k0 * 2 + 0][base_col + 3] = 1   # C[0,k0] +1
        W[k0 * 2 + 1][base_col + 3] = -1  # C[1,k0] -1
        
        # m5 = (A[0,0] + A[1,0]) * (B[0,k1] + B[1,k1])
        # contributes to C[0,k1] (-1), C[1,k1] (+1)
        U[0][base_col + 4] = 1    # A[0,0]
        U[3][base_col + 4] = 1    # A[1,0]
        V[0 + k1][base_col + 4] = 1   # B[0,k1]
        V[10 + k1][base_col + 4] = 1  # B[1,k1]
        W[k1 * 2 + 0][base_col + 4] = -1  # C[0,k1] -1
        W[k1 * 2 + 1][base_col + 4] = 1   # C[1,k1] +1
        
        # m6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        # contributes to C[1,k0] (-1), C[1,k1] (+1)
        U[1][base_col + 5] = 1    # A[0,1]
        U[4][base_col + 5] = -1   # A[1,1]
        V[10 + k0][base_col + 5] = 1  # B[1,k0]
        V[10 + k1][base_col + 5] = 1  # B[1,k1]
        W[k0 * 2 + 1][base_col + 5] = -1  # C[1,k0] -1
        W[k1 * 2 + 1][base_col + 5] = 1   # C[1,k1] +1
        
        # m7 = (A[0,0] - A[1,0]) * (B[0,k0] + B[1,k0])
        # contributes to C[0,k0] (-1), C[1,k0] (+1), C[0,k1] (+1)
        U[0][base_col + 6] = 1    # A[0,0]
        U[3][base_col + 6] = -1   # A[1,0]
        V[0 + k0][base_col + 6] = 1   # B[0,k0]
        V[10 + k0][base_col + 6] = 1  # B[1,k0]
        W[k0 * 2 + 0][base_col + 6] = -1  # C[0,k0] -1
        W[k0 * 2 + 1][base_col + 6] = 1   # C[1,k0] +1
        W[k1 * 2 + 0][base_col + 6] = 1   # C[0,k1] +1
    
    # Remaining part: C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,...,9}
    # Rank 2*1*10 = 20 terms
    for k in range(10):
        for i in range(2):
            col = 35 + k * 2 + i
            U[i * 3 + 2][col] = 1   # A[i,2] at U[i*3+2]
            V[2 * 10 + k][col] = 1  # B[2,k] at V[2*10+k]
            W[k * 2 + i][col] = 1   # C[i,k] at W[k*2+i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
