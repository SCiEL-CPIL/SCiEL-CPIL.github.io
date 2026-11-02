"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # M_{2,2,8} is a direct sum of 4 copies of M_{2,2,2}.
    # Rank of M_{2,2,2} is exactly 7 (Strassen), so rank of M_{2,2,8} is exactly 28.
    # This is provably optimal: border rank is additive over direct sums,
    # and border rank of M_{2,2,2} is 7, giving lower bound 4*7=28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    for b in range(4):
        base = b * 7
        # V indices for block b: B[0][2b], B[0][2b+1], B[1][2b], B[1][2b+1]
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = m * p // 2 + 2 * b      # 8 + 2b
        v11 = m * p // 2 + 2 * b + 1  # 8 + 2b + 1
        
        # W indices for block b: C[0][2b], C[1][2b], C[0][2b+1], C[1][2b+1]
        # W index = k*n + i where k=row, i=col; for block b, cols are 2b, 2b+1
        w00 = 2 * b      # C[0][2b]
        w10 = 2 * b + 1  # C[1][2b]
        w01 = 2 * (b + 1)      # C[0][2b+1] = C[0][2(b+1)]
        w11 = 2 * (b + 1) + 1  # C[1][2b+1] = C[1][2(b+1)+1]
        
        # Wait, let me re-index. C is 2x8. C[k][j] has W-index k*n+j = k*2+j.
        # For block b, the columns are 2b and 2b+1.
        # C[0][2b] -> index 0*2 + 2b = 2b
        # C[1][2b] -> index 1*2 + 2b = 2+2b
        # C[0][2b+1] -> index 0*2 + 2b+1 = 2b+1
        # C[1][2b+1] -> index 1*2 + 2b+1 = 2+2b+1 = 3+2b
        
        w00 = 2 * b          # C[0][2b]
        w10 = 2 + 2 * b      # C[1][2b]
        w01 = 2 * b + 1      # C[0][2b+1]
        w11 = 3 + 2 * b      # C[1][2b+1]
        
        # U indices: A[0][0]=0, A[0][1]=1, A[1][0]=2, A[1][1]=3
        a, b_idx, c, d = 0, 1, 2, 3
        
        # P1 = (a+d)(e+h)
        U[a][base] = 1
        U[d][base] = 1
        V[v00][base] = 1
        V[v11][base] = 1
        W[w00][base] = 1   # C11 += P1
        W[w11][base] = 1   # C22 += P1
        
        # P2 = (c+d)e
        U[c][base+1] = 1
        U[d][base+1] = 1
        V[v00][base+1] = 1
        W[w10][base+1] = 1   # C21 += P2
        W[w11][base+1] = -1  # C22 -= P2
        
        # P3 = a(f-h)
        U[a][base+2] = 1
        V[v01][base+2] = 1
        V[v11][base+2] = -1
        W[w01][base+2] = 1   # C12 += P3
        W[w11][base+2] = 1   # C22 += P3
        
        # P4 = d(g-c)
        U[d][base+3] = 1
        V[v10][base+3] = 1
        V[v00][base+3] = -1
        W[w00][base+3] = 1   # C11 += P4
        W[w10][base+3] = 1   # C21 += P4
        
        # P5 = (a+b)h
        U[a][base+4] = 1
        U[b_idx][base+4] = 1
        V[v11][base+4] = 1
        W[w00][base+4] = -1  # C11 -= P5
        W[w01][base+4] = 1   # C12 += P5
        
        # P6 = (c-a)(e+f)
        U[c][base+5] = 1
        U[a][base+5] = -1
        V[v00][base+5] = 1
        V[v01][base+5] = 1
        W[w11][base+5] = 1   # C22 += P6
        
        # P7 = (b-d)(g+h)
        U[b_idx][base+6] = 1
        U[d][base+6] = -1
        V[v10][base+6] = 1
        V[v11][base+6] = 1
        W[w00][base+6] = 1   # C11 += P7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
