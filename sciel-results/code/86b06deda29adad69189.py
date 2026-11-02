"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank-17 decomposition using Strassen for 2x2x2 + schoolbook for cross terms
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 6 rows
    
    # Strassen's 7 products for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # Then:
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M7
    
    # Cross terms (involving j=2 or k=2):
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]  -> 3 schoolbook
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]  -> 3 schoolbook
    # C[0,0] also has A[0,2]*B[2,0], C[0,1] has A[0,2]*B[2,1]
    # C[1,0] also has A[1,2]*B[2,0], C[1,1] has A[1,2]*B[2,1]
    # That's 4 more schoolbook terms for row 2 of B with cols 0,1
    # And A[0,2]*B[2,2], A[1,2]*B[2,2] are already counted above
    
    # Total: 7 (Strassen) + 3 + 3 + 4 = 17
    
    # Let me assign columns:
    # cols 0-6: Strassen products M1..M7
    # col 7: A[0,2]*B[2,0] -> contributes to C[0,0]
    # col 8: A[0,2]*B[2,1] -> contributes to C[0,1]
    # col 9: A[1,2]*B[2,0] -> contributes to C[1,0]
    # col 10: A[1,2]*B[2,1] -> contributes to C[1,1]
    # col 11: A[0,0]*B[0,2] -> contributes to C[0,2]
    # col 12: A[0,1]*B[1,2] -> contributes to C[0,2]
    # col 13: A[0,2]*B[2,2] -> contributes to C[0,2]
    # col 14: A[1,0]*B[0,2] -> contributes to C[1,2]
    # col 15: A[1,1]*B[1,2] -> contributes to C[1,2]
    # col 16: A[1,2]*B[2,2] -> contributes to C[1,2]
    
    # U indices: i*m + j where i in {0,1}, j in {0,1,2}
    # U[0] = A[0,0], U[1] = A[0,1], U[2] = A[0,2]
    # U[3] = A[1,0], U[4] = A[1,1], U[5] = A[1,2]
    
    # V indices: j*p + k where j in {0,1,2}, k in {0,1,2}
    # V[0] = B[0,0], V[1] = B[0,1], V[2] = B[0,2]
    # V[3] = B[1,0], V[4] = B[1,1], V[5] = B[1,2]
    # V[6] = B[2,0], V[7] = B[2,1], V[8] = B[2,2]
    
    # W indices: k*n + i where k in {0,1,2}, i in {0,1}
    # W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1], W[4] = C[0,2], W[5] = C[1,2]
    
    # Strassen M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: U[0]=1, U[4]=1
    # V: V[0]=1, V[4]=1
    # W: W[0]=1, W[2]=-1, W[1]=0, W[3]=1  (C[0,0]+=M1, C[0,1]-=M1, C[1,0]+=0, C[1,1]+=M1)
    # Wait, let me re-derive:
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M7
    
    # M1 contributes: C[0,0]+=1, C[1,1]+=1
    U[0][0] = 1  # A[0,0]
    U[4][0] = 1  # A[1,1]
    V[0][0] = 1  # B[0,0]
    V[4][0] = 1  # B[1,1]
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # Contributes: C[1,0]+=1
    U[3][1] = 1  # A[1,0]
    U[4][1] = 1  # A[1,1]
    V[0][1] = 1  # B[0,0]
    W[1][1] = 1  # C[1,0]
    
    # M3: A[0,0] * (B[0,1]-B[1,1])
    # Contributes: C[0,1]+=1, C[1,1]-=1
    U[0][2] = 1  # A[0,0]
    V[1][2] = 1  # B[0,1]
    V[4][2] = -1  # B[1,1]
    W[2][2] = 1  # C[0,1]
    W[3][2] = -1  # C[1,1]
    
    # M4: A[1,1] * (B[1,0]-B[0,0])
    # Contributes: C[0,0]-=1, C[1,0]+=1
    U[4][3] = 1  # A[1,1]
    V[3][3] = 1  # B[1,0]
    V[0][3] = -1  # B[0,0]
    W[0][3] = -1  # C[0,0]
    W[1][3] = 1  # C[1,0]
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # Contributes: C[0,1]+=1
    U[0][4] = 1  # A[0,0]
    U[1][4] = 1  # A[0,1]
    V[4][4] = 1  # B[1,1]
    W[2][4] = 1  # C[0,1]
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Contributes: C[0,0]+=1
    U[3][5] = 1  # A[1,0]
    U[0][5] = -1  # A[0,0]
    V[0][5] = 1  # B[0,0]
    V[1][5] = 1  # B[0,1]
    W[0][5] = 1  # C[0,0]
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # Contributes: C[1,1]+=1
    U[1][6] = 1  # A[0,1]
    U[4][6] = -1  # A[1,1]
    V[3][6] = 1  # B[1,0]
    V[4][6] = 1  # B[1,1]
    W[3][6] = 1  # C[1,1]
    
    # Schoolbook cross terms:
    # col 7: A[0,2]*B[2,0] -> C[0,0]+=1
    U[2][7] = 1
    V[6][7] = 1
    W[0][7] = 1
    
    # col 8: A[0,2]*B[2,1] -> C[0,1]+=1
    U[2][8] = 1
    V[7][8] = 1
    W[2][8] = 1
    
    # col 9: A[1,2]*B[2,0] -> C[1,0]+=1
    U[5][9] = 1
    V[6][9] = 1
    W[1][9] = 1
    
    # col 10: A[1,2]*B[2,1] -> C[1,1]+=1
    U[5][10] = 1
    V[7][10] = 1
    W[3][10] = 1
    
    # col 11: A[0,0]*B[0,2] -> C[0,2]+=1
    U[0][11] = 1
    V[2][11] = 1
    W[4][11] = 1
    
    # col 12: A[0,1]*B[1,2] -> C[0,2]+=1
    U[1][12] = 1
    V[5][12] = 1
    W[4][12] = 1
    
    # col 13: A[0,2]*B[2,2] -> C[0,2]+=1
    U[2][13] = 1
    V[8][13] = 1
    W[4][13] = 1
    
    # col 14: A[1,0]*B[0,2] -> C[1,2]+=1
    U[3][14] = 1
    V[2][14] = 1
    W[5][14] = 1
    
    # col 15: A[1,1]*B[1,2] -> C[1,2]+=1
    U[4][15] = 1
    V[5][15] = 1
    W[5][15] = 1
    
    # col 16: A[1,2]*B[2,2] -> C[1,2]+=1
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 18
        rank_schoolbook = n * m * p
        U_s = [[0] * rank_schoolbook for _ in range(n * m)]
        V_s = [[0] * rank_schoolbook for _ in range(m * p)]
        W_s = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
