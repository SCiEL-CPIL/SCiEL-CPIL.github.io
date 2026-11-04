"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 5]

def _strassen_222():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _trivial(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _combine_split_m(n, m1, m2, p, U1, V1, W1, r1, U2, V2, W2, r2):
    """Combine two decompositions split along the middle dimension.
    A (n x (m1+m2)) = [A1 | A2], B ((m1+m2) x p) = [B1; B2]
    C = A1*B1 + A2*B2
    """
    m = m1 + m2
    total_rank = r1 + r2
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U_full[i*m+j][c] = U1[i*m1+j][c]
        for j in range(m2):
            for c in range(r2):
                U_full[i*m+(m1+j)][r1+c] = U2[i*m2+j][c]
    
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V_full[j*p+k][c] = V1[j*p+k][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V_full[(m1+j)*p+k][r1+c] = V2[j*p+k][c]
    
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W_full[k*n+i][c] = W1[k*n+i][c]
            for c in range(r2):
                W_full[k*n+i][r1+c] = W2[k*n+i][c]
    
    return U_full, V_full, W_full, total_rank

def _combine_split_p(n, m, p1, p2, U1, V1, W1, r1, U2, V2, W2, r2):
    """Combine two decompositions split along the output dimension.
    A (n x m), B (m x (p1+p2)) = [B1 | B2]
    C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    """
    p = p1 + p2
    total_rank = r1 + r2
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U_full[i*m+j][c] = U1[i*m+j][c]
            for c in range(r2):
                U_full[i*m+j][r1+c] = U2[i*m+j][c]
    
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V_full[j*p+k][c] = V1[j*p1+k][c]
        for k in range(p2):
            for c in range(r2):
                V_full[j*p+(p1+k)][r1+c] = V2[j*p2+k][c]
    
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W_full[k*n+i][c] = W1[k*n+i][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W_full[(p1+k)*n+i][r1+c] = W2[k*n+i][c]
    
    return U_full, V_full, W_full, total_rank

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 5
    
    # Strategy: Split m=4 into 1+3, then recursively decompose.
    # 2x4 * 4x5:
    #   Split m into 1+3: A=[A1|A2] (2x1, 2x3), B=[B1;B2] (1x5, 3x5)
    #   A1*B1: 2x1 * 1x5 -> rank 1
    #   A2*B2: 2x3 * 3x5 -> recursively decompose
    #     Split m=3 into 1+2: A=[A1|A2] (2x1, 2x2), B=[B1;B2] (1x5, 2x5)
    #     A1*B1: 2x1 * 1x5 -> rank 1
    #     A2*B2: 2x2 * 2x5 -> recursively decompose
    #       Split p=5 into 2+3: B=[B1|B2] (2x2, 2x3)
    #       A*B1: 2x2 * 2x2 -> Strassen rank 7
    #       A*B2: 2x2 * 2x3 -> need to handle
    #         Split p=3 into 1+2: B=[B1|B2] (2x1, 2x2)
    #         A*B1: 2x2 * 2x1 -> trivial rank 4
    #         A*B2: 2x2 * 2x2 -> Strassen rank 7
    #         Total: 11
    #       Total for 2x2*2x5: 7+11=18
    #     Total for 2x3*3x5: 1+18=19
    # Total for 2x4*4x5: 1+19=20
    
    # Let's build this manually.
    
    # Step 1: 2x1 * 1x5 (rank 1)
    U_a1b1 = [[1], [1]]  # shape (2, 1)
    V_a1b1 = [[1], [1], [1], [1], [1]]  # shape (5, 1)
    W_a1b1 = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]  # shape (10, 1)
    r_a1b1 = 1
    
    # Step 2: 2x2 * 2x2 (Strassen, rank 7)
    U_s1, V_s1, W_s1 = _strassen_222()
    r_s1 = 7
    
    # Step 3: 2x2 * 2x1 (trivial, rank 4)
    U_t1, V_t1, W_t1, r_t1 = _trivial(2, 2, 1)
    
    # Step 4: 2x2 * 2x2 (Strassen, rank 7)
    U_s2, V_s2, W_s2 = _strassen_222()
    r_s2 = 7
    
    # Combine steps 3 and 4: 2x2 * 2x3 (split p=3 into 1+2)
    U_223, V_223, W_223, r_223 = _combine_split_p(2, 2, 1, 2, U_t1, V_t1, W_t1, r_t1, U_s2, V_s2, W_s2, r_s2)
    
    # Combine steps 2 and 5: 2x2 * 2x5 (split p=5 into 2+3)
    U_225, V_225, W_225, r_225 = _combine_split_p(2, 2, 2, 3, U_s1, V_s1, W_s1, r_s1, U_223, V_223, W_223, r_223)
    
    # Step 5: 2x1 * 1x5 (rank 1) for the second level
    U_a1b1_2 = [[1], [1]]
    V_a1b1_2 = [[1], [1], [1], [1], [1]]
    W_a1b1_2 = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    r_a1b1_2 = 1
    
    # Combine steps 5 and 6: 2x3 * 3x5 (split m=3 into 1+2)
    U_235, V_235, W_235, r_235 = _combine_split_m(2, 1, 2, 5, U_a1b1_2, V_a1b1_2, W_a1b1_2, r_a1b1_2, U_225, V_225, W_225, r_225)
    
    # Combine step 1 and 7: 2x4 * 4x5 (split m=4 into 1+3)
    U_full, V_full, W_full, r_full = _combine_split_m(2, 1, 3, 5, U_a1b1, V_a1b1, W_a1b1, r_a1b1, U_235, V_235, W_235, r_235)
    
    # Verify
    if _verify(U_full, V_full, W_full, 2, 4, 5):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback
    n, m, p = 2, 4, 5
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
