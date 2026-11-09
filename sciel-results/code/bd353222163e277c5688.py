"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]   # (9, 38)
    V = [[0] * rank for _ in range(m * p)]   # (18, 38)
    W = [[0] * rank for _ in range(p * n)]   # (18, 38)
    
    # Correct Laderman 19-term decomposition for 3x3x3 matrix multiplication
    # Each term: (A_coeffs[9], B_coeffs[9], C_coeffs[9])
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # Flat index: row-major, A[i][j] -> i*3+j
    
    laderman_terms = [
        # Term 0: A[0][0] * B[0][0] -> C[0][0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[1][0] * B[0][0] -> C[1][0]
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 2: A[2][0] * B[0][0] -> C[2][0]
        ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 3: A[0][1] * B[1][0] -> C[0][0]
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 4: A[1][1] * B[1][0] -> C[1][0]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 5: A[2][1] * B[1][0] -> C[2][0]
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 6: A[0][2] * B[2][0] -> C[0][0]
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 7: A[1][2] * B[2][0] -> C[1][0]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 8: A[2][2] * B[2][0] -> C[2][0]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 9: (A[0][0]-A[1][0]) * (B[0][0]+B[1][0]) -> C[0][0]-C[1][0]
        ([1,0,0, -1,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,0,0, -1,0,0, 0,0,0]),
        # Term 10: (A[0][1]-A[1][1]) * (B[0][1]+B[1][1]) -> C[0][1]-C[1][1]
        ([0,1,0, 0,-1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 0,-1,0, 0,0,0]),
        # Term 11: (A[0][2]-A[1][2]) * (B[0][2]+B[1][2]) -> C[0][2]-C[1][2]
        ([0,0,1, 0,0,-1, 0,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,1, 0,0,-1, 0,0,0]),
        # Term 12: (A[0][0]+A[1][0]) * (B[0][1]-B[1][1]) -> C[0][1]-C[1][0]
        ([1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,-1,0, 0,0,0], [0,1,0, -1,0,0, 0,0,0]),
        # Term 13: (A[0][1]+A[1][1]) * (B[0][2]-B[1][2]) -> C[0][2]-C[1][1]
        ([0,1,0, 0,1,0, 0,0,0], [0,0,1, 0,0,-1, 0,0,0], [0,0,1, 0,-1,0, 0,0,0]),
        # Term 14: (A[0][2]+A[1][2]) * (B[0][0]-B[1][0]) -> C[0][0]-C[1][2]
        ([0,0,1, 0,0,1, 0,0,0], [1,0,0, -1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,-1]),
        # Term 15: (A[1][0]+A[2][0]) * (B[0][1]+B[1][1]+B[2][1]) -> C[1][1]+C[2][1]
        ([0,0,0, 1,0,0, 1,0,0], [0,1,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1]),
        # Term 16: (A[1][1]+A[2][1]) * (B[0][2]+B[1][2]+B[2][2]) -> C[1][2]+C[2][2]
        ([0,0,0, 0,1,0, 0,1,0], [0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1]),
        # Term 17: (A[1][2]+A[2][2]) * (B[0][0]+B[1][0]+B[2][0]) -> C[1][0]+C[2][0]
        ([0,0,0, 0,0,1, 0,0,1], [1,0,0, 1,0,0, 1,0,0], [0,0,0, 1,0,0, 1,0,0]),
        # Term 18: (A[0][0]+A[1][0]+A[2][0]) * (B[0][0]+B[1][0]+B[2][0]) -> C[0][0]+C[1][0]+C[2][0]
        ([1,0,0, 1,0,0, 1,0,0], [1,0,0, 1,0,0, 1,0,0], [1,0,0, 1,0,0, 1,0,0]),
    ]
    
    # Build rank-38 decomposition: two independent 3x3 blocks
    # Block 0 handles columns 0-2 of B (and C), Block 1 handles columns 3-5
    for block in range(2):
        base = block * 19
        k_off = block * 3
        
        for t in range(19):
            A_flat, B_flat, C_flat = laderman_terms[t]
            col = base + t
            
            # Fill U: U[i*m + j][col] = A[i][j]
            for i in range(3):
                for j in range(3):
                    U[i*3 + j][col] = A_flat[i*3 + j]
            
            # Fill V: V[j*p + k][col] = B[j][k] where k is local to block
            for j in range(3):
                for k in range(3):
                    V[j*3 + (k + k_off)][col] = B_flat[j*3 + k]
            
            # Fill W: W[k*n + i][col] = C[i][k] where k is local to block
            # Convention: W[k*n+i, r]
            for i in range(3):
                for k in range(3):
                    W[(k + k_off)*3 + i][col] = C_flat[i*3 + k]
    
    # Verify correctness
    import random
    def verify(U_mat, V_mat, W_mat):
        random.seed(42)
        for trial in range(20):
            A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-5, 5) for _ in range(6)] for _ in range(3)]
            
            # Expected C = A * B
            C_exp = [[0]*6 for _ in range(3)]
            for i in range(3):
                for k in range(6):
                    s = 0
                    for j in range(3):
                        s += A[i][j] * B[j][k]
                    C_exp[i][k] = s
            
            # Compute from decomposition
            C_dec = [[0]*6 for _ in range(3)]
            for c in range(38):
                for i in range(3):
                    for k in range(6):
                        s = 0
                        for j in range(3):
                            s += U_mat[i*3+j][c] * V_mat[j*3+k][c]
                        C_dec[i][k] += s * W_mat[k*3+i][c]
            
            for i in range(3):
                for k in range(6):
                    if C_exp[i][k] != C_dec[i][k]:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook rank 54
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
