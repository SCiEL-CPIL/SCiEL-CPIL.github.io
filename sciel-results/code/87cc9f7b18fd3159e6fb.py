"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    
    # Part 1: A_1 (1x4) * B (4x8)
    # Split B into 4 blocks of 4x2
    for b in range(4):
        k_start = 2 * b
        for j in range(4):
            for k_prime in range(2):
                k = k_start + k_prime
                # U: A[0, j]
                U[0 * m + j][r_idx] = 1
                # V: B[j, k]
                V[j * p + k][r_idx] = 1
                # W: C[0, k] -> index k*n + 0
                W[k * n + 0][r_idx] = 1
                r_idx += 1
                
    # Part 2: A_2 (2x4) * B (4x8)
    # Split B into 4 blocks of 4x2
    # For each block, use 2 Strassen instances
    for b in range(4):
        k_start = 2 * b
        k0 = k_start
        k1 = k_start + 1
        
        # Instance 1: A_2_1 (2x2) * B_top (2x2)
        # A_2_1 indices: rows 1,2; cols 0,1
        # B_top indices: rows 0,1; cols k0, k1
        # C contributions: rows 1,2; cols k0, k1
        
        # Define Strassen terms for X, Y
        # X = A_2_1, Y = B_top
        # X_00=A_10, X_01=A_11, X_10=A_20, X_11=A_21
        # Y_00=B_0k0, Y_01=B_0k1, Y_10=B_1k0, Y_11=B_1k1
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1: (X00+X11)(Y00+Y11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # Z00+=1, Z01+=0, Z10+=0, Z11+=1? 
            # Wait, W_coeffs are for Z00, Z01, Z10, Z11
            # Z00 = M1 + M4 - M5 + M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M6
            
            # Let's list them properly
        ]
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # Z00+=1, Z01+=0, Z10+=0, Z11+=1
        # Map to global
        # U: A[1,0]+=1, A[2,1]+=1
        U[1 * m + 0][r_idx] += 1
        U[2 * m + 1][r_idx] += 1
        # V: B[0,k0]+=1, B[1,k1]+=1
        V[0 * p + k0][r_idx] += 1
        V[1 * p + k1][r_idx] += 1
        # W: C[1,k0]+=1, C[2,k1]+=1
        W[k0 * n + 1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M2: (X10+X11)Y00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # Z10+=1, Z11+=-1
        U[1 * m + 0][r_idx] += 0 # X10 is A[2,0]
        # X10 = A[2,0], X11 = A[2,1]
        U[2 * m + 0][r_idx] += 1
        U[2 * m + 1][r_idx] += 1
        V[0 * p + k0][r_idx] += 1
        W[k0 * n + 2][r_idx] += 1
        W[k1 * n + 2][r_idx] += -1
        r_idx += 1
        
        # M3: X00(Y10-Y11)
        u = [1, 0, 0, 0] # X00 = A[1,0]
        v = [0, 0, 1, -1] # Y10=B[1,k0], Y11=B[1,k1]
        w = [0, 1, 0, 1] # Z01+=1, Z11+=1
        U[1 * m + 0][r_idx] += 1
        V[1 * p + k0][r_idx] += 1
        V[1 * p + k1][r_idx] += -1
        W[k1 * n + 1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M4: X11(Y01-Y00)
        u = [0, 0, 0, 1] # X11 = A[2,1]
        v = [0, 1, 0, -1] # Y01=B[0,k1], Y00=B[0,k0]
        w = [1, 0, 1, 0] # Z00+=1, Z10+=1
        U[2 * m + 1][r_idx] += 1
        V[0 * p + k1][r_idx] += 1
        V[0 * p + k0][r_idx] += -1
        W[k0 * n + 1][r_idx] += 1
        W[k0 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M5: (X00+X01)Y11
        u = [1, 1, 0, 0] # X00=A[1,0], X01=A[1,1]
        v = [0, 0, 0, 1] # Y11=B[1,k1]
        w = [-1, 1, 0, 0] # Z00+=-1, Z01+=1
        U[1 * m + 0][r_idx] += 1
        U[1 * m + 1][r_idx] += 1
        V[1 * p + k1][r_idx] += 1
        W[k0 * n + 1][r_idx] += -1
        W[k1 * n + 1][r_idx] += 1
        r_idx += 1
        
        # M6: (X10-X00)(Y00+Y01)
        u = [0, 0, 1, -1] # X10=A[2,0], X00=A[1,0] -> Wait, X10 is row 1 col 0 of X.
        # X = A_2_1. X[1,0] is A[2,0]. X[0,0] is A[1,0].
        u = [0, 0, 1, 0] # X10
        # Need to subtract X00.
        # u = [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0]
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0] # Y00=B[0,k0], Y01=B[0,k1]
        w = [0, 0, 0, 1] # Z11+=1
        U[1 * m + 0][r_idx] += -1
        U[2 * m + 0][r_idx] += 1
        V[0 * p + k0][r_idx] += 1
        V[0 * p + k1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M7: (X01-X11)(Y10+Y11)
        # X01=A[1,1], X11=A[2,1]
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1] # Y10=B[1,k0], Y11=B[1,k1]
        w = [1, 0, 0, 0] # Z00+=1
        U[1 * m + 1][r_idx] += 1
        U[2 * m + 1][r_idx] += -1
        V[1 * p + k0][r_idx] += 1
        V[1 * p + k1][r_idx] += 1
        W[k0 * n + 1][r_idx] += 1
        r_idx += 1
        
        # Instance 2: A_2_2 (2x2) * B_bot (2x2)
        # A_2_2 indices: rows 1,2; cols 2,3
        # B_bot indices: rows 2,3; cols k0, k1
        
        # M1: (X00+X11)(Y00+Y11)
        # X00=A[1,2], X11=A[2,3]
        # Y00=B[2,k0], Y11=B[3,k1]
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1]
        U[1 * m + 2][r_idx] += 1
        U[2 * m + 3][r_idx] += 1
        V[2 * p + k0][r_idx] += 1
        V[3 * p + k1][r_idx] += 1
        W[k0 * n + 1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M2: (X10+X11)Y00
        # X10=A[2,2], X11=A[2,3]
        # Y00=B[2,k0]
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1]
        U[2 * m + 2][r_idx] += 1
        U[2 * m + 3][r_idx] += 1
        V[2 * p + k0][r_idx] += 1
        W[k0 * n + 2][r_idx] += 1
        W[k1 * n + 2][r_idx] += -1
        r_idx += 1
        
        # M3: X00(Y10-Y11)
        # X00=A[1,2]
        # Y10=B[3,k0], Y11=B[3,k1]
        u = [1, 0, 0, 0]
        v = [0, 0, 1, -1]
        w = [0, 1, 0, 1]
        U[1 * m + 2][r_idx] += 1
        V[3 * p + k0][r_idx] += 1
        V[3 * p + k1][r_idx] += -1
        W[k1 * n + 1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M4: X11(Y01-Y00)
        # X11=A[2,3]
        # Y01=B[2,k1], Y00=B[2,k0]
        u = [0, 0, 0, 1]
        v = [0, 1, 0, -1]
        w = [1, 0, 1, 0]
        U[2 * m + 3][r_idx] += 1
        V[2 * p + k1][r_idx] += 1
        V[2 * p + k0][r_idx] += -1
        W[k0 * n + 1][r_idx] += 1
        W[k0 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M5: (X00+X01)Y11
        # X00=A[1,2], X01=A[1,3]
        # Y11=B[3,k1]
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0]
        U[1 * m + 2][r_idx] += 1
        U[1 * m + 3][r_idx] += 1
        V[3 * p + k1][r_idx] += 1
        W[k0 * n + 1][r_idx] += -1
        W[k1 * n + 1][r_idx] += 1
        r_idx += 1
        
        # M6: (X10-X00)(Y00+Y01)
        # X10=A[2,2], X00=A[1,2]
        # Y00=B[2,k0], Y01=B[2,k1]
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1]
        U[1 * m + 2][r_idx] += -1
        U[2 * m + 2][r_idx] += 1
        V[2 * p + k0][r_idx] += 1
        V[2 * p + k1][r_idx] += 1
        W[k1 * n + 2][r_idx] += 1
        r_idx += 1
        
        # M7: (X01-X11)(Y10+Y11)
        # X01=A[1,3], X11=A[2,3]
        # Y10=B[3,k0], Y11=B[3,k1]
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0]
        U[1 * m + 3][r_idx] += 1
        U[2 * m + 3][r_idx] += -1
        V[3 * p + k0][r_idx] += 1
        V[3 * p + k1][r_idx] += 1
        W[k0 * n + 1][r_idx] += 1
        r_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
