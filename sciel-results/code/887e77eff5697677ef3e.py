"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 (Strassen 2x2 composed with itself)
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen coefficients
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[3][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1; W2[3][4] = -1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 from 2x2
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for I in range(2):
                for J in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = I * 2 + i_prime
                            j = J * 2 + j_prime
                            u_val = U2[I*2+J][r1] * U2[i_prime*2+j_prime][r2]
                            U[i*m+j][r] = u_val
                            
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for J in range(2):
                for K in range(2):
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = J * 2 + j_prime
                            k = K * 2 + k_prime
                            v_val = V2[J*2+K][r1] * V2[j_prime*2+k_prime][r2]
                            V[j*p+k][r] = v_val
                            
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for K in range(2):
                for I in range(2):
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = K * 2 + k_prime
                            i = I * 2 + i_prime
                            w_val = W2[K*2+I][r1] * W2[k_prime*2+i_prime][r2]
                            W[k*n+i][r] = w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
