"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen factors
    # U2, V2, W2 are 4x7 matrices
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Fill U2
    U2[0][0] = 1; U2[3][0] = 1
    U2[0][1] = 1; U2[3][1] = 1
    U2[0][2] = 1
    U2[3][3] = 1
    U2[1][4] = 1; U2[3][4] = 1
    U2[0][5] = 1; U2[1][5] = -1
    U2[2][6] = 1; U2[0][6] = -1
    
    # Fill V2
    V2[0][0] = 1; V2[3][0] = 1
    V2[3][1] = 1
    V2[0][2] = 1; V2[2][2] = -1
    V2[3][3] = 1; V2[1][3] = -1
    V2[0][4] = 1
    V2[1][5] = 1; V2[3][5] = 1
    V2[0][6] = 1; V2[2][6] = 1
    
    # Fill W2
    W2[0][0] = 1; W2[3][0] = 1
    W2[2][1] = 1; W2[3][1] = -1
    W2[1][2] = 1; W2[3][2] = 1
    W2[0][3] = 1; W2[2][3] = 1
    W2[0][4] = -1; W2[1][4] = 1
    W2[3][5] = 1
    W2[0][6] = 1
    
    # Construct 4x4 factors via Kronecker product logic
    for I in range(2):
        for J in range(2):
            for i_prime in range(2):
                for j_prime in range(2):
                    row_U = (2*I + i_prime)*4 + (2*J + j_prime)
                    for r_prime in range(7):
                        for r_double_prime in range(7):
                            col = 7*r_prime + r_double_prime
                            val = U2[2*I + J][r_prime] * U2[2*i_prime + j_prime][r_double_prime]
                            U[row_U][col] += val
                            
    for J in range(2):
        for K in range(2):
            for j_prime in range(2):
                for k_prime in range(2):
                    row_V = (2*J + j_prime)*4 + (2*K + k_prime)
                    for r_prime in range(7):
                        for r_double_prime in range(7):
                            col = 7*r_prime + r_double_prime
                            val = V2[2*J + K][r_prime] * V2[2*j_prime + k_prime][r_double_prime]
                            V[row_V][col] += val
                            
    for K in range(2):
        for I in range(2):
            for k_prime in range(2):
                for i_prime in range(2):
                    row_W = (2*K + k_prime)*4 + (2*I + i_prime)
                    for r_prime in range(7):
                        for r_double_prime in range(7):
                            col = 7*r_prime + r_double_prime
                            val = W2[2*K + I][r_prime] * W2[2*k_prime + i_prime][r_double_prime]
                            W[row_W][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
