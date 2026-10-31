"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    # Define 2x2 Strassen factors
    # U2[u, r], V2[v, r], W2[w, r] for u,v,w in 0..3, r in 0..6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # r=2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # r=3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # r=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # r=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1
    
    # Construct 4x4 factors
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        r_prime = r // 7
        r_double_prime = r % 7
        
        for u in range(n*m):
            i = u // m
            j = u % m
            i_prime = i // 2
            i_double_prime = i % 2
            j_prime = j // 2
            j_double_prime = j % 2
            u_prime = 2*i_prime + j_prime
            u_double_prime = 2*i_double_prime + j_double_prime
            U[u][r] = U2[u_prime][r_prime] * U2[u_double_prime][r_double_prime]
            
        for v in range(m*p):
            j = v // p
            k = v % p
            j_prime = j // 2
            j_double_prime = j % 2
            k_prime = k // 2
            k_double_prime = k % 2
            v_prime = 2*j_prime + k_prime
            v_double_prime = 2*j_double_prime + k_double_prime
            V[v][r] = V2[v_prime][r_prime] * V2[v_double_prime][r_double_prime]
            
        for w in range(p*n):
            k = w // n
            i = w % n
            k_prime = k // 2
            k_double_prime = k % 2
            i_prime = i // 2
            i_double_prime = i % 2
            w_prime = 2*k_prime + i_prime
            w_double_prime = 2*k_double_prime + i_double_prime
            W[w][r] = W2[w_prime][r_prime] * W2[w_double_prime][r_double_prime]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
