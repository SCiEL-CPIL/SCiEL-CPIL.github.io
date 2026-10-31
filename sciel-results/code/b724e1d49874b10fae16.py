import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Recursive Strassen: 7 outer terms * 7 inner terms = 49 rank
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index helpers
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(j, k):
        return j * p + k
        
    def w_idx(i, k):
        # W[k*n+i, r] convention: index = k*n + i
        return k * n + i

    # 2x2 Strassen coefficient arrays
    # beta[s][I][J]: coefficient for A[I*2+i, J*2+j] in outer term s
    # gamma[s][J][K]: coefficient for B[J*2+j, K*2+k] in outer term s
    # delta[s][I][K]: coefficient for C[I*2+i, K*2+k] in outer term s
    
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))
    
    # s=0 (M1): X=A00+A11, Y=B00+B11, C: C00+=1, C11+=1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1 (M2): X=A10+A11, Y=B00, C: C10+=1, C11-=1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2 (M3): X=A00, Y=B01-B11, C: C01+=1, C11+=1
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3 (M4): X=A11, Y=B10-B00, C: C00+=1, C10+=1
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4 (M5): X=A00+A01, Y=B11, C: C00-=1, C01+=1
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5 (M6): X=A10-A00, Y=B00+B01, C: C11+=1
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6 (M7): X=A01-A11, Y=B10+B11, C: C00+=1
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Inner 2x2 Strassen decomposition (rank 7)
    # U2[row][t], V2[row][t], W2[row][t]
    # Row indices: 0=A00, 1=A01, 2=A10, 3=A11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0 (M1): (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1 (M2): (A10+A11)B00 -> C10+=1, C11-=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # t=2 (M3): A00(B01-B11) -> C01+=1, C11+=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # t=3 (M4): A11(B10-B00) -> C00+=1, C10+=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # t=4 (M5): (A00+A01)B11 -> C00-=1, C01+=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # t=5 (M6): (A10-A00)(B00+B01) -> C11+=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6 (M7): (A01-A11)(B10+B11) -> C00+=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct U, V, W using Kronecker-like product structure
    # U[i,j, col] = sum_{I,J} beta[s][I][J] * U2[i%2, j%2, t]
    # where i = I*2 + i', j = J*2 + j', col = s*7 + t
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    b_coeff = beta[s, I, J]
                    if b_coeff == 0:
                        continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            u_coeff = U2[i_prime, j_prime, t]
                            if u_coeff == 0:
                                continue
                            i = I * 2 + i_prime
                            j = J * 2 + j_prime
                            U[a_idx(i, j), col] += b_coeff * u_coeff
            
            # Fill V
            for J in range(2):
                for K in range(2):
                    g_coeff = gamma[s, J, K]
                    if g_coeff == 0:
                        continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            v_coeff = V2[j_prime, k_prime, t]
                            if v_coeff == 0:
                                continue
                            j = J * 2 + j_prime
                            k = K * 2 + k_prime
                            V[b_idx(j, k), col] += g_coeff * v_coeff
            
            # Fill W
            for I in range(2):
                for K in range(2):
                    d_coeff = delta[s, I, K]
                    if d_coeff == 0:
                        continue
                    for i_prime in range(2):
                        for k_prime in range(2):
                            w_coeff = W2[k_prime, i_prime, t]
                            if w_coeff == 0:
                                continue
                            i = I * 2 + i_prime
                            k = K * 2 + k_prime
                            W[w_idx(i, k), col] += d_coeff * w_coeff
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
