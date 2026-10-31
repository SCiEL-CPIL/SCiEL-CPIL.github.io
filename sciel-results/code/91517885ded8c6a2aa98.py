import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 block level (s=0..6)
    # beta[s][I][J] for A[I][J]
    # gamma[s][J][K] for B[J][K]
    # delta[s][I][K] for C[I][K]
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))
    
    # s=0
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Strassen coefficients for 2x2 scalar level (t=0..6)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # Fill U
            # U[i, j, c] = beta[s, i//2, j//2] * U2[i%2*2 + j%2, t]
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    local_u_idx = i_prime * 2 + j_prime
                    coeff = beta[s, I, J] * U2[local_u_idx, t]
                    if coeff != 0:
                        U[row_u, c] = coeff
                        
            # Fill V
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    local_v_idx = j_prime * 2 + k_prime
                    coeff = gamma[s, J, K] * V2[local_v_idx, t]
                    if coeff != 0:
                        V[row_v, c] = coeff
                        
            # Fill W
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i # Trace convention: W[k*n+i, r]
                    local_w_idx = k_prime * 2 + i_prime
                    coeff = delta[s, I, K] * W2[local_w_idx, t]
                    if coeff != 0:
                        W[row_w, c] = coeff
                        
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
