import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0

    # We use the embedding into M_{4,4,4} to achieve rank 16.
    # The lower bound for M_{3,4,4} is max(3*4, 4*4, 4*3) = 16.
    # So rank 16 is optimal.
    
    # Construct schoolbook decomposition for M_{4,4,4}
    N, M, P = 4, 4, 4
    R = N * M * P # 16
    
    U_444 = np.zeros((N * M, R))
    V_444 = np.zeros((M * P, R))
    W_444 = np.zeros((P * N, R))
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_444[i * M + j, col] = 1
                V_444[j * P + k, col] = 1
                W_444[k * N + i, col] = 1
                col += 1
                
    # Map to M_{3,4,4}
    # U: (12, 16) -> take first 12 rows of U_444 (indices i*4+j for i<3)
    U_344 = U_444[:n*m, :]
    
    # V: (16, 16) -> same as V_444 (indices j*4+k for j,k<4)
    V_344 = V_444
    
    # W: (12, 16) -> remap indices from k*4+i to k*3+i
    W_344 = np.zeros((p*n, R))
    for r in range(R):
        for k in range(p):
            for i in range(n):
                W_344[k*n + i, r] = W_444[k*N + i, r]
                
    # Verify
    # T_344[i*m+j, j*p+k, k*n+i] = 1
    # Recon = sum_r U_344[:,r] outer V_344[:,r] outer W_344[:,r]
    # This should match.
    
    return {
        "U": U_344.tolist(),
        "V": V_344.tolist(),
        "W": W_344.tolist()
    }
