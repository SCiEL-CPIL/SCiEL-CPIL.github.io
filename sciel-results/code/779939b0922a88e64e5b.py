import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 110  # 110s budget
    
    n, m, p = DIMENSIONS
    # Use Strassen for each 2x2 block of B, giving rank 4*7=28
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7  # 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 x 2x2
    for b in range(num_blocks):
        base = b * 7
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0, base + 0] += 1
        U[3, base + 0] += 1
        V[0 * p + 2 * b, base + 0] += 1
        V[1 * p + 2 * b + 1, base + 0] += 1
        W[2 * b * n + 0, base + 0] += 1
        W[(2 * b + 1) * n + 1, base + 0] += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2, base + 1] += 1
        U[3, base + 1] += 1
        V[0 * p + 2 * b, base + 1] += 1
        W[2 * b * n + 1, base + 1] += 1
        W[(2 * b + 1) * n + 1, base + 1] -= 1
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0, base + 2] += 1
        V[0 * p + 2 * b + 1, base + 2] += 1
        V[1 * p + 2 * b + 1, base + 2] -= 1
        W[(2 * b + 1) * n + 0, base + 2] += 1
        W[(2 * b + 1) * n + 1, base + 2] += 1
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3, base + 3] += 1
        V[1 * p + 2 * b, base + 3] += 1
        V[0 * p + 2 * b, base + 3] -= 1
        W[2 * b * n + 0, base + 3] += 1
        W[2 * b * n + 1, base + 3] += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0, base + 4] += 1
        U[1, base + 4] += 1
        V[1 * p + 2 * b + 1, base + 4] += 1
        W[2 * b * n + 0, base + 4] -= 1
        W[(2 * b + 1) * n + 0, base + 4] += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2, base + 5] += 1
        U[0, base + 5] -= 1
        V[0 * p + 2 * b, base + 5] += 1
        V[0 * p + 2 * b + 1, base + 5] += 1
        W[(2 * b + 1) * n + 1, base + 5] += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1, base + 6] += 1
        U[3, base + 6] -= 1
        V[1 * p + 2 * b, base + 6] += 1
        V[1 * p + 2 * b + 1, base + 6] += 1
        W[2 * b * n + 0, base + 6] += 1

    # Verification
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.random.RandomState(42).randint(-5, 6, size=(2, 8)).astype(float)
    C_test = A_test @ B_test
    
    C_decomp = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                for j in range(m):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_decomp[i, k] = total
    
    if not np.allclose(C_decomp, C_test, atol=1e-10):
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s), dtype=int)
        V_s = np.zeros((m * p, rank_s), dtype=int)
        W_s = np.zeros((p * n, rank_s), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_s[i * m + j, r] = 1
                    V_s[j * p + k, r] = 1
                    W_s[k * n + i, r] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
