import numpy as np
import time

DIMENSIONS = [2, 3, 3]
_DEADLINE = 150  # seconds, with safety margin

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 17 decomposition: Strassen for the 2x2 block + schoolbook for the rest
    rank = 17
    
    # Shapes
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6
    
    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)
    
    # Helper indices
    # A is n x m, index (i, j) -> i*m + j
    # B is m x p, index (j, k) -> j*p + k
    # C is n x p, index (i, k) -> k*n + i (per problem statement W[k*n+i])
    
    # --- Strassen's Algorithm for A[0:2, 0:2] * B[0:2, 0:2] ---
    # Indices 0-6
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00: 0*3+0=0, A11: 1*3+1=4
    # B00: 0*3+0=0, B11: 1*3+1=4
    # C00: k=0, i=0 -> 0
    # C11: k=1, i=1 -> 1*2+1=3
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10 + A11) * B00
    # A10: 1*3+0=3, A11: 4
    # B00: 0
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # M3 = A00 * (B01 - B11)
    # A00: 0
    # B01: 0*3+1=1, B11: 4
    # C01: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A11 * (B10 - B00)
    # A11: 4
    # B10: 1*3+0=3, B00: 0
    # C00: k=0, i=0 -> 0
    # C10: k=0, i=1 -> 1
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C[0,0]
    W[1, 3] = 1  # C[1,0]
    
    # M5 = (A00 + A01) * B11
    # A00: 0, A01: 0*3+1=1
    # B11: 4
    # C01: k=1, i=0 -> 2
    # C00: k=0, i=0 -> 0
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1  # C[0,1]
    W[0, 4] = -1 # C[0,0]
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10: 3, A00: 0
    # B00: 0, B01: 1
    # C11: k=1, i=1 -> 3
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01: 1, A11: 4
    # B10: 3, B11: 4
    # C00: k=0, i=0 -> 0
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # --- Schoolbook for A[:, 2] * B[2, :] (contribution to C[:, 0:2]) ---
    # Indices 7-10
    
    # Term: A02 * B20 -> C00
    # A02: 0*3+2=2
    # B20: 2*3+0=6
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1  # C[0,0]
    
    # Term: A02 * B21 -> C01
    # B21: 2*3+1=7
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1  # C[0,1]
    
    # Term: A12 * B20 -> C10
    # A12: 1*3+2=5
    # B20: 6
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1  # C[1,0]
    
    # Term: A12 * B21 -> C11
    # B21: 7
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1  # C[1,1]
    
    # --- Schoolbook for A[:, 0:2] * B[:, 2] (contribution to C[:, 2]) ---
    # Indices 11-14
    
    # Term: A00 * B02 -> C02
    # B02: 0*3+2=2
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1  # C[0,2] (k=2, i=0 -> 4)
    
    # Term: A01 * B12 -> C02
    # B12: 1*3+2=5
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1  # C[0,2]
    
    # Term: A10 * B02 -> C12
    # B02: 2
    U[3, 13] = 1
    V[2, 13] = 1
    W[5, 13] = 1  # C[1,2] (k=2, i=1 -> 5)
    
    # Term: A11 * B12 -> C12
    # B12: 5
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1  # C[1,2]
    
    # --- Schoolbook for A[:, 2] * B[2, 2] (contribution to C[:, 2]) ---
    # Indices 15-16
    
    # Term: A02 * B22 -> C02
    # B22: 2*3+2=8
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1  # C[0,2]
    
    # Term: A12 * B22 -> C12
    # B22: 8
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1  # C[1,2]
    
    # Verification step to ensure exactness
    def check_solution(U, V, W, R):
        np.random.seed(42)
        for trial in range(30):
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                w_vec = W[:, r]
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
            if not np.allclose(C_approx, C_true, atol=1e-9):
                return False
        return True

    if time.time() - t0 > _DEADLINE - 5:
        # Deadline approaching, return current best without full verification
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if not check_solution(U, V, W, rank):
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = 18
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
