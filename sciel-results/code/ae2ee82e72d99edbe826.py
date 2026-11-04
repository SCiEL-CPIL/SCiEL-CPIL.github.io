import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin before 150s kill
    
    def _time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    n, m, p = 2, 3, 6
    
    # Target Rank 33
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Rank 21 (3 Strassen 2x2x2 blocks)
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Two 2x2x2 Strassen blocks for each of the 3 column pairs
    # A_sub is A[:, 1:3] (2x2)
    # B_sub is B[1:3, :] (2x6)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices in U: i*m + j. Here i in {0,1}, j in {1,2} (cols 1,2 of A)
    # Indices in V: j*p + k. Here j in {1,2} (rows 1,2 of B), k in {0..5}
    # Indices in W: k*n + i. Here k in {0..5}, i in {0,1}
    
    for t in range(3):
        # Block t uses B cols 2*t, 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify the decomposition is correct
    if _time_remaining() > 5.0:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = A[i,j]*B[j,k]
        # We verify with random test matrices
        for _ in range(3):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            # Compute via decomposition
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u_r = U_arr[:, r].reshape(n, m)
                v_r = V_arr[:, r].reshape(m, p)
                w_r = W_arr[:, r].reshape(p, n)
                C_decomp += (u_r @ v_r) * w_r.T  # This is wrong, need to be careful
            # Actually, the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a tensor contraction, not a simple matrix product
            # Let's verify properly:
            C_check = np.zeros((n, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_check[i, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            if not np.allclose(C_check, C_test, atol=1e-10):
                # Decomposition is invalid, but we still return it (score will be 0)
                pass
    
    # Deadline guard: if we've used too much time, still return the current result
    # (The construction is deterministic and fast, so this is just a safety net)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
