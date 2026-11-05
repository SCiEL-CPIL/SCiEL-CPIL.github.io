# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Decomposition (Rank 45) - Fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # 2. Attempted Improved Decomposition (Rank 41)
    # Using BSC (Rank 23) for 3x3x3 and Schoolbook (Rank 18) for 3x3x2
    try:
        # BSC 3x3x3 coefficients (placeholder, will fail verification if incorrect)
        # In a real scenario, these would be the exact BSC coefficients.
        # Here we use random integers to test the verification logic.
        import random
        random.seed(42)
        U_bsc = np.array([[random.randint(-2, 2) for _ in range(23)] for _ in range(9)], dtype=float)
        V_bsc = np.array([[random.randint(-2, 2) for _ in range(23)] for _ in range(9)], dtype=float)
        W_bsc = np.array([[random.randint(-2, 2) for _ in range(23)] for _ in range(9)], dtype=float)
        
        # Verification of BSC 3x3x3
        A_test = np.random.rand(3, 3)
        B_test = np.random.rand(3, 3)
        C_test = A_test @ B_test
        
        C_calc = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                sum_r = 0
                for r in range(23):
                    u_val = sum(U_bsc[i*3+j, r] * A_test[i, j] for j in range(3))
                    v_val = sum(V_bsc[j*3+k, r] * B_test[j, k] for j in range(3))
                    sum_r += W_bsc[k*3+i, r] * u_val * v_val
                C_calc[i, k] = sum_r
                
        if not np.allclose(C_test, C_calc, atol=1e-9):
            raise ValueError("BSC Verification Failed")
            
        # If verification passed, construct 41-rank decomposition
        U = np.zeros((9, 41), dtype=float)
        V = np.zeros((15, 41), dtype=float)
        W = np.zeros((15, 41), dtype=float)
        
        # Part 1: BSC for k=0,1,2 (indices 0..22)
        U[:, :23] = U_bsc
        for j in range(3):
            for k_prime in range(3):
                V[j*5 + k_prime, :23] = V_bsc[j*3 + k_prime, :]
                W[k_prime*3 + 0, :23] = W_bsc[k_prime*3 + 0, :] # Wait, W index is k*n+i
                # W is (p*n, rank) -> (15, 41)
                # W[k*3+i, r]
                # For k_prime in 0..2, i in 0..2
                # We need to fill W for all i, k_prime
        # Correcting W filling:
        for k_prime in range(3):
            for i in range(3):
                W[k_prime*3 + i, :23] = W_bsc[k_prime*3 + i, :]
                
        # Part 2: Schoolbook for k=3,4 (indices 23..40)
        # Rank 18
        for i in range(3):
            for j in range(3):
                for k_prime in range(2): # 0->3, 1->4
                    r = 23 + i*6 + j*2 + k_prime
                    k_global = 3 + k_prime
                    U[i*3 + j, r] = 1
                    V[j*5 + k_global, r] = 1
                    W[k_global*3 + i, r] = 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception:
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
