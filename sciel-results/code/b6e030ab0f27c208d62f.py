"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

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
    def bsc_3x3x3():
        # Returns U, V, W for 3x3x3 with rank 23
        # If coefficients are not available or invalid, returns None
        return None

    try:
        bsc_factors = bsc_3x3x3()
        if bsc_factors is None:
            raise ValueError("BSC factors not available")
        
        U_bsc, V_bsc, W_bsc = bsc_factors
        
        U_full = np.zeros((9, 41), dtype=float)
        V_full = np.zeros((15, 41), dtype=float)
        W_full = np.zeros((15, 41), dtype=float)
        
        U_full[:, 0:23] = U_bsc
        
        for j in range(3):
            for k in range(3):
                idx_bsc = j * 3 + k
                idx_full = j * 5 + k
                V_full[idx_full, 0:23] = V_bsc[idx_bsc, :]
        
        W_full[0:9, 0:23] = W_bsc
        
        col = 23
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    U_full[i * 3 + j, col] = 1
                    V_full[j * 5 + k, col] = 1
                    W_full[k * 3 + i, col] = 1
                    col += 1
        
        # Verification
        for _ in range(5):
            A = np.random.rand(3, 3)
            B = np.random.rand(3, 5)
            C_true = A @ B
            C_calc = np.zeros((3, 5))
            for i in range(3):
                for k in range(5):
                    val = 0
                    for r in range(41):
                        u_val = sum(U_full[i * 3 + j, r] * A[i, j] for j in range(3))
                        v_val = sum(V_full[j * 5 + k, r] * B[j, k] for j in range(3))
                        val += W_full[k * 3 + i, r] * u_val * v_val
                    C_calc[i, k] = val
            if not np.allclose(C_true, C_calc, atol=1e-5):
                raise ValueError("Verification Failed")
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    except Exception:
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
