import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 fallback
    def get_strassen_factors():
        block_size = 2
        blocks_per_dim = 3 # 6 / 2
        rank = (blocks_per_dim ** 3) * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        col_idx += 1
        return {"U": U, "V": V, "W": W}

    # Laderman 3x3 attempt
    def get_laderman_factors():
        block_size = 3
        blocks_per_dim = 2 # 6 / 3
        rank = (blocks_per_dim ** 3) * 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Laderman's algorithm coefficients
        # Format: (u_dict, v_dict, w_dict)
        # Indices are 0-based
        laderman_patterns = [
            # M1
            ({(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
            # M2
            ({(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,0): 1}),
            # M3
            ({(1,1): 1, (1,2): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}),
            # M4
            ({(0,0): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M5
            ({(1,1): 1, (2,2): -1}, {(1,1): 1, (2,2): 1}, {(1,1): 1, (2,2): 1}),
            # M6
            ({(0,0): 1, (0,1): 1, (1,1): -1, (1,2): -1}, {(1,0): 1, (1,1): 1, (2,0): 1, (2,1): 1}, {(0,1): 1, (1,0): 1, (0,0): 1, (1,1): 1}), # Guess
            # M7
            ({(0,0): 1, (0,2): 1, (1,1): -1, (1,2): -1}, {(2,0): 1, (2,1): 1, (0,0): 1, (0,1): 1}, {(1,0): 1, (0,0): 1, (1,1): 1, (0,1): 1}), # Guess
            # M8
            ({(0,1): 1, (1,1): 1, (2,2): 1}, {(1,0): 1, (0,0): 1, (2,2): 1}, {(0,1): 1, (1,1): 1, (2,0): 1}), # Guess
            # M9
            ({(0,2): 1, (1,2): 1, (2,2): 1}, {(2,0): 1, (0,0): 1, (1,1): 1}, {(0,2): 1, (1,0): 1, (1,1): 1}), # Guess
            # M10
            ({(0,0): 1, (0,1): 1, (0,2): 1}, {(1,1): 1, (1,2): 1, (2,2): 1}, {(0,1): 1, (0,2): 1, (1,1): 1}), # Guess
            # M11
            ({(1,0): 1, (1,1): 1, (1,2): 1}, {(0,0): 1, (0,2): 1, (2,2): 1}, {(1,0): 1, (1,2): 1, (2,0): 1}), # Guess
            # M12
            ({(2,0): 1, (2,1): 1, (2,2): 1}, {(0,0): 1, (0,1): 1, (1,1): 1}, {(2,0): 1, (2,1): 1, (1,1): 1}), # Guess
            # M13
            ({(0,0): 1, (1,0): 1}, {(0,2): 1, (1,2): 1}, {(0,2): 1, (1,2): 1}), # Guess
            # M14
            ({(1,1): 1, (2,1): 1}, {(0,2): 1, (2,2): 1}, {(1,2): 1, (2,2): 1}), # Guess
            # M15
            ({(0,0): 1, (2,2): -1}, {(0,2): 1, (2,2): 1}, {(0,2): 1, (2,2): 1}), # Guess
            # M16
            ({(1,1): 1, (0,0): -1}, {(1,2): 1, (0,2): 1}, {(1,2): 1, (0,2): 1}), # Guess
            # M17
            ({(2,2): 1, (1,1): -1}, {(2,0): 1, (1,0): 1}, {(2,0): 1, (1,0): 1}), # Guess
            # M18
            ({(0,0): 1, (2,2): 1, (1,1): -1}, {(0,0): 1, (2,2): 1, (1,1): -1}, {(0,0): 1, (2,2): 1, (1,1): 1}), # Guess
            # M19
            ({(0,1): 1, (2,1): 1, (1,1): -1}, {(1,1): 1, (0,1): 1, (2,1): -1}, {(0,1): 1, (2,1): 1, (1,1): 1}), # Guess
            # M20
            ({(0,2): 1, (2,2): 1, (1,2): -1}, {(2,2): 1, (0,2): 1, (1,2): -1}, {(0,2): 1, (2,2): 1, (1,2): 1}), # Guess
            # M21
            ({(1,0): 1, (0,0): 1, (2,0): -1}, {(0,0): 1, (1,0): 1, (2,0): -1}, {(1,0): 1, (0,0): 1, (2,0): 1}), # Guess
            # M22
            ({(2,1): 1, (1,1): 1, (0,1): -1}, {(1,1): 1, (2,1): 1, (0,1): -1}, {(2,1): 1, (1,1): 1, (0,1): 1}), # Guess
            # M23
            ({(2,2): 1, (0,2): 1, (1,2): -1}, {(2,2): 1, (1,2): 1, (0,2): -1}, {(2,2): 1, (0,2): 1, (1,2): 1}), # Guess
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in laderman_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        col_idx += 1
        return {"U": U, "V": V, "W": W}

    def verify_factors(factors):
        U = np.array(factors["U"])
        V = np.array(factors["V"])
        W = np.array(factors["W"])
        
        # Random test matrices
        np.random.seed(42)
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C = A @ B
        
        # Compute using factors
        # C_ik = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} A_{ij}) * (sum_j V_{j*p+k, r} B_{jk})
        # Vectorized computation
        # U is (n*m, rank). Reshape to (n, m, rank)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # u_prod[i, r] = sum_j U[i, j, r] * A[i, j]
        # Using einsum: 'ijr,ij->ir'
        u_prod = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        # v_prod[k, r] = sum_j V[j, k, r] * B[j, k]
        # Using einsum: 'jkr,jk->kr'
        v_prod = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        # C_calc[i, k] = sum_r W[k, i, r] * u_prod[i, r] * v_prod[k, r]
        # W is (p, n, rank) -> (k, i, r)
        # u_prod is (n, rank) -> (i, r)
        # v_prod is (p, rank) -> (k, r)
        
        # We need sum_r W[k, i, r] * u_prod[i, r] * v_prod[k, r]
        # Let's compute term[i, k, r] = W[k, i, r] * u_prod[i, r] * v_prod[k, r]
        # This is tricky with broadcasting.
        
        # u_prod[i, r] -> shape (n, 1, rank)
        # v_prod[k, r] -> shape (1, p, rank)
        # W[k, i, r] -> shape (p, n, rank)
        
        # term = W_reshaped * u_prod[:, np.newaxis, :] * v_prod[np.newaxis, :, :]
        # But W is (p, n, rank), u is (n, rank), v is (p, rank).
        # u_prod[:, np.newaxis, :] is (n, 1, rank). Broadcasting to (p, n, rank) works if we align n with axis 1.
        # v_prod[np.newaxis, :, :] is (1, p, rank). Broadcasting to (p, n, rank) works if we align p with axis 0.
        
        term = W_reshaped * u_prod[:, np.newaxis, :] * v_prod[np.newaxis, :, :]
        
        # Sum over rank (axis 2)
        C_calc = np.sum(term, axis=2)
        
        return np.allclose(C_calc, C, atol=1e-6)

    # Try Laderman first
    try:
        laderman_factors = get_laderman_factors()
        if verify_factors(laderman_factors):
            return laderman_factors
    except Exception:
        pass
        
    # Fallback to Strassen
    return get_strassen_factors()
# EVOLVE-BLOCK-END
