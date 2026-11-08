import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. Attempt rank 161 decomposition using Kronecker product of 
    #    Rank 7 (2x2) and Rank 23 (3x3) decompositions.
    # 2. Verify the decomposition.
    # 3. If valid, return rank 161.
    # 4. Else, fallback to rank 189 (3x3 blocks of 2x2 using Strassen).
    
    # --- Data for Rank 7 (2x2 Strassen) ---
    # Flattened indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # --- Data for Rank 23 (3x3) ---
    # Flattened indices: 0..8 correspond to (0,0)..(2,2)
    # Using Bini-Lotti-Romani (1980) Rank 23 coefficients.
    # If these are incorrect, verification will fail and fallback to 189.
    
    U3 = [
        [1, 1, 0, 0, 1, 1, 0, 0, 1], # 0
        [0, 0, 1, 1, 0, 0, 1, 1, 0], # 1
        [1, 0, 0, 1, 1, 0, 0, 1, 1], # 2
        [0, 1, 1, 0, 0, 1, 1, 0, 0], # 3
        [1, 1, 0, 0, 0, 0, 1, 1, 0], # 4
        [0, 0, 1, 1, 0, 0, 0, 0, 1], # 5
        [1, 0, 0, 1, 0, 0, 0, 0, 0], # 6
        [0, 1, 1, 0, 0, 0, 0, 0, 0], # 7
        [1, 1, 0, 0, 0, 0, 0, 0, 0], # 8
        # ... I don't have the full 23 rows memorized correctly. 
        # I will return None for U3 to force fallback if I can't guarantee correctness.
        # However, I can try to use a known valid set from a library if available? No.
        # I will use a placeholder that is likely wrong to test the fallback logic, 
        # but actually I should just return None to be safe.
    ]
    
    # Since I cannot guarantee the Rank 23 coefficients from memory, 
    # I will return None for the 161 solution.
    
    def get_189_solution():
        block_size = 2
        blocks_per_dim = 3
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

    def get_161_solution():
        # I will return None as I don't have the 23 coefficients.
        return None

    sol_161 = get_161_solution()
    if sol_161 is not None:
        U, V, W = sol_161['U'], sol_161['V'], sol_161['W']
        rank = len(U[0])
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                C_calc[i, k] = s
                
        if np.allclose(C_calc, C_true):
            return sol_161

    return get_189_solution()
# EVOLVE-BLOCK-END
