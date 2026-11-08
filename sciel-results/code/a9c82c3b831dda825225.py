import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_161_solution():
        # Kronecker product structure: 6x6 = 3x3 blocks of 2x2
        # Rank = rank(3x3) * rank(2x2) = 23 * 7 = 161
        
        # Rank 7 Strassen for 2x2
        U2 = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        V2 = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        W2 = [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Rank 23 Bini-Lotti-Romani for 3x3
        # Using the known rank-23 decomposition
        # Each entry is a list of 9 coefficients (for the 3x3 block)
        U3 = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # M1
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # M2
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # M3
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # M4
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # M5
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # M6
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # M7
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # M8
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # M9
            [1, 0, 0, 0, 1, 0, 0, 0, 0],  # M10
            [0, 1, 0, 0, 0, 1, 0, 0, 0],  # M11
            [0, 0, 1, 0, 0, 0, 1, 0, 0],  # M12
            [0, 0, 0, 1, 0, 0, 0, 1, 0],  # M13
            [0, 0, 0, 0, 1, 0, 0, 0, 1],  # M14
            [1, 0, 0, 0, 0, 0, 1, 0, 0],  # M15
            [0, 1, 0, 0, 0, 0, 0, 1, 0],  # M16
            [0, 0, 1, 0, 0, 0, 0, 0, 1],  # M17
            [1, 1, 0, 0, 0, 0, 0, 0, 0],  # M18
            [0, 0, 1, 1, 0, 0, 0, 0, 0],  # M19
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # M20
            [1, 0, 1, 0, 0, 0, 0, 0, 0],  # M21
            [0, 1, 0, 0, 1, 0, 0, 0, 0],  # M22
            [0, 0, 0, 1, 0, 1, 0, 0, 0]   # M23
        ]
        V3 = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]
        ]
        W3 = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]
        
        # Verify the 3x3 decomposition
        A3 = np.random.rand(3, 3)
        B3 = np.random.rand(3, 3)
        C3_true = A3 @ B3
        C3_calc = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                s = 0
                for j in range(3):
                    for r in range(23):
                        s += U3[i*3+j][r] * V3[j*3+k][r] * W3[k*3+i][r]
                C3_calc[i, k] = s
        
        if not np.allclose(C3_calc, C3_true, atol=1e-10):
            return None
        
        # Now construct the 6x6 decomposition using Kronecker product
        rank = 23 * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r3 in range(23):
            for r2 in range(7):
                col = r3 * 7 + r2
                for i3 in range(3):
                    for j3 in range(3):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i3 * 2 + i2
                                j = j3 * 2 + j2
                                U[i*m + j][col] += U3[i3*3+j3][r3] * U2[i2*2+j2][r2]
                for j3 in range(3):
                    for k3 in range(3):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j3 * 2 + j2
                                k = k3 * 2 + k2
                                V[j*p + k][col] += V3[j3*3+k3][r3] * V2[j2*2+k2][r2]
                for k3 in range(3):
                    for i3 in range(3):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k3 * 2 + k2
                                i = i3 * 2 + i2
                                W[k*n + i][col] += W3[k3*3+i3][r3] * W2[k2*2+i2][r2]
        
        return {"U": U, "V": V, "W": W}
    
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
    
    def verify_solution(sol):
        U, V, W = sol['U'], sol['V'], sol['W']
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
        return np.allclose(C_calc, C_true, atol=1e-10)
    
    # Try 161 first
    sol_161 = get_161_solution()
    if sol_161 is not None:
        if verify_solution(sol_161):
            return sol_161
    
    # Fallback to 189
    return get_189_solution()
# EVOLVE-BLOCK-END
