"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen():
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: i=0, schoolbook
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
            
        # Part 2: i=1,2, Strassen
        # Strassen 2x2x2 decomposition
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We apply this to 2x4x8 blocks.
        # The 2x4x8 block is composed of 4 2x2x2 blocks?
        # No, 2x4x8 is 2x(2+2)x(4+4).
        # It is 4 blocks of 2x2x2 in the inner and outer dimensions?
        # A is 2x4. B is 4x8.
        # Split A into A1 (2x2) and A2 (2x2).
        # Split B into B11, B12, B21, B22 (each 2x4? No, B is 4x8).
        # B = [B11 B12; B21 B22] where each is 2x4.
        # C = A1 B11 + A2 B21 + A1 B12 + A2 B22? No.
        # C = [A1 A2] [B11; B21] + [A1 A2] [B12; B22]
        # C_left = A1 B11 + A2 B21 (2x4)
        # C_right = A1 B12 + A2 B22 (2x4)
        # Each 2x4 output is formed by 2x2 * 2x4.
        # 2x2 * 2x4 is two 2x2 * 2x2 multiplications.
        # So 4 2x2 * 2x2 multiplications total.
        # Each 2x2 * 2x2 has rank 7.
        # Total rank 28.
        
        strassen_U = [
            [1, 0, 0, 1], 
            [0, 0, 1, 1], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0], 
            [0, 1, 0, -1]  
        ]
        strassen_V = [
            [1, 0, 0, 1], 
            [1, 0, 0, 0], 
            [0, 1, 0, -1], 
            [-1, 0, 1, 0], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [0, 0, 1, 1]  
        ]
        strassen_W = [
            [1, 0, 0, 1], 
            [0, 0, 1, -1], 
            [0, 1, 0, 1], 
            [1, 0, 1, 0], 
            [-1, 1, 0, 0], 
            [0, 0, 0, 1], 
            [1, 0, 0, 0]  
        ]
        
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                for t in range(7):
                    for l in range(4):
                        if strassen_U[t][l] != 0:
                            local_i_prime = l // 2
                            local_j_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_j = j_start + local_j_prime
                            U[global_i * m + global_j][col] += strassen_U[t][l]
                            
                    for l in range(4):
                        if strassen_V[t][l] != 0:
                            local_j_prime = l // 2
                            local_k_prime = l % 2
                            global_j = j_start + local_j_prime
                            global_k = k_start + local_k_prime
                            V[global_j * p + global_k][col] += strassen_V[t][l]
                            
                    for l in range(4):
                        if strassen_W[t][l] != 0:
                            local_i_prime = l // 2
                            local_k_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_k = k_start + local_k_prime
                            W[global_k * n + global_i][col] += strassen_W[t][l]
                            
                    col += 1
        
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]
        
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if time.time() < deadline - 10:
        res = build_strassen()
        if res and verify(res):
            return res
            
    return build_schoolbook()
