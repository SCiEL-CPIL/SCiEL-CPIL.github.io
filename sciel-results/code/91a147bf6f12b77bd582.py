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
        
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We need to sum over j and r.
        # C[i, k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Efficient check:
        # Construct tensor T from U, V, W
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This should be 1 for all i, j, k.
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :]) # Wait, W is missing?
                    
        # The formula is sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # This is not a simple dot product of U and V.
        # It is sum_r (U[u_idx, r] * V[v_idx, r]) * W[w_idx, r]
        
        # Let's just compute it directly for verification
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

    # Try Strassen
    if time.time() < deadline - 10:
        res = build_strassen()
        if res and verify(res):
            return res
            
    # Fallback
    return build_schoolbook()
