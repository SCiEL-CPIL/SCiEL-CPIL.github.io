"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # U2[i,c], V2[j,c], W2[k,c] where i=(r1,c1), j=(c1',r2), k=(r2',c2)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # a11
        [0, 0, 1, 0, 1, 0, -1],   # a12
        [0, 1, 0, 0, 0, 1, 0],    # a21
        [1, 1, 0, 1, 0, 0, 0]     # a22
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # b11
        [0, 0, 1, 0, 0, 1, 0],    # b12
        [0, 0, -1, 0, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 1]    # b22
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ], dtype=float)
    
    # Schoolbook rank-27 decomposition for 3x3 matrix multiplication
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, c] = 1.0
                V3[j*3+k, c] = 1.0
                W3[k*3+i, c] = 1.0
                c += 1
    
    # Verify Strassen decomposition
    def verify_mm(U, V, W, size):
        for i in range(size*size):
            r1, c1 = divmod(i, size)
            for j in range(size*size):
                c1p, r2 = divmod(j, size)
                if c1 != c1p:
                    continue
                for k in range(size*size):
                    r2p, c2 = divmod(k, size)
                    expected = 1.0 if r2 == r2p else 0.0
                    actual = np.dot(np.multiply(U[i,:], V[j,:]), W[k,:])
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    if not verify_mm(U2, V2, W2, 2):
        # Should not happen, but fallback
        return _schoolbook_6x6()
    
    if not verify_mm(U3, V3, W3, 3):
        return _schoolbook_6x6()
    
    # Tensor product: MM_6 = MM_2 ⊗ MM_3
    # Index: i = i1*3 + i2, where i1 in {0,1} (for MM_2), i2 in {0,1,2} (for MM_3)
    rank2, rank3 = 7, 27
    total_rank = rank2 * rank3  # 189
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    for i in range(36):
        i1, i2 = divmod(i, 3)
        for j in range(36):
            j1, j2 = divmod(j, 3)
            for k in range(36):
                k1, k2 = divmod(k, 3)
                for a in range(rank2):
                    for b in range(rank3):
                        c = a * rank3 + b
                        U6[i, c] = U2[i1, a] * U3[i2, b]
                        V6[j, c] = V2[j1, a] * V3[j2, b]
                        W6[k, c] = W2[k1, a] * W3[k2, b]
    
    # Verify MM6 decomposition (spot check)
    def verify_mm6(U, V, W, num_checks=500):
        import random
        random.seed(123)
        for _ in range(num_checks):
            i = random.randint(0, 35)
            j = random.randint(0, 35)
            k = random.randint(0, 35)
            r1, c1 = divmod(i, 6)
            c1p, r2 = divmod(j, 6)
            r2p, c2 = divmod(k, 6)
            expected = 1.0 if (c1 == c1p and r2 == r2p) else 0.0
            actual = np.dot(np.multiply(U[i,:], V[j,:]), W[k,:])
            if abs(actual - expected) > 1e-10:
                return False
        return True
    
    if not verify_mm6(U6, V6, W6):
        return _schoolbook_6x6()
    
    # Convert to lists with integer/half-integer entries
    U_list = [[round(x * 2) / 2 for x in row] for row in U6.tolist()]
    V_list = [[round(x * 2) / 2 for x in row] for row in V6.tolist()]
    W_list = [[round(x * 2) / 2 for x in row] for row in W6.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_6x6():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
