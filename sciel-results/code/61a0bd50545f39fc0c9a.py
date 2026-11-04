"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's 7-rank decomposition for 2x2 matrix multiplication
    # U2: 4x7 (input A), V2: 4x7 (input B), W2: 4x7 (output C)
    # Indexing: U2[i*2+j, r] for A[i,j], V2[j*2+k, r] for B[j,k], W2[k*2+i, r] for C[i,k]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker products for 4x4
    # Rank = 7 * 7 = 49
    U4 = np.kron(U2, U2)  # 16x49
    V4 = np.kron(V2, V2)  # 16x49
    W4 = np.kron(W2, W2)  # 16x49
    
    # Permutation to map Kronecker indices to Linear Row-Major indices
    # For U (A matrix): row index S = r*4 + c, where r = r_out*2 + r_in, c = c_out*2 + c_in
    # Kronecker row index K = L2_out*4 + L2_in where L2_out = r_out*2 + c_out, L2_in = r_in*2 + c_in
    # We need U4_fixed[S, :] = U4[K, :]
    
    def make_permutation():
        """Create permutation array mapping standard row-major index to Kronecker index."""
        perm = np.zeros(16, dtype=int)
        for S in range(16):
            r = S // 4
            c = S % 4
            r_out = r // 2
            r_in = r % 2
            c_out = c // 2
            c_in = c % 2
            
            L2_out = r_out * 2 + c_out
            L2_in = r_in * 2 + c_in
            
            K = L2_out * 4 + L2_in
            perm[S] = K
        return perm
    
    perm = make_permutation()
    
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Verify the decomposition with a quick test
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, (4, 4)).astype(float)
    B_test = np.random.randint(-5, 6, (4, 4)).astype(float)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            total = 0.0
            for r in range(49):
                for j in range(4):
                    total += U4_fixed[i*4+j, r] * V4_fixed[j*4+k, r] * W4_fixed[k*4+i, r]
            C_reconstructed[i, k] = total
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
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
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
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
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
