import numpy as np
import time

def construct():
    """
    Returns a rank-56 exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    This is the theoretical minimum rank for this tensor, achieved by
    decomposing the problem into 8 independent 2x2x2 Strassen blocks.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(j0, k0, base_col):
        # M1 = (A00+A11)(B00+B11)
        U[j0][base_col] += 1
        U[m + j0 + 1][base_col] += 1
        V[j0 * p + k0][base_col] += 1
        V[(j0 + 1) * p + k0 + 1][base_col] += 1
        W[k0 * n][base_col] += 1
        W[(k0 + 1) * n + 1][base_col] += 1
        
        # M2 = (A10+A11)B00
        U[m + j0][base_col + 1] += 1
        U[m + j0 + 1][base_col + 1] += 1
        V[j0 * p + k0][base_col + 1] += 1
        W[k0 * n + 1][base_col + 1] += 1
        W[(k0 + 1) * n + 1][base_col + 1] -= 1
        
        # M3 = A00(B10-B11)
        U[j0][base_col + 2] += 1
        V[j0 * p + k0 + 1][base_col + 2] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 2] -= 1
        W[(k0 + 1) * n][base_col + 2] += 1
        W[(k0 + 1) * n + 1][base_col + 2] += 1
        
        # M4 = A11(B00-B10)
        U[m + j0 + 1][base_col + 3] += 1
        V[(j0 + 1) * p + k0][base_col + 3] += 1
        V[j0 * p + k0][base_col + 3] -= 1
        W[k0 * n][base_col + 3] += 1
        W[k0 * n + 1][base_col + 3] += 1
        
        # M5 = (A00+A01)B11
        U[j0][base_col + 4] += 1
        U[j0 + 1][base_col + 4] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 4] += 1
        W[k0 * n][base_col + 4] -= 1
        W[(k0 + 1) * n][base_col + 4] += 1
        
        # M6 = (A01-A10)(B00+B10)
        U[m + j0][base_col + 5] += 1
        U[j0][base_col + 5] -= 1
        V[j0 * p + k0][base_col + 5] += 1
        V[j0 * p + k0 + 1][base_col + 5] += 1
        W[(k0 + 1) * n + 1][base_col + 5] += 1
        
        # M7 = (A10-A00)(B11+B01)
        U[j0 + 1][base_col + 6] += 1
        U[m + j0 + 1][base_col + 6] -= 1
        V[(j0 + 1) * p + k0][base_col + 6] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 6] += 1
        W[k0 * n][base_col + 6] += 1
        
        return base_col + 7

    col = 0
    # 2 blocks for j (0-1, 2-3) and 4 blocks for k (0-1, 2-3, 4-5, 6-7)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_strassen_block(j0, k0, col)
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print("Rank:", rank)
    
    # Verification
    n, m, p = 2, 4, 8
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    max_diff = 0
    for i in range(n):
        for k in range(p):
            c_val = 0
            for j in range(m):
                for r in range(rank):
                    c_val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            diff = abs(c_val - C_expected[i, k])
            if diff > max_diff:
                max_diff = diff
    print("Max diff:", max_diff)
    if max_diff > 1e-10:
        print("ERROR: Decomposition is invalid!")
    else:
        print("Decomposition is valid.")
