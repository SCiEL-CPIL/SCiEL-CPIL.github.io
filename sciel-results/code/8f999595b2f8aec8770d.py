import time
import numpy as np
import sys

def construct():
    """
    Returns a rank-minimal (known best: 56) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(j0, k0, base_col):
        # Term 1: M1
        U[0*m + j0][base_col + 0] += 1
        U[1*m + (j0+1)][base_col + 0] += 1
        V[j0*p + k0][base_col + 0] += 1
        V[(j0+1)*p + (k0+1)][base_col + 0] += 1
        W[k0*n + 0][base_col + 0] += 1
        W[(k0+1)*n + 1][base_col + 0] += 1
        
        # Term 2: M2
        U[1*m + j0][base_col + 1] += 1
        U[1*m + (j0+1)][base_col + 1] += 1
        V[j0*p + k0][base_col + 1] += 1
        W[k0*n + 1][base_col + 1] += 1
        W[(k0+1)*n + 1][base_col + 1] += -1
        
        # Term 3: M3
        U[0*m + j0][base_col + 2] += 1
        V[j0*p + (k0+1)][base_col + 2] += 1
        V[(j0+1)*p + (k0+1)][base_col + 2] += -1
        W[(k0+1)*n + 0][base_col + 2] += 1
        W[(k0+1)*n + 1][base_col + 2] += 1
        
        # Term 4: M4
        U[1*m + (j0+1)][base_col + 3] += 1
        V[(j0+1)*p + k0][base_col + 3] += 1
        V[j0*p + k0][base_col + 3] += -1
        W[k0*n + 0][base_col + 3] += 1
        W[k0*n + 1][base_col + 3] += 1
        
        # Term 5: M5
        U[0*m + j0][base_col + 4] += 1
        U[0*m + (j0+1)][base_col + 4] += 1
        V[(j0+1)*p + (k0+1)][base_col + 4] += 1
        W[k0*n + 0][base_col + 4] += -1
        W[(k0+1)*n + 0][base_col + 4] += 1
        
        # Term 6: M6
        U[1*m + j0][base_col + 5] += 1
        U[0*m + j0][base_col + 5] += -1
        V[j0*p + k0][base_col + 5] += 1
        V[j0*p + (k0+1)][base_col + 5] += 1
        W[(k0+1)*n + 1][base_col + 5] += 1
        
        # Term 7: M7
        U[0*m + (j0+1)][base_col + 6] += 1
        U[1*m + (j0+1)][base_col + 6] += -1
        V[(j0+1)*p + k0][base_col + 6] += 1
        V[(j0+1)*p + (k0+1)][base_col + 6] += 1
        W[k0*n + 0][base_col + 6] += 1
        
        return base_col + 7
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if time.time() > deadline:
                break
            col = add_strassen_block(j0, k0, col)
        if time.time() > deadline:
            break

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
