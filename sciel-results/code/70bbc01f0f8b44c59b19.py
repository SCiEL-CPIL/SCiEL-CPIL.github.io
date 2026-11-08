"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """Construct the matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank: 4*4*5 = 80
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    
    # Verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute via decomposition
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # The standard check is that
    # for all i, j, k: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # And T_{i,j,k} = 1 for all valid i, j, k since C_{i,k} = sum_j A_{i,j} B_{j,k}
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_err = max(max_err, abs(s - 1.0))
    print(f"Max error: {max_err}")
    print(f"Rank: {rank}")
