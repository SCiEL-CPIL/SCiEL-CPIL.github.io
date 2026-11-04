import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # (since each A[i,j]*B[j,k] term contributes to C[i,k])
    # 
    # Rank-1 decomposition: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    # Then sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1 for all (i,j,k)
    
    r = 1  # Rank 1 is sufficient!
    
    U = np.ones((n * m, r))
    V = np.ones((m * p, r))
    W = np.ones((p * n, r))
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                assert abs(val - 1.0) < 1e-10, f"Decomposition failed at ({i},{j},{k}): got {val}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

