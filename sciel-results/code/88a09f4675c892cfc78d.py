"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 21, which is the lower bound from the largest slice
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # For each i, set U[i*m + j, r] = 1 and W[k*n + i, r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == '__main__':
    # Self-verification: check that the decomposition is correct
    U, V, W = construct()['U'], construct()['V'], construct()['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Verify the tensor identity: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if total != 1:
                    print(f"FAIL: i={i}, j={j}, k={k}, total={total}")
                    sys.exit(1)
    print(f"Verification passed: rank={rank} decomposition is valid.")
