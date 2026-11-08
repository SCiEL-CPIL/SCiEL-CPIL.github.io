"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Entries are integers.
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    
    # Schoolbook decomposition with rank n*m*p
    rank = n * m * p
    num_U = n * m  # 12
    num_V = m * p  # 20
    num_W = p * n  # 15
    
    U = [[0] * rank for _ in range(num_U)]
    V = [[0] * rank for _ in range(num_V)]
    W = [[0] * rank for _ in range(num_W)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify correctness (optional, for debugging)
    # if _verify_decomposition(U, V, W, n, m, p):
    #     pass
    # else:
    #     raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            # Compute sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
                            total = 0
                            for r in range(rank):
                                total += U[i * m + j][r] * V[j2 * p + k][r] * W[k2 * n + i2][r]
                            # Expected: 1 if j==j2 and k==k2 and i==i2, else 0
                            expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                            if total != expected:
                                return False
    return True
# EVOLVE-BLOCK-END
