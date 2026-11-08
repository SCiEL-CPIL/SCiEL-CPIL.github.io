import time

DEADLINE = time.time() + 140.0

def construct():
    """
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer entries that exactly decompose the 6x6 matrix multiplication tensor.
    
    Strategy: Recursive Strassen (2x2) applied to 3x3 blocks, yielding rank 7^3 = 343.
    6x6 = 2x2 blocks of 3x3 matrices.
    Each 3x3 multiplication is further decomposed using a 3-term Strassen-like decomposition
    (since 3 = 1*3, we can't directly apply 2x2 Strassen, but we can use the fact that
    the rank of 3x3 matrix multiplication is at most 9, and we can achieve a decomposition
    with 3 rank-3 terms, each of which can be further decomposed).
    
    Actually, let me reconsider. The directive says rank 7^3 = 343, which suggests
    three levels of 7-term Strassen. Since 6 = 2*3, and 3 is not a power of 2,
    we can't directly apply Strassen three times.
    
    Alternative: Use schoolbook for 6x6, which gives rank 216. This is guaranteed correct.
    """
    n, m, p = 6, 6, 6
    
    # Use schoolbook decomposition: rank = n*m*p = 216
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row i, col j of A)
                # V index: j*p + k (row j, col k of B)
                # W index: k*n + i (col k, row i of C, using trace convention W[k*n+i])
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
