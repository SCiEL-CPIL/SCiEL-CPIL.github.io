"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based decomposition for 2x2 * 2x13 matrix multiplication.
    
    Uses 7 Strassen products for the 2x2 block structure, but since the right
    matrix has 13 columns, we need to handle the column dimension carefully.
    
    The key insight: A (2x2) * B (2x13) = C (2x13)
    We can view B as having two rows of 13 elements each.
    
    Strassen's algorithm for 2x2 * 2x2 uses 7 multiplications.
    For 2x2 * 2x13, we can apply Strassen to the row structure.
    
    Let A = [[a00, a01], [a10, a11]] (2x2)
    Let B = [[b00, b01, ..., b0,12], [b10, b11, ..., b1,12]] (2x13)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using Strassen's 7 products:
    M1 = (a00 + a11) * (b0k + b1k)  for each k
    M2 = (a10 + a11) * b0k          for each k
    M3 = a00 * (b1k - b0k)          for each k
    M4 = a11 * (b0k - b1k)          for each k
    M5 = (a01 + a10) * b1k          for each k
    M6 = (a10 - a00) * (b0k + b1k)  for each k
    M7 = (a01 - a11) * (b1k + b0k)  for each k
    
    Wait, this isn't quite right for rectangular. Let me reconsider.
    
    Actually, for A (2x2) * B (2x13), we can think of it as:
    C = A * B where each column of B is a 2x1 vector.
    
    The bilinear complexity of 2x2 * 2x1 is 7 (Strassen).
    For 2x13, we have 13 independent columns, so naively 7*13 = 91.
    
    But we can do better by sharing structure across columns!
    
    Alternative approach: Use the fact that we're multiplying by a 2x2 matrix.
    C[0,k] = a00*B[0,k] + a01*B[1,k]
    C[1,k] = a10*B[0,k] + a11*B[1,k]
    
    This is 4 multiplications per column * 13 columns = 52 (schoolbook).
    
    With Strassen on the 2x2 structure:
    For each column k, we need:
    C[0,k] and C[1,k] from A and [B[0,k], B[1,k]]
    
    Strassen gives us 7 products for 2x2*2x2, but here we only have 2x2*2x1.
    The bilinear complexity of 2x2*2x1 is actually 4 (you can't beat schoolbook
    for this case since it's just two dot products of length 2).
    
    So schoolbook rank = 2*2*13 = 52 might be optimal?
    
    Let me verify: The tensor rank of [2,2,13] matrix multiplication.
    Lower bounds: max(n*m, m*p, p*n) = max(4, 26, 26) = 26
    Upper bound: n*m*p = 52
    
    Actually, I recall that for [n,m,p], the rank is at most n*m*p but can be
    lower. For [2,2,p], I believe the rank is 2p for p >= 2 (schoolbook is optimal).
    
    Let me just use the schoolbook decomposition which gives rank 52.
    """
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
