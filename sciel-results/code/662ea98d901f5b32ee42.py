import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    start = time.time()
    deadline = start + 140  # 140s safety margin within 150s limit
    
    # Schoolbook rank-42 decomposition as fallback
    n, m, p = 2, 3, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using block structure
    # Split B into B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1)
    # C = A*B1 + A*B2 + A*B3 + A*B4
    # Each A*Bi where Bi is 3x2: use rank-11 decomposition
    # A*B4 where B4 is 3x1: use rank-6 decomposition
    # Total: 11*3 + 6 = 39
    
    # For 2x3 * 3x2 multiplication (n=2, m=3, p=2):
    # A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01],[b10,b11],[b20,b21]]
    # C = [[c00,c01],[c10,c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # We can use a Strassen-like decomposition for the 2x2 blocks after splitting
    # Split A into A1 (2x2) = [[a00,a01],[a10,a11]] and A2 (2x1) = [a02,a12]^T
    # Split B into B1 (2x2) = [[b00,b01],[b10,b11]] and B2 (1x2) = [b20,b21]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2*2x2, rank 7 (Strassen)
    # A2*B2: 2x1*1x2, rank 1 (outer product)
    # Total: 7 + 1 = 8? But we need to verify the exact decomposition.
    
    # Actually, let's use a simpler approach: direct sum of schoolbook for each block
    # But that gives 42 again. Let's use the 2x2*2x2 Strassen for the overlapping parts.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The search for a lower-rank decomposition is complex and risky.
    
    return {"U": U, "V": V, "W": W}
