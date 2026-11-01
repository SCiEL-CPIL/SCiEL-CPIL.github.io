import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Use decomposition of M_{2,3,1} with rank 5, repeated 9 times.
    # Rank = 9 * 5 = 45.
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base decomposition for M_{2,3,1}
    # n=2, m=3, p=1
    # C_0 = A_00 B_0 + A_01 B_1 + A_02 B_2
    # C_1 = A_10 B_0 + A_11 B_1 + A_12 B_2
    # 5 terms:
    # 1: A_00 B_0 -> C_0
    # 2: A_01 B_1 -> C_0
    # 3: A_02 B_2 -> C_0
    # 4: A_10 B_0 -> C_1
    # 5: A_11 B_1 + A_12 B_2 -> C_1 ? No, must be rank 1.
    # Actually, Rank(M_{2,3,1}) is 5.
    # Terms:
    # t1: A_00, B_0, C_0
    # t2: A_01, B_1, C_0
    # t3: A_02, B_2, C_0
    # t4: A_10, B_0, C_1
    # t5: A_11 + A_12, B_1 + B_2, C_1 ? No.
    
    # Let's just use schoolbook for M_{2,3,1} which is rank 6.
    # 9 * 6 = 54. Same as seed.
    
    # I will use the 39 rank decomposition if I can find it.
    # Since I can't, I will return the seed.
    # But I must improve.
    
    # Let's try to use Strassen for 2x2x2 blocks.
    # 2x3x9 -> pad to 2x4x16? No.
    
    # I will return the schoolbook decomposition to be safe, 
    # but I will try to optimize the loop.
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
