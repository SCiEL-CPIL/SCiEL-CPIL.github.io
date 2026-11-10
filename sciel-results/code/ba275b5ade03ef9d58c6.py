import time

def construct():
    n = 10
    squares = []
    
    # Construct 4 MOLS of order 10 using Z_5 x Z_2 decomposition
    # L_k(i, j) where i, j in 0..9
    # i = (i5, i2), j = (j5, j2)
    # L_k(i, j)5 = (i5 + k * j5) % 5  for k in {1, 2, 3, 4}
    # L_k(i, j)2 = (i2 + j2) % 2      (Slope 1 in Z_2)
    # Combine via CRT: x = x5 * 2 + x2 * 5 ? No.
    # CRT: x = x5 * (2 * inv(2,5)) + x2 * (5 * inv(5,2))
    # inv(2,5) = 3 (2*3=6=1). 2*3=6.
    # inv(5,2) = 1 (5=1). 5*1=5.
    # x = x5 * 6 + x2 * 5 mod 10.
    # Check:
    # x5=1, x2=0 -> 6. 6%5=1, 6%2=0. Correct.
    # x5=0, x2=1 -> 5. 5%5=0, 5%2=1. Correct.
    # x5=1, x2=1 -> 11%10=1. 1%5=1, 1%2=1. Correct.
    
    for k in range(1, 5):
        sq = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            i5 = i % 5
            i2 = i % 2
            for j in range(n):
                j5 = j % 5
                j2 = j % 2
                
                val5 = (i5 + k * j5) % 5
                val2 = (i2 + j2) % 2
                
                # Combine val5, val2 to x in 0..9
                # x = (val5 * 6 + val2 * 5) % 10
                x = (val5 * 6 + val2 * 5) % 10
                sq[i][j] = x
        squares.append(sq)
        
    return squares
