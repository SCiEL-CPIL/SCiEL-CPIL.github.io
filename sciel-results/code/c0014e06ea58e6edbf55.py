import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Trivial decomposition (Rank 45)
    rank_trivial = n * m * p
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
                
    # Verify trivial decomposition
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Should be 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    # We check a few random points and the specific points.
    def verify(U, V, W, rank):
        # Check all specific points
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        # Check a few random "off-diagonal" points
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            # If (a,b,c) is not a consistent triple, val should be 0
            # Consistent if:
            # i = a // m, j = a % m
            # j' = b // p, k = b % p
            # k' = c // n, i' = c % n
            # i == i' and j == j' and k == k'
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            is_consistent = (i == i2 and j == j2 and k == k2)
            val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            if is_consistent:
                if val != 1:
                    return False
            else:
                if val != 0:
                    return False
        return True

    if not verify(U_triv, V_triv, W_triv, rank_trivial):
        # Should not happen
        return {"U": U_triv, "V": V_triv, "W": W_triv}

    return {"U": U_triv, "V": V_triv, "W": W_triv}
