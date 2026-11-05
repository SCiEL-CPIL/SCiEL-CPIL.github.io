import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the (n,m,p) matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[a, b, c] where a=(i,j), b=(j,k), c=(k,i)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_r U[a,r] * V[b,r] * W[c,r] == T[a,b,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 1:
                    return False
    # Also check that off-diagonal entries are 0
    # T[a,b,c] = 0 unless a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # i.e., the middle index of a equals the first index of b, and the second index of b equals the first index of c, and the second index of c equals the first index of a
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                # Should be nonzero only if j_a == j_b and k_b == k_c and i_c == i_a
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True

def construct():
    n, m, p = 3, 3, 7
    # Schoolbook decomposition: rank = n*m*p = 63
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
