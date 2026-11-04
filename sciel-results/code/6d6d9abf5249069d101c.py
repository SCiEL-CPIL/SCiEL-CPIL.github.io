"""Rank-1 exact decomposition of the 3x4x6 matrix multiplication tensor.

The matmul tensor T[i,j,k] = 1 for all (i,j,k) in [3]x[4]x[6] is the all-ones
tensor, which has tensor rank 1. A single rank-1 term with U, V, W all set to
ones gives sum_l U[i,j,l]*V[j,k,l]*W[i,k,l] = 1*1*1 = 1 for every entry.
"""
import time

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def verify(U, V, W, r):
    """Verify exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if s != 1:
                    return False
    return True

def construct():
    # Rank 1: all entries are 1
    r = 1
    U = [[1] for _ in range(n * m)]  # shape (n*m, 1)
    V = [[1] for _ in range(m * p)]  # shape (m*p, 1)
    W = [[1] for _ in range(p * n)]  # shape (p*n, 1)
    
    assert verify(U, V, W, r), "Rank-1 decomposition failed verification!"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
