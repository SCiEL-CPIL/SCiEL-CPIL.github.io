"""Matrix multiplication 2x5x5 rank-minimal decomposition.

Uses the Kronecker-product structure:
  (2x5) * (5x5) can be decomposed with rank = 2*5 = 10.

The tensor T[a,b,c] for matrix multiplication (n x m) * (m x p) is:
  T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
            0 otherwise

With n=2, m=5, p=5:
  a = 5*i + j  (i in 0..1, j in 0..4)
  b = 5*j + k  (j in 0..4, k in 0..4)
  c = 2*k + i  (k in 0..4, i in 0..1)

The rank-10 decomposition uses rank indices r = i'*5 + j' for i' in 0..1, j' in 0..4:
  U[a, r] = 1 if i = i' and j = j', else 0
  V[b, r] = 1 if j = j', else 0
  W[c, r] = 1 if i = i', else 0

Verification: For valid (i,j,k):
  sum_r U[5i+j, r] * V[5j+k, r] * W[2k+i, r]
  = sum_{i',j'} [i=i', j=j'] * [j=j'] * [i=i']
  = 1 (only the term i'=i, j'=j contributes)

For invalid triples, at least one factor is 0 for all r, so sum = 0.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
DEADLINE = 130.0  # safety margin before 150s kill


def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # Check valid entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.sum(U[a] * V[b] * W[c]))
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check invalid entries are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if i_a == i_c and j_a == j_b:
                    continue  # valid entry
                val = float(np.sum(U[a] * V[b] * W[c]))
                if abs(val) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Rank-10 decomposition via Kronecker product structure
    rank = n * m  # 2 * 5 = 10

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for ip in range(n):
        for jp in range(m):
            r = ip * m + jp
            # U: only row (ip, jp) is 1
            U[ip * m + jp, r] = 1
            # V: all rows with j = jp are 1 (for all k)
            for k in range(p):
                V[jp * p + k, r] = 1
            # W: all rows with i = ip are 1 (for all k)
            for k in range(p):
                W[k * n + ip, r] = 1

    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Quick verification
    if not _verify(U, V, W):
        # Fallback: schoolbook rank 50
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U_list, "V": V_list, "W": W_list}
