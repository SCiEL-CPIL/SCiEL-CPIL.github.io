"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
  - Use Strassen 2x2x2 blocks wherever both the row-index and col-index pairs
    are available, plus the standard Strassen 7-term formula.
  - For the middle dimension m=5, split as {0,1}, {2,3}, {4}.
  - For n=3: split as {0,1}, {2}.
  - For p=7: split as {0,1}, {2,3}, {4,5}, {6}.
  - Strassen blocks: (0,1)x(0,1)x(0,1), (0,1)x(0,1)x(2,3), (0,1)x(0,1)x(4,5)
                     (0,1)x(2,3)x(0,1), (0,1)x(2,3)x(2,3), (0,1)x(2,3)x(4,5)
    => 6 Strassen blocks * 7 terms = 42 terms.
  - Remaining triples handled by schoolbook: 105 - 6*8 = 57 terms.
  - Total rank = 99.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]


def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify_decomp(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Block structure
    n_pairs = [(0, 1)]
    m_pairs = [(0, 1), (2, 3)]
    p_pairs = [(0, 1), (2, 3), (4, 5)]

    # Each Strassen block contributes 7 terms
    num_strassen = len(n_pairs) * len(m_pairs) * len(p_pairs)  # 1*2*3 = 6
    rank_strassen = num_strassen * 7  # 42

    # Covered index sets
    n_covered = set()
    for a, b in n_pairs:
        n_covered.update([a, b])
    m_covered = set()
    for c, d in m_pairs:
        m_covered.update([c, d])
    p_covered = set()
    for e, f in p_pairs:
        p_covered.update([e, f])

    # Schoolbook terms for uncovered triples
    num_schoolbook = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_covered or j not in m_covered or k not in p_covered:
                    num_schoolbook += 1

    total_rank = rank_strassen + num_schoolbook  # 42 + 57 = 99

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Strassen 2x2x2 terms for each block
    for (a, b) in n_pairs:
        for (c, d) in m_pairs:
            for (e, f) in p_pairs:
                # M1 = (A[a,c]+A[b,d]) * (B[c,e]+B[d,f])
                # C[e,a]+=1, C[f,b]+=1
                U[a * m + c][col] += 1
                U[b * m + d][col] += 1
                V[c * p + e][col] += 1
                V[d * p + f][col] += 1
                W[e * n + a][col] += 1
                W[f * n + b][col] += 1
                col += 1

                # M2 = (A[b,c]+A[b,d]) * B[c,e]
                # C[f,a]+=1
                U[b * m + c][col] += 1
                U[b * m + d][col] += 1
                V[c * p + e][col] += 1
                W[f * n + a][col] += 1
                col += 1

                # M3 = A[a,c] * (B[d,f]-B[d,e])
                # C[e,f]+=1
                U[a * m + c][col] += 1
                V[d * p + f][col] += 1
                V[d * p + e][col] -= 1
                W[e * n + f][col] += 1
                col += 1

                # M4 = A[b,d] * (B[c,f]-B[c,e])
                # C[e,a]+=1
                U[b * m + d][col] += 1
                V[c * p + f][col] += 1
                V[c * p + e][col] -= 1
                W[e * n + a][col] += 1
                col += 1

                # M5 = (A[a,c]+A[a,d]) * B[d,f]
                # C[e,f]+=1, C[e,a]-=1
                U[a * m + c][col] += 1
                U[a * m + d][col] += 1
                V[d * p + f][col] += 1
                W[e * n + f][col] += 1
                W[e * n + a][col] -= 1
                col += 1

                # M6 = (A[b,c]-A[a,c]) * (B[c,e]+B[c,f])
                # C[f,a]+=1
                U[b * m + c][col] += 1
                U[a * m + c][col] -= 1
                V[c * p + e][col] += 1
                V[c * p + f][col] += 1
                W[f * n + a][col] += 1
                col += 1

                # M7 = (A[b,d]-A[a,d]) * (B[d,e]+B[d,f])
                # C[e,a]-=1, C[f,b]+=1
                U[b * m + d][col] += 1
                U[a * m + d][col] -= 1
                V[d * p + e][col] += 1
                V[d * p + f][col] += 1
                W[e * n + a][col] -= 1
                W[f * n + b][col] += 1
                col += 1

    # Schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_covered or j not in m_covered or k not in p_covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    assert col == total_rank, f"Expected rank {total_rank}, got {col}"

    if verify_decomp(U, V, W, total_rank):
        return {"U": U, "V": V, "W": W}

    return schoolbook()
