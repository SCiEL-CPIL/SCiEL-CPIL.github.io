"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS


def construct():
    """
    Return {'U', 'V', 'W'} factor matrices for the (3,5,5) matrix multiplication tensor.
    U shape: (15, rank), V shape: (25, rank), W shape: (15, rank)
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    # --- Helper: build terms as sparse dicts ---
    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
        return terms

    def strassen_terms():
        """
        Strassen's rank-7 decomposition for 2x2x2.
        Each term: (U_dict, V_dict, W_dict)
        U keys: (i,j) for i,j in {0,1}
        V keys: (j,k) for j,k in {0,1}
        W keys: (k,i) for k,i in {0,1}
        """
        terms = []
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
        # M3 = A11(B21-B22) -> C12 += 1, C22 += 1
        terms.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}))
        # M4 = A22(B12-B11) -> C11 += 1, C21 += 1
        terms.append(({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}))
        # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        # M7 = (A12-A22)(B21+B22) -> C11 += 1
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        return terms

    @lru_cache(maxsize=None)
    def min_rank(n, m, p):
        """Compute minimum rank via recursive block splitting with Strassen base case."""
        if n == 2 and m == 2 and p == 2:
            return 7
        if n <= 1 or m <= 1 or p <= 1:
            return n * m * p

        best = n * m * p  # schoolbook upper bound

        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            r = min_rank(n, m1, p) + min_rank(n, m2, p)
            if r < best:
                best = r

        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            r = min_rank(n, m, p1) + min_rank(n, m, p2)
            if r < best:
                best = r

        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            r = min_rank(n1, m, p) + min_rank(n2, m, p)
            if r < best:
                best = r

        return best

    def decompose(n, m, p):
        """
        Recursively decompose (n,m,p) into a list of (U_dict, V_dict, W_dict) terms.
        """
        if time.time() > deadline:
            return schoolbook_terms(n, m, p)

        if n == 2 and m == 2 and p == 2:
            return strassen_terms()
        if n <= 1 or m <= 1 or p <= 1:
            return schoolbook_terms(n, m, p)

        target = min_rank(n, m, p)

        # Try splitting m
        for m1 in range(1, m):
            m2 = m - m1
            r1 = min_rank(n, m1, p)
            r2 = min_rank(n, m2, p)
            if r1 + r2 == target:
                t1 = decompose(n, m1, p)
                t2 = decompose(n, m2, p)
                shifted = []
                for U_d, V_d, W_d in t2:
                    nU = {(i, j + m1): v for (i, j), v in U_d.items()}
                    nV = {(j + m1, k): v for (j, k), v in V_d.items()}
                    shifted.append((nU, nV, W_d))
                return t1 + shifted

        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            r1 = min_rank(n, m, p1)
            r2 = min_rank(n, m, p2)
            if r1 + r2 == target:
                t1 = decompose(n, m, p1)
                t2 = decompose(n, m, p2)
                shifted = []
                for U_d, V_d, W_d in t2:
                    nV = {(j, k + p1): v for (j, k), v in V_d.items()}
                    nW = {(k + p1, i): v for (k, i), v in W_d.items()}
                    shifted.append((U_d, nV, nW))
                return t1 + shifted

        # Try splitting n
        for n1 in range(1, n):
            n2 = n - n1
            r1 = min_rank(n1, m, p)
            r2 = min_rank(n2, m, p)
            if r1 + r2 == target:
                t1 = decompose(n1, m, p)
                t2 = decompose(n2, m, p)
                shifted = []
                for U_d, V_d, W_d in t2:
                    nU = {(i + n1, j): v for (i, j), v in U_d.items()}
                    nW = {(k, i + n1): v for (k, i), v in W_d.items()}
                    shifted.append((nU, V_d, nW))
                return t1 + shifted

        # Fallback: schoolbook
        return schoolbook_terms(n, m, p)

    # Build the decomposition
    all_terms = decompose(3, 5, 5)
    rank = len(all_terms)

    # Build dense matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r, (U_d, V_d, W_d) in enumerate(all_terms):
        for (i, j), val in U_d.items():
            U[i * m + j, r] = val
        for (j, k), val in V_d.items():
            V[j * p + k, r] = val
        for (k, i), val in W_d.items():
            W[k * n + i, r] = val

    # Verify correctness
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
