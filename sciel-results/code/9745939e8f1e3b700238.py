import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s kill

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def _elapsed_ok():
        return (time.time() - start) < deadline

    # Schoolbook for (n,m,p): rank = n*m*p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return U, V, W

    # Strassen for 2x2 * 2x2: rank 7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    def strassen_222():
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22

        # Define the 7 products as lists of (a_idx, b_idx) pairs
        # Each pair represents a term A[a_idx] * B[b_idx] in the expansion
        products = []

        # M1: (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        products.append([(0, 0), (0, 3), (3, 0), (3, 3)])

        # M2: (A21+A22)B11 = A21*B11 + A22*B11
        products.append([(2, 0), (3, 0)])

        # M3: A11(B21-B22) = A11*B21 - A11*B22
        products.append([(0, 2), (0, 3)])

        # M4: A22(B11-B21) = A22*B11 - A22*B21
        products.append([(3, 0), (3, 2)])

        # M5: (A11+A12)B22 = A11*B22 + A12*B22
        products.append([(0, 3), (1, 3)])

        # M6: (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        products.append([(2, 0), (2, 1), (0, 0), (0, 1)])

        # M7: (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        products.append([(1, 2), (1, 3), (3, 2), (3, 3)])

        # Now build U, V, W
        # Each term in each product contributes one rank-1 term.
        # We need to track the coefficient for each term.
        #
        # For M1: coefficients are all +1
        # For M2: coefficients are all +1
        # For M3: A11*B21 has +1, A11*B22 has -1
        # For M4: A22*B11 has +1, A22*B21 has -1
        # For M5: coefficients are all +1
        # For M6: A21*B11 has +1, A21*B12 has +1, A11*B11 has -1, A11*B12 has -1
        # For M7: A12*B21 has +1, A12*B22 has +1, A22*B21 has -1, A22*B22 has -1

        # Coefficients for each term in each product
        coeffs = [
            [1, 1, 1, 1],      # M1
            [1, 1],             # M2
            [1, -1],            # M3
            [1, -1],            # M4
            [1, 1],             # M5
            [1, 1, -1, -1],     # M6
            [1, 1, -1, -1]      # M7
        ]

        total_terms = sum(len(prod) for prod in products)
        U = [[0] * total_terms for _ in range(4)]
        V = [[0] * total_terms for _ in range(4)]
        W = [[0] * total_terms for _ in range(4)]

        # C contributions for each product:
        # prod 0 = M1: C11 += 1, C22 += 1
        # prod 1 = M2: C21 += 1, C22 += -1
        # prod 2 = M3: C12 += 1, C22 += 1
        # prod 3 = M4: C11 += 1, C21 += 1
        # prod 4 = M5: C11 += -1, C12 += 1
        # prod 5 = M6: C22 += 1
        # prod 6 = M7: C11 += 1

        c_contribs = {
            0: {0: 1, 3: 1},   # M1 -> C11, C22
            1: {2: 1, 3: -1},  # M2 -> C21, C22
            2: {1: 1, 3: 1},   # M3 -> C12, C22
            3: {0: 1, 2: 1},   # M4 -> C11, C21
            4: {0: -1, 1: 1},  # M5 -> C11, C12
            5: {3: 1},         # M6 -> C22
            6: {0: 1},         # M7 -> C11
        }

        t = 0
        for prod_idx, prod in enumerate(products):
            for term_idx, (ai, bi) in enumerate(prod):
                coeff = coeffs[prod_idx][term_idx]
                U[ai][t] = coeff
                V[bi][t] = 1
                for c, wval in c_contribs[prod_idx].items():
                    W[c][t] = wval
                t += 1

        return U, V, W

    def make_221():
        """Return U,V,W for 2x2*2x1, schoolbook rank 4."""
        U, V, W = schoolbook(2, 2, 1)
        return U, V, W

    def make_223():
        """Return U,V,W for 2x2*2x3, rank 11."""
        # C = A*B1 + A*B2 where B1 is 2x1, B2 is 2x2
        U1, V1, W1 = make_221()  # 2x2*2x1
        U2, V2, W2 = make_222()  # 2x2*2x2
        #
        # U1: 4 rows, 4 cols (A11,A12,A21,A22)
        # V1: 2 rows, 4 cols (B11,B21)
        # W1: 2 rows, 4 cols (C11,C21)
        #
        # U2: 4 rows, 7 cols
        # V2: 4 rows, 7 cols (B11,B12,B21,B22)
        # W2: 4 rows, 7 cols (C11,C12,C21,C22)
        #
        # Combined:
        # U: 4 rows, 11 cols
        # V: 6 rows, 11 cols (2 from V1, 4 from V2)
        # W: 6 rows, 11 cols (2 from W1, 4 from W2)

        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(2):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*4 + V2[i])
        W = []
        for i in range(2):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*4 + W2[i])
        return U, V, W

    def make_224():
        """Return U,V,W for 2x2*2x4, rank 14."""
        # C = A*B1 + A*B2 where B1 is 2x2, B2 is 2x2
        U1, V1, W1 = make_222()
        U2, V2, W2 = make_222()
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(4):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*7 + V2[i])
        W = []
        for i in range(4):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*7 + W2[i])
        return U, V, W

    def make_227():
        """Return U,V,W for 2x2*2x7, rank 25."""
        # C = A*B1 + A*B2 where B1 is 2x4, B2 is 2x3
        U1, V1, W1 = make_224()  # 2x2*2x4
        U2, V2, W2 = make_223()  # 2x2*2x3
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(8):
            V.append(V1[i] + [0]*11)
        for i in range(6):
            V.append([0]*14 + V2[i])
        W = []
        for i in range(8):
            W.append(W1[i] + [0]*11)
        for i in range(6):
            W.append([0]*14 + W2[i])
        return U, V, W

    def make_247():
        """Return U,V,W for 2x4*4x7, rank 50."""
        # C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x7
        U1, V1, W1 = make_227()  # 2x2*2x7
        U2, V2, W2 = make_227()  # 2x2*2x7
        # U: 8 rows (A1 has 4, A2 has 4)
        # V: 14 rows (B1 has 14, B2 has 14)
        # W: 14 rows (C1 has 14, C2 has 14)
        U = []
        for i in range(4):
            U.append(U1[i] + [0]*25)
        for i in range(4):
            U.append([0]*25 + U2[i])
        V = []
        for i in range(14):
            V.append(V1[i] + [0]*25)
        for i in range(14):
            V.append([0]*25 + V2[i])
        W = []
        for i in range(14):
            W.append(W1[i] + [0]*25)
        for i in range(14):
            W.append([0]*25 + W2[i])
        return U, V, W

    try:
        if not _elapsed_ok():
            raise TimeoutError()

        U, V, W = make_247()
        rank = len(U[0])

        # Verify with a random test
        rng = np.random.default_rng(42)
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        ref = A @ B

        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        # C[i,k] = sum_t U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]
        # More precisely: C_flat[k*n+i] = sum_t (sum_j U[i*m+j,t]*A[i,j]) * (sum_j V[j*p+k,t]*B[j,k]) * W[k*n+i,t]
        # Wait, that's not right. The tensor is:
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_t ... 
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # And the decomposition says: C[i,k] = sum_t (sum_j U[i*m+j,t] * A[i,j]) * ... 
        # No, the standard tensor decomposition is:
        # C[i,k] = sum_t U[i*m+*, t] contracted with A, V[*, j*p+k, t] contracted with B, times W[k*n+i, t]
        # More precisely:
        # Let a_t = sum_j U[i*m+j, t] * A[i,j]  -- this depends on i, so it's per i
        # Let b_t = sum_j V[j*p+k, t] * B[j,k]  -- this depends on k, so it's per k
        # Then C[i,k] = sum_t a_t(i) * b_t(k) * W[k*n+i, t]
        # That's not a simple tensor contraction. Let me re-examine.
        #
        # The matrix multiplication tensor T has T[i,j,k] = delta such that
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The rank-r decomposition is:
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # where U, V, W are the factor matrices.
        #
        # So to verify: for each (i,j,k), check that
        # 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # (since T[i,j,k] = 1 if we think of A[i,j]*B[j,k] contributing to C[i,k])
        #
        # Wait, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
        # So we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k.

        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            U, V, W = schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

    except Exception as e:
        # Fall back to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
