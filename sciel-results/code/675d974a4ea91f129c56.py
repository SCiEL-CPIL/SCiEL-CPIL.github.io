import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Diagonal entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Off-diagonal entries should be 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m; j = a % m
                    jj = b // p; k = b % p
                    kk = c // n; ii = c % n
                    if i == ii and j == jj and k == kk:
                        continue
                    val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                    if abs(val) > 1e-9:
                        return False
        return True

    def try_rank28():
        if time.time() > deadline:
            return None
        r = 28
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))

        # Strassen block 1: A[0:2,0:2] x B[0:2,0:2]
        # U idx: i*3+j ; V idx: j*5+k ; W idx: k*2+i
        # M1 = (A00+A11)(B00+B11)
        U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
        V[0*5+0, 0] = 1; V[1*5+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        # M2 = (A10+A11)B00
        U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
        V[0*5+0, 1] = 1
        W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
        # M3 = A00(B01-B11)
        U[0*3+0, 2] = 1
        V[0*5+1, 2] = 1; V[1*5+1, 2] = -1
        W[0*2+0, 2] = 1; W[1*2+1, 2] = 1
        # M4 = A11(B10-B00)
        U[1*3+1, 3] = 1
        V[1*5+0, 3] = 1; V[0*5+0, 3] = -1
        W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
        # M5 = (A00+A01)B11
        U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
        V[1*5+1, 4] = 1
        W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
        V[0*5+0, 5] = 1; V[0*5+1, 5] = 1
        W[1*2+1, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
        V[1*5+0, 6] = 1; V[1*5+1, 6] = 1
        W[0*2+0, 6] = 1

        # Strassen block 2: A[0:2,3:5] x B[3:5,0:2]
        # A col 3,4 ; B col 3,4 ; output cols 0,1
        # U idx: i*3+3, i*3+4 ; V idx: 3*5+k, 4*5+k (k in 0,1) ; W idx: k*2+i (k in 0,1)
        # M8 = (A03+A14)(B30+B41)
        U[0*3+3, 7] = 1; U[1*3+4, 7] = 1
        V[3*5+0, 7] = 1; V[4*5+1, 7] = 1
        W[0*2+0, 7] = 1; W[1*2+1, 7] = 1
        # M9 = (A13+A14)B30
        U[1*3+3, 8] = 1; U[1*3+4, 8] = 1
        V[3*5+0, 8] = 1
        W[0*2+1, 8] = 1; W[1*2+1, 8] = -1
        # M10 = A03(B31-B41)
        U[0*3+3, 9] = 1
        V[3*5+1, 9] = 1; V[4*5+1, 9] = -1
        W[0*2+0, 9] = 1; W[1*2+1, 9] = 1
        # M11 = A14(B40-B30)
        U[1*3+4, 10] = 1
        V[4*5+0, 10] = 1; V[3*5+0, 10] = -1
        W[0*2+0, 10] = 1; W[0*2+1, 10] = 1
        # M12 = (A03+A04)B41
        U[0*3+3, 11] = 1; U[0*3+4, 11] = 1
        V[4*5+1, 11] = 1
        W[0*2+0, 11] = -1; W[1*2+0, 11] = 1
        # M13 = (A13-A03)(B30+B31)
        U[1*3+3, 12] = 1; U[0*3+3, 12] = -1
        V[3*5+0, 12] = 1; V[3*5+1, 12] = 1
        W[1*2+1, 12] = 1
        # M14 = (A04-A14)(B40+B41)
        U[0*3+4, 13] = 1; U[1*3+4, 13] = -1
        V[4*5+0, 13] = 1; V[4*5+1, 13] = 1
        W[0*2+0, 13] = 1

        # Schoolbook: A col 2 (j=2) x B all cols (k=0..4) for i=0,1 -> 10 terms
        col = 14
        for i in range(2):
            for k in range(5):
                U[i*3+2, col] = 1
                V[2*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1

        # Schoolbook: A cols 0,1 (j=0,1) x B col 2 (k=2) for i=0,1 -> 4 terms
        for i in range(2):
            for j in range(2):
                U[i*3+j, col] = 1
                V[j*5+2, col] = 1
                W[2*2+i, col] = 1
                col += 1

        # Schoolbook: A cols 3,4 (j=3,4) x B col 2 (k=2) for i=0,1 -> 4 terms
        for i in range(2):
            for j in range(3, 5):
                U[i*3+j, col] = 1
                V[j*5+2, col] = 1
                W[2*2+i, col] = 1
                col += 1

        if col != r:
            return None
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    result28 = try_rank28()
    if result28 is not None:
        best_result = result28

    # Attempt rank-27 via parameterized ansatz with Schwartz-Zippel-style verification
    def try_rank27():
        if time.time() > deadline:
            return None
        # A rank-27 ansatz: take the rank-28 and try to merge two schoolbook terms
        # using a shared linear combination. We parameterize and verify.
        r = 27
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))

        # Same Strassen block 1 (terms 0-6)
        U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
        V[0*5+0, 0] = 1; V[1*5+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
        V[0*5+0, 1] = 1
        W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
        U[0*3+0, 2] = 1
        V[0*5+1, 2] = 1; V[1*5+1, 2] = -1
        W[0*2+0, 2] = 1; W[1*2+1, 2] = 1
        U[1*3+1, 3] = 1
        V[1*5+0, 3] = 1; V[0*5+0, 3] = -1
        W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
        U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
        V[1*5+1, 4] = 1
        W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
        U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
        V[0*5+0, 5] = 1; V[0*5+1, 5] = 1
        W[1*2+1, 5] = 1
        U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
        V[1*5+0, 6] = 1; V[1*5+1, 6] = 1
        W[0*2+0, 6] = 1

        # Same Strassen block 2 (terms 7-13)
        U[0*3+3, 7] = 1; U[1*3+4, 7] = 1
        V[3*5+0, 7] = 1; V[4*5+1, 7] = 1
        W[0*2+0, 7] = 1; W[1*2+1, 7] = 1
        U[1*3+3, 8] = 1; U[1*3+4, 8] = 1
        V[3*5+0, 8] = 1
        W[0*2+1, 8] = 1; W[1*2+1, 8] = -1
        U[0*3+3, 9] = 1
        V[3*5+1, 9] = 1; V[4*5+1, 9] = -1
        W[0*2+0, 9] = 1; W[1*2+1, 9] = 1
        U[1*3+4, 10] = 1
        V[4*5+0, 10] = 1; V[3*5+0, 10] = -1
        W[0*2+0, 10] = 1; W[0*2+1, 10] = 1
        U[0*3+3, 11] = 1; U[0*3+4, 11] = 1
        V[4*5+1, 11] = 1
        W[0*2+0, 11] = -1; W[1*2+0, 11] = 1
        U[1*3+3, 12] = 1; U[0*3+3, 12] = -1
        V[3*5+0, 12] = 1; V[3*5+1, 12] = 1
        W[1*2+1, 12] = 1
        U[0*3+4, 13] = 1; U[1*3+4, 13] = -1
        V[4*5+0, 13] = 1; V[4*5+1, 13] = 1
        W[0*2+0, 13] = 1

        # Schoolbook A col 2 x B all cols: 10 terms (14-23)
        col = 14
        for i in range(2):
            for k in range(5):
                U[i*3+2, col] = 1
                V[2*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1

        # Try to merge the 8 schoolbook terms for B col 2 (k=2) into fewer.
        # We have A cols 0,1,3,4 x B col 2 for i=0,1 -> 8 terms.
        # Attempt a rank-reduction: use 4 terms with half-integer coefficients.
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,3]B[3,2] + A[0,4]B[4,2]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,3]B[3,2] + A[1,4]B[4,2]
        # This is a 2x4x1 tensor. Its border rank is 2, but exact rank is 4.
        # So we cannot reduce below 4 for this sub-tensor. Hence rank 27 is not achievable
        # with this structure. Return None.
        return None

    result27 = try_rank27()
    if result27 is not None:
        best_result = result27

    return best_result
