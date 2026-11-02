import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_block():
        """
        Decompose 2x4 * 4x8 into 8 independent 2x2 * 2x2 blocks,
        each using Strassen's 7-multiplication algorithm.
        
        A is 2x4: rows 0,1; columns 0,1,2,3
        B is 4x8: rows 0,1,2,3; columns 0,1,...,7
        C is 2x8: rows 0,1; columns 0,1,...,7
        
        We split B into 4 column-blocks of width 2: B[:, 2v:2v+2] for v=0,1,2,3
        A is split into 2 column-blocks of width 2: A[:, 2u:2u+2] for u=0,1
        
        C[:, 2v:2v+2] = sum_u A[:, 2u:2u+2] * B[2u:2u+2, 2v:2v+2]
        
        For each (u,v) pair, we have a 2x2 * 2x2 multiplication, rank 7.
        Total: 2*4*7 = 56.
        """
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        for u in range(2):
            for v in range(4):
                base_col = 7 * (4 * u + v)

                # A indices: row i, col j -> i*m + j = i*4 + j
                # A[:, 2u:2u+2]
                a11 = 0 * m + 2 * u       # A[0, 2u]
                a12 = 0 * m + 2 * u + 1   # A[0, 2u+1]
                a21 = 1 * m + 2 * u       # A[1, 2u]
                a22 = 1 * m + 2 * u + 1   # A[1, 2u+1]

                # B indices: row j, col k -> j*p + k = j*8 + k
                # B[2u:2u+2, 2v:2v+2]
                b11 = (2 * u) * p + 2 * v           # B[2u, 2v]
                b12 = (2 * u) * p + 2 * v + 1       # B[2u, 2v+1]
                b21 = (2 * u + 1) * p + 2 * v       # B[2u+1, 2v]
                b22 = (2 * u + 1) * p + 2 * v + 1   # B[2u+1, 2v+1]

                # C indices: row i, col k -> k*n + i (trace convention W[k*n+i, r])
                # C[:, 2v:2v+2]
                c11 = (2 * v) * n + 0       # C[0, 2v]
                c12 = (2 * v + 1) * n + 0   # C[0, 2v+1]
                c21 = (2 * v) * n + 1       # C[1, 2v]
                c22 = (2 * v + 1) * n + 1   # C[1, 2v+1]

                # Strassen for 2x2 * 2x2:
                # M1 = (A11 + A22)(B11 + B22)
                # M2 = (A21 + A22)B11
                # M3 = A11(B12 - B22)
                # M4 = A22(B21 - B11)
                # M5 = (A11 + A12)B22
                # M6 = (A21 - A11)(B11 + B12)
                # M7 = (A12 - A22)(B21 + B22)
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6

                # M1
                U[a11, base_col] += 1; U[a22, base_col] += 1
                V[b11, base_col] += 1; V[b22, base_col] += 1
                W[c11, base_col] += 1; W[c22, base_col] += 1

                # M2
                U[a21, base_col+1] += 1; U[a22, base_col+1] += 1
                V[b11, base_col+1] += 1
                W[c21, base_col+1] += 1; W[c22, base_col+1] -= 1

                # M3
                U[a11, base_col+2] += 1
                V[b12, base_col+2] += 1; V[b22, base_col+2] -= 1
                W[c12, base_col+2] += 1; W[c22, base_col+2] += 1

                # M4
                U[a22, base_col+3] += 1
                V[b21, base_col+3] += 1; V[b11, base_col+3] -= 1
                W[c11, base_col+3] += 1; W[c21, base_col+3] += 1

                # M5
                U[a11, base_col+4] += 1; U[a12, base_col+4] += 1
                V[b22, base_col+4] += 1
                W[c11, base_col+4] -= 1; W[c12, base_col+4] += 1

                # M6
                U[a21, base_col+5] += 1; U[a11, base_col+5] -= 1
                V[b11, base_col+5] += 1; V[b12, base_col+5] += 1
                W[c22, base_col+5] += 1

                # M7
                U[a12, base_col+6] += 1; U[a22, base_col+6] -= 1
                V[b21, base_col+6] += 1; V[b22, base_col+6] += 1
                W[c11, base_col+6] += 1

        return U, V, W, rank

    def verify(U, V, W, rank, atol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    # Build Strassen block decomposition (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Try to reduce rank by checking if any columns are linearly dependent
    # or if we can merge some terms
    # For now, just return the best we have
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
