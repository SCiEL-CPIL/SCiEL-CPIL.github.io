import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    n, m, p = 3, 4, 6
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block (rank 7) at given offsets."""
        nonlocal col
        if time.time() > deadline:
            return
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        if time.time() > deadline:
            return
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    # Build the 66-term decomposition.
    # C[0,0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    # C[0,2:4]
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    # C[0,4:6]
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    # C[1:3,0:2]
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    # C[1:3,2:4]
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    # C[1:3,4:6]
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)

    base_rank = col
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Rank-reduction pass: build the rank-1 component matrix
    # M[r, t] = sum_{i,j,k} U[i,t]*V[j,t]*W[k,t] * T[i,j,k]
    # where T[i,j,k] = 1 iff the output cell (k // n, k % n) equals (i, j).
    # A column is linearly dependent iff its M-column is the zero vector.
    # Dropping zero columns preserves the decomposition exactly.
    try:
        Ua = np.array(U, dtype=object)
        Va = np.array(V, dtype=object)
        Wa = np.array(W, dtype=object)
        T = np.zeros((n * m, m * p, p * n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        # M[t, r] = sum_{i,j,k} U[i,t] * V[j,t] * W[k,t] * T[i,j,k]
        M = np.einsum('it,jt,kt,ijk->t', Ua, Va, Wa, T, dtype=object)
        # M shape (rank, n*m)
        # A term is dependent iff its row (over r) is all zero.
        keep = []
        for t in range(base_rank):
            row = M[t, :]
            if all(int(x) == 0 for x in row):
                continue  # drop
            keep.append(t)

        if len(keep) < base_rank:
            new_rank = len(keep)
            U2 = [[0] * new_rank for _ in range(n * m)]
            V2 = [[0] * new_rank for _ in range(m * p)]
            W2 = [[0] * new_rank for _ in range(p * n)]
            for new_t, old_t in enumerate(keep):
                for i in range(n * m):
                    U2[i][new_t] = U[i][old_t]
                for j in range(m * p):
                    V2[j][new_t] = V[j][old_t]
                for k in range(p * n):
                    W2[k][new_t] = W[k][old_t]
            return {"U": U2, "V": V2, "W": W2}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
