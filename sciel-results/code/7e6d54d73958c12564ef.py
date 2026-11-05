"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []

    # Strassen rank-7 decomposition for a 2x2x2 block
    # Using the standard Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def add_strassen_2x2x2(i0, j0, k0):
        # M1: A:(i0,j0):1, (i0+1,j0+1):1; B:(j0,k0):1, (j0+1,k0+1):1; C:(i0,k0):1, (i0+1,k0+1):1
        U_cols.append({i0*m+j0: 1, (i0+1)*m+(j0+1): 1})
        V_cols.append({j0*p+k0: 1, (j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: 1, (k0+1)*n+(i0+1): 1})

        # M2: A:(i0+1,j0):1, (i0+1,j0+1):1; B:(j0,k0):1; C:(i0+1,k0):1
        U_cols.append({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1})
        V_cols.append({j0*p+k0: 1})
        W_cols.append({k0*n+(i0+1): 1})

        # M3: A:(i0,j0):1; B:(j0,k0+1):1, (j0+1,k0+1):-1; C:(i0,k0+1):1
        U_cols.append({i0*m+j0: 1})
        V_cols.append({j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1})
        W_cols.append({(k0+1)*n+i0: 1})

        # M4: A:(i0+1,j0+1):1; B:(j0+1,k0):1, (j0,k0):-1; C:(i0+1,k0):1, (i0,k0):1
        U_cols.append({(i0+1)*m+(j0+1): 1})
        V_cols.append({(j0+1)*p+k0: 1, j0*p+k0: -1})
        W_cols.append({k0*n+(i0+1): 1, k0*n+i0: 1})

        # M5: A:(i0,j0):1, (i0,j0+1):1; B:(j0+1,k0+1):1; C:(i0,k0):-1, (i0,k0+1):1
        U_cols.append({i0*m+j0: 1, i0*m+(j0+1): 1})
        V_cols.append({(j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: -1, (k0+1)*n+i0: 1})

        # M6: A:(i0+1,j0):1, (i0,j0):-1; B:(j0,k0):1, (j0,k0+1):1; C:(i0+1,k0+1):1
        U_cols.append({(i0+1)*m+j0: 1, i0*m+j0: -1})
        V_cols.append({j0*p+k0: 1, j0*p+(k0+1): 1})
        W_cols.append({(k0+1)*n+(i0+1): 1})

        # M7: A:(i0,j0+1):1, (i0+1,j0+1):-1; B:(j0+1,k0):1, (j0+1,k0+1):1; C:(i0,k0):1
        U_cols.append({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1})
        V_cols.append({(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1})
        W_cols.append({k0*n+i0: 1})

    def add_schoolbook_block(i0, i1, j0, j1, k0, k1):
        """Add schoolbook terms for a block."""
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    U_cols.append({i*m+j: 1})
                    V_cols.append({j*p+k: 1})
                    W_cols.append({k*n+i: 1})

    # Partition: i: [0,1],[2,3],[4]; j: [0,1],[2,3],[4]; k: [0,1],[2,3],[4]
    # 3x3 = 9 blocks
    # (0,0,0): 2x2x2 -> Strassen (rank 7)
    # All other 8 blocks: mixed sizes -> schoolbook

    for pi in range(3):
        for pj in range(3):
            for pk in range(3):
                i0 = [0, 2, 4][pi]
                i1 = [2, 4, 5][pi]
                j0 = [0, 2, 4][pj]
                j1 = [2, 4, 5][pj]
                k0 = [0, 2, 4][pk]
                k1 = [2, 4, 5][pk]

                si = i1 - i0
                sj = j1 - j0
                sk = k1 - k0

                if si == 2 and sj == 2 and sk == 2:
                    add_strassen_2x2x2(i0, j0, k0)
                else:
                    add_schoolbook_block(i0, i1, j0, j1, k0, k1)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Quick verification with a few test cases
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 9) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    w_val = W[k * n + i][r]
                    if w_val == 0:
                        continue
                    u_dot_v = 0
                    for j in range(m):
                        u_val = U[i * m + j][r]
                        if u_val != 0:
                            u_dot_v += u_val * V[j * p + k][r]
                    s += u_dot_v * w_val
                C_test[i][k] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
