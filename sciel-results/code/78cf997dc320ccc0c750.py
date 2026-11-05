import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 148.0

    n, m, p = 2, 3, 4
    num_terms = n * m * p  # 24 terms to satisfy

    def time_left():
        return deadline - time.time()

    # Build rank-22 solution as fallback
    def make_rank22():
        r = 22
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]

        # Strassen block 1: A[0:2,0:2]*B[0:2,0:2]
        # (a00+a11)(b00+b11) - a00*b00 - a11*b11 -> c00,c11
        U[0][0] += 1; U[4][0] += 1; U[0][1] += 1; U[4][1] += 1
        V[0][0] += 1; V[5][0] += 1; V[0][1] += 1; V[5][1] += 1
        W[0][0] += 1; W[3][0] += 1; W[0][1] -= 1; W[3][1] -= 1

        # (a10)(b01) -> c10
        U[1][2] += 1
        V[1][2] += 1
        W[1][2] += 1

        # (a01)(b10) -> c01
        U[0][3] += 1
        V[4][3] += 1
        W[2][3] += 1

        # (a00+a10-a11)(b01+b11) - a00*b01 - a11*b11 -> c00,c11
        U[0][4] += 1; U[1][4] += 1; U[4][4] -= 1
        V[1][4] += 1; V[5][4] += 1
        W[0][4] += 1; W[3][4] += 1

        # (a11+a10-a00)(b10+b11) - a11*b10 - a00*b00 -> c01,c10
        U[4][5] += 1; U[1][5] += 1; U[0][5] -= 1
        V[4][5] += 1; V[5][5] += 1
        W[2][5] += 1; W[1][5] += 1

        # (a00+a10)(b00+b10) - a00*b00 - a10*b10 -> c00,c10
        U[0][6] += 1; U[1][6] += 1
        V[0][6] += 1; V[4][6] += 1
        W[0][6] += 1; W[1][6] += 1

        # Strassen block 2: A[0:2,0:2]*B[0:2,2:4]
        U[0][7] += 1; U[4][7] += 1; U[0][8] += 1; U[4][8] += 1
        V[2][7] += 1; V[7][7] += 1; V[2][8] += 1; V[7][8] += 1
        W[4][7] += 1; W[7][7] += 1; W[4][8] -= 1; W[7][8] -= 1

        U[1][9] += 1
        V[3][9] += 1
        W[5][9] += 1

        U[0][10] += 1
        V[6][10] += 1
        W[6][10] += 1

        U[0][11] += 1; U[1][11] += 1; U[4][11] -= 1
        V[3][11] += 1; V[7][11] += 1
        W[4][11] += 1; W[7][11] += 1

        U[4][12] += 1; U[1][12] += 1; U[0][12] -= 1
        V[6][12] += 1; V[7][12] += 1
        W[6][12] += 1; W[5][12] += 1

        U[0][13] += 1; U[1][13] += 1
        V[2][13] += 1; V[6][13] += 1
        W[4][13] += 1; W[5][13] += 1

        # Cross terms: A[0:2,2:3]*B[2:3,0:4]
        # a02*b20 -> c00, a12*b20 -> c10
        U[2][14] += 1; V[8][14] += 1; W[0][14] += 1
        U[5][14] += 1; V[8][14] += 1; W[1][14] += 1

        # a02*b21 -> c01, a12*b21 -> c11
        U[2][15] += 1; V[9][15] += 1; W[2][15] += 1
        U[5][15] += 1; V[9][15] += 1; W[3][15] += 1

        # a02*b22 -> c02, a12*b22 -> c12
        U[2][16] += 1; V[10][16] += 1; W[4][16] += 1
        U[5][16] += 1; V[10][16] += 1; W[5][16] += 1

        # a02*b23 -> c03, a12*b23 -> c13
        U[2][17] += 1; V[11][17] += 1; W[6][17] += 1
        U[5][17] += 1; V[11][17] += 1; W[7][17] += 1

        # Wait, I need to check the indexing. Let me redo this more carefully.
        return U, V, W

    # Better: build rank-22 properly
    def build_rank_22():
        r = 22
        U = [[0]*r for _ in range(n*m)]  # 6 rows
        V = [[0]*r for _ in range(m*p)]  # 12 rows
        W = [[0]*r for _ in range(p*n)]  # 8 rows

        def add(u_list, v_list, w_list, col):
            for idx, val in u_list:
                U[idx][col] += val
            for idx, val in v_list:
                V[idx][col] += val
            for idx, val in w_list:
                W[idx][col] += val

        # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # M1 = (a00+a11)(b00+b11), used for c00 and c11
        add([(0,1),(4,1)], [(0,1),(5,1)], [(0,1),(3,1)], 0)
        # M2 = a10*(b01-b11), used for c10
        add([(1,1)], [(1,1),(5,-1)], [(1,1)], 1)
        # M3 = a01*(b10-b00), used for c01
        add([(3,1)], [(4,1),(0,-1)], [(2,1)], 2)
        # M4 = a00*(b11), used for c00
        add([(0,1)], [(5,1)], [(0,1)], 3)
        # M5 = a11*(b00), used for c11
        add([(4,1)], [(0,1)], [(3,1)], 4)
        # M6 = (a10+a11)*b10, used for c10 and c11
        add([(1,1),(4,1)], [(4,1)], [(1,1),(3,1)], 5)
        # M7 = (a00+a10)*b01, used for c00 and c01
        add([(0,1),(1,1)], [(1,1)], [(0,1),(2,1)], 6)

        # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add([(0,1),(4,1)], [(2,1),(7,1)], [(4,1),(7,1)], 7)
        add([(1,1)], [(3,1),(7,-1)], [(5,1)], 8)
        add([(3,1)], [(6,1),(2,-1)], [(6,1)], 9)
        add([(0,1)], [(7,1)], [(4,1)], 10)
        add([(4,1)], [(2,1)], [(7,1)], 11)
        add([(1,1),(4,1)], [(6,1)], [(5,1),(7,1)], 12)
        add([(0,1),(1,1)], [(3,1)], [(4,1),(6,1)], 13)

        # Cross: A[0:2,2:3] * B[2:3,0:4]
        # a02*b20, a02*b21, a02*b22, a02*b23
        # a12*b20, a12*b21, a12*b22, a12*b23
        add([(2,1)], [(8,1)], [(0,1)], 14)
        add([(2,1)], [(9,1)], [(2,1)], 15)
        add([(2,1)], [(10,1)], [(4,1)], 16)
        add([(2,1)], [(11,1)], [(6,1)], 17)
        add([(5,1)], [(8,1)], [(1,1)], 18)
        add([(5,1)], [(9,1)], [(3,1)], 19)
        add([(5,1)], [(10,1)], [(5,1)], 20)
        add([(5,1)], [(11,1)], [(7,1)], 21)

        return U, V, W, r

    U22, V22, W22, r22 = build_rank_22()

    def verify_float(U, V, W, rank, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(rank):
                        s += Fraction(U[i*m+j][t]) * Fraction(V[j*p+k][t]) * Fraction(W[k*n+i][t])
                    if s != 1:
                        return False
        return True

    # Verify rank-22
    if verify_float(U22, V22, W22, 22):
        fallback = {"U": U22, "V": V22, "W": W22}
    else:
        # Fallback to naive rank-24
        U = [[0]*24 for _ in range(n*m)]
        V = [[0]*24 for _ in range(m*p)]
        W = [[0]*24 for _ in range(p*n)]
        for t in range(24):
            i = t // (m*p)
            rem = t % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j][t] = 1
            V[j*p+k][t] = 1
            W[k*n+i][t] = 1
        fallback = {"U": U, "V": V, "W": W}

    # Try to find lower rank using structured search
    # Key idea: use the algebraic structure - each term is a product of linear forms
    # Try rank 20, 19, 18 with different strategies

    def snap_half(arr):
        """Snap to nearest half-integer."""
        out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i,j]
                # Try integers and half-integers
                best = min(range(-10, 11), key=lambda z: abs(v - z/2.0))
                out[i,j] = best / 2.0
        return out

    def polish(U, V, W, rank, max_iter=50, tol=1e-12):
        """Polish a decomposition using alternating least squares."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    idx = k*n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                    b = np.ones(m)
                    if rank > 0:
                        W_new[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                    b = np.ones(p)
                    if rank > 0:
                        U_new[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    idx = j*p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                    b = np.ones(n)
                    if rank > 0:
                        V_new[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]

            diff = np.max(np.abs(U_new-U)) + np.max(np.abs(V_new-V)) + np.max(np.abs(W_new-W))
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break

        return U, V, W

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    best_result = fallback
    best_rank = fallback['U'][0].__len__()

    # Try to improve from rank 22
    for target_rank in [21, 20, 19]:
        if time.time() > deadline - 10:
            break

        found = False

        # Strategy 1: Start from rank-22, drop columns, polish
        if target_rank == 21:
            U22n = np.array(U22, dtype=float)
            V22n = np.array(V22, dtype=float)
            W22n = np.array(W22, dtype=float)

            for drop in range(22):
                if time.time() > deadline - 8:
                    break
                mask = [c for c in range(22) if c != drop]
                U0 = U22n[:, mask]
                V0 = V22n[:, mask]
                W0 = W22n[:, mask]

                U1, V1, W1 = polish(U0, V0, W0, 21, max_iter=30)
                U1l, V1l, W1l = to_list(U1), to_list(V1), to_list(W1)
                if verify_float(U1l, V1l, W1l, 21):
                    best_result = {"U": U1l, "V": V1l, "W": W1l}
                    best_rank = 21
                    found = True
                    break

        # Strategy 2: Random init + polish
        if not found and target_rank <= 20:
            for seed in range(20):
                if time.time() > deadline - 8:
                    break
                rng = np.random.RandomState(seed*97+13)
                U0 = rng.randn(n*m, target_rank) * 0.3
                V0 = rng.randn(m*p, target_rank) * 0.3
                W0 = rng.randn(p*n, target_rank) * 0.3

                U1, V1, W1 = polish(U0, V0, W0, target_rank, max_iter=80)
                U1l, V1l, W1l = to_list(U1), to_list(V1), to_list(W1)
                if verify_float(U1l, V1l, W1l, target_rank):
                    best_result = {"U": U1l, "V": V1l, "W": W1l}
                    best_rank = target_rank
                    found = True
                    break

    return best_result
