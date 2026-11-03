import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2_2():
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    return U, V, W

def _get_17_rank_block():
    """Rank-17 decomposition of the 3x3x2 multiplication tensor (local coords)."""
    rank = 17
    U = np.zeros((9, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))

    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    col = 0
    for l in range(7):
        for li, coeff in U_coeffs[l].items():
            i_local = li // 2
            j_local = li % 2
            U[i_local*3 + j_local, col] += coeff
        for lj, coeff in V_coeffs[l].items():
            j_local = lj // 2
            k_local = lj % 2
            V[j_local*2 + k_local, col] += coeff
        for lc, coeff in W_coeffs[l].items():
            i_local = lc // 2
            k_local = lc % 2
            W[k_local*3 + i_local, col] += coeff
        col += 1

    # Schoolbook part (10 terms)
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + i, col] = 1.0
            col += 1
    for j in range(2):
        for k in range(2):
            U[2*3 + j, col] = 1.0
            V[j*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
    for k in range(2):
        U[2*3 + 2, col] = 1.0
        V[2*2 + k, col] = 1.0
        W[k*3 + 2, col] = 1.0
        col += 1

    return U, V, W

def _try_rank_16_block():
    """Attempt to find an exact rank-16 decomposition of the 3x3x2 block."""
    U17, V17, W17 = _get_17_rank_block()
    # Drop the last (isolated) schoolbook term as the seed for rank 16.
    U_init = U17[:, :16]
    V_init = V17[:, :16]
    W_init = W17[:, :16]
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

    def residuals(x):
        U = x[:144].reshape(9, 16)
        V = x[144:240].reshape(6, 16)
        W = x[240:336].reshape(6, 16)
        T = np.zeros((3, 3, 2))
        for c in range(16):
            u = U[:, c].reshape(3, 3)
            v = V[:, c].reshape(3, 2)
            w = W[:, c].reshape(2, 3)
            for i in range(3):
                for j in range(3):
                    if abs(u[i, j]) < 1e-15:
                        continue
                    for k in range(2):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return (T - 1.0).flatten()

    best = None
    for attempt in range(3):
        if _time_left() < 30:
            break
        seed = x0.copy()
        if attempt > 0:
            seed = seed + 0.05 * np.random.RandomState(100 + attempt).standard_normal(seed.size)
        try:
            res = least_squares(residuals, seed, method='lm', max_nfev=3000)
        except Exception:
            continue
        if np.max(np.abs(res.fun)) < 1e-8:
            x_opt = res.x
            U_opt = x_opt[:144].reshape(9, 16)
            V_opt = x_opt[144:240].reshape(6, 16)
            W_opt = x_opt[240:336].reshape(6, 16)
            U_r = np.round(U_opt * 2) / 2
            V_r = np.round(V_opt * 2) / 2
            W_r = np.round(W_opt * 2) / 2
            T = np.zeros((3, 3, 2))
            for c in range(16):
                u = U_r[:, c].reshape(3, 3)
                v = V_r[:, c].reshape(3, 2)
                w = W_r[:, c].reshape(2, 3)
                for i in range(3):
                    for j in range(3):
                        if abs(u[i, j]) < 1e-15:
                            continue
                        for k in range(2):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            if np.max(np.abs(T - 1.0)) < 1e-10:
                return U_r, V_r, W_r
    return None

def _assemble_full(n, m, p, U_block, V_block, W_block, rank_block):
    rank_total = 4 * rank_block
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        for a in range(n*m):
            U[a][col_start:col_start+rank_block] = [float(x) for x in U_block[a, :rank_block]]
        for j in range(m):
            for k_local in range(2):
                k_global = k0 + k_local
                for c in range(rank_block):
                    V[j*p + k_global][col_start + c] = float(V_block[j*2 + k_local, c])
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(n):
                for c in range(rank_block):
                    W[k_global*n + i][col_start + c] = float(W_block[k_local*3 + i, c])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    # 1) Fixed known-good Strassen placement (4 blocks along k) -> rank 68.
    if _time_left() > 5:
        try:
            blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # 2) Refinement: search for a rank-16 3x3x2 block -> full rank 64.
    if _time_left() > 40:
        try:
            block = _try_rank_16_block()
            if block is not None:
                U, V, W = _assemble_full(n, m, p, block[0], block[1], block[2], 16)
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_result
