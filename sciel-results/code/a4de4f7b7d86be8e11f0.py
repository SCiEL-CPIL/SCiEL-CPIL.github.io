"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
DEADLINE = 150.0
START_TIME = time.time()

def time_left():
    return DEADLINE - 5.0 - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school), dtype=float)
    V = np.zeros((m*p, rank_school), dtype=float)
    W = np.zeros((p*n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    idx_i = np.repeat(np.arange(n), m*p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n*m)
    U_idx = idx_i*m + idx_j
    V_idx = idx_j*p + idx_k
    W_idx = idx_k*n + idx_i

    def error(U, V, W):
        prod = U[U_idx] * V[V_idx] * W[W_idx]
        sums = prod.sum(axis=1)
        return np.sum((sums - 1.0)**2)

    def exact_check(U, V, W):
        R = U.shape[1]
        Uf = [[Fraction(str(U[i,j])) for j in range(R)] for i in range(n*m)]
        Vf = [[Fraction(str(V[i,j])) for j in range(R)] for i in range(m*p)]
        Wf = [[Fraction(str(W[i,j])) for j in range(R)] for i in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r] for r in range(R))
                    if s != 1:
                        return False
        return True

    best_rank = rank_school
    best_U, best_V, best_W = U, V, W

    for R in [55, 54, 53, 52, 50, 48]:
        if time_left() < 15.0:
            break
        for attempt in range(2):
            if time_left() < 15.0:
                break
            U = np.random.randint(-2, 3, size=(n*m, R)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, R)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, R)) * 0.5
            err = error(U, V, W)
            for it in range(800):
                if time_left() < 15.0:
                    break
                mat = random.randint(0, 2)
                if mat == 0:
                    row, col = random.randint(0, n*m-1), random.randint(0, R-1)
                    old = U[row, col]
                    U[row, col] += random.choice([-0.5, 0.5])
                    ne = error(U, V, W)
                    if ne >= err:
                        U[row, col] = old
                    else:
                        err = ne
                elif mat == 1:
                    row, col = random.randint(0, m*p-1), random.randint(0, R-1)
                    old = V[row, col]
                    V[row, col] += random.choice([-0.5, 0.5])
                    ne = error(U, V, W)
                    if ne >= err:
                        V[row, col] = old
                    else:
                        err = ne
                else:
                    row, col = random.randint(0, p*n-1), random.randint(0, R-1)
                    old = W[row, col]
                    W[row, col] += random.choice([-0.5, 0.5])
                    ne = error(U, V, W)
                    if ne >= err:
                        W[row, col] = old
                    else:
                        err = ne
                if err < 1e-10:
                    if exact_check(U, V, W):
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                        break
            if best_rank == R:
                break

    U_out = [[int(x) if x == int(x) else float(x) for x in row] for row in best_U]
    V_out = [[int(x) if x == int(x) else float(x) for x in row] for row in best_V]
    W_out = [[int(x) if x == int(x) else float(x) for x in row] for row in best_W]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
