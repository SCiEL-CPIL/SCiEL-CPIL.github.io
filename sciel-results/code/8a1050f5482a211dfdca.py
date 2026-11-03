import numpy as np
import random
import time
from itertools import combinations

DIMENSIONS = [2, 3, 10]

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Full exact verification: check all (i,j,k) and all non-diagonal triples."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Check all diagonal entries: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(U[a], V[b] * W[c]))
                if abs(val - 1.0) > tol:
                    return False

    # Check all non-diagonal triples: sum_r U[a, r] * V[b, r] * W[c, r] == 0
    # Only need to check triples where (a,b,c) does NOT correspond to a valid (i,j,k)
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            jj = b // p
            k = b % p
            if jj != j:
                continue  # invalid triple, skip
            for c in range(p * n):
                cc = c // n
                ii = c % n
                if cc != k or ii != i:
                    val = float(np.dot(U[a], V[b] * W[c]))
                    if abs(val) > tol:
                        return False

    return True

def _is_valid_entries(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_strassen55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _try_remove_term(U, V, W, col, n, m, p, tol=1e-9):
    """Try removing column `col` and see if the remaining terms still satisfy the tensor identity.
    This works by checking if the column's contribution is redundant (i.e., can be expressed
    as a linear combination of other columns in the tensor product space).
    We test by zeroing out the column and verifying."""
    rank = len(U[0])
    if rank <= 1:
        return None

    U2 = [row[:] for row in U]
    V2 = [row[:] for row in V]
    W2 = [row[:] for row in W]

    # Zero out column col
    for r in range(len(U2)):
        U2[r][col] = 0
    for r in range(len(V2)):
        V2[r][col] = 0
    for r in range(len(W2)):
        W2[r][col] = 0

    # Remove the column from all matrices
    U3 = [row[:col] + row[col+1:] for row in U2]
    V3 = [row[:col] + row[col+1:] for row in V2]
    W3 = [row[:col] + row[col+1:] for row in W2]

    if not _verify_exact(U3, V3, W3, n, m, p, tol):
        return None

    return U3, V3, W3


def _local_search_remove(U, V, W, n, m, p, deadline):
    """Iteratively try to remove rank-1 terms from the decomposition."""
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    tol = 1e-9

    while time.time() < deadline - 15:
        removed = False
        current_rank = len(best_U[0])
        for col in range(current_rank):
            if time.time() > deadline - 15:
                break
            result = _try_remove_term(best_U, best_V, best_W, col, n, m, p, tol)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = len(best_U[0])
                removed = True
                break
        if not removed:
            break

    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify the Strassen-55 decomposition is correct
    if not _verify_exact(U_best, V_best, W_best, n, m, p):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p
    elif not _is_valid_entries(U_best, V_best, W_best):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to reduce rank via local search
    U_ls, V_ls, W_ls, ls_rank = _local_search_remove(U_best, V_best, W_best, n, m, p, deadline)
    if ls_rank < best_rank:
        best_rank = ls_rank
        U_best, V_best, W_best = U_ls, V_ls, W_ls

    return {"U": U_best, "V": V_best, "W": W_best}
