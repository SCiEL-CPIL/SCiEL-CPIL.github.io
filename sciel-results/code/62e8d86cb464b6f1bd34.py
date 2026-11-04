"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    def verify_numpy(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Efficient verification: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        rank = U_mat.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    v_row = V_mat[j*p+k, :]
                    w_row = W_mat[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > tol:
                        return False, max_err
        return True, max_err

    def smooth_loss(U_mat, V_mat, W_mat, n, m, p):
        """Sum of squared errors over all tensor entries (target = 1)."""
        rank = U_mat.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    v_row = V_mat[j*p+k, :]
                    w_row = W_mat[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    total += (val - 1.0) ** 2
        return total

    # ---- Deterministic structured constructions (exploit rectangular shape) ----
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (schoolbook identity).
    # We seek rank-minimal U,V,W with integer/half-integer entries.

    best_U = best_V = best_W = None
    best_rank = None

    def record(U, V, W, rank):
        nonlocal best_U, best_V, best_W, best_rank
        if best_rank is None or rank < best_rank:
            best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Construction A: Group over the middle index j.
    # For a fixed j, the "j-slice" is the outer product of A-row-block (n x 1) and B-col-block (1 x p).
    # We can factor the all-ones n*p block using rank <= min(n,p) via a rank factorization.
    # T = sum_j (a_j outer b_j) where a_j in R^n, b_j in R^p.
    # Total rank = m * min(n,p).
    rA = min(n, p)
    rankA = m * rA
    U = np.zeros((n*m, rankA)); V = np.zeros((m*p, rankA)); W = np.zeros((p*n, rankA))
    # For each j, factor J_{n,p} (all ones) = A_j B_j^T with A_j (n x rA), B_j (p x rA).
    # Use a simple factorization: pick rA columns; A_j = ones(n, rA), B_j = ones(p, rA) scaled.
    # Then sum over c in rA of A_j[c]*B_j[c] = rA. We need it to equal 1, so scale by 1/rA.
    col = 0
    for j in range(m):
        for c in range(rA):
            U[j*n + np.arange(n), col] = 1.0 / rA   # A-row index i -> i*m? No: U index is i*m+j
            # Correct: U[i*m + j, col]
            for i in range(n):
                U[i*m + j, col] = 1.0 / rA
            for k in range(p):
                V[j*p + k, col] = 1.0
                W[k*n + i if False else 0, 0] = 0  # placeholder
            # W[k*n + i, col] must be 1 for all i,k (since it multiplies U*V which is 1/rA * 1)
            for i in range(n):
                for k in range(p):
                    W[k*n + i, col] = 1.0
            col += 1
    # Verify construction A
    ok, _ = verify_numpy(U, V, W, n, m, p)
    if ok:
        record(U, V, W, rankA)

    # Construction B: Group over i (row of A / row of C).
    # T = sum_i (row_i of A block) outer (col_i of C block)
    # For fixed i, slice is all-ones (m x p) block, factorable with rank min(m,p).
    rB = min(m, p)
    rankB = n * rB
    U = np.zeros((n*m, rankB)); V = np.zeros((m*p, rankB)); W = np.zeros((p*n, rankB))
    col = 0
    for i in range(n):
        for c in range(rB):
            for j in range(m):
                U[i*m + j, col] = 1.0 / rB
            for k in range(p):
                V[j*p + k if False else 0, 0] = 0
                for j in range(m):
                    V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
            col += 1
    ok, _ = verify_numpy(U, V, W, n, m, p)
    if ok:
        record(U, V, W, rankB)

    # Construction C: Group over k (col of B / col of C).
    # For fixed k, slice is all-ones (n x m) block, factorable with rank min(n,m).
    rC = min(n, m)
    rankC = p * rC
    U = np.zeros((n*m, rankC)); V = np.zeros((m*p, rankC)); W = np.zeros((p*n, rankC))
    col = 0
    for k in range(p):
        for c in range(rC):
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col] = 1.0 / rC
                V[j*p + k if False else 0, 0] = 0
            for j in range(m):
                V[j*p + k, col] = 1.0
            for i in range(n):
                W[k*n + i, col] = 1.0
            col += 1
    ok, _ = verify_numpy(U, V, W, n, m, p)
    if ok:
        record(U, V, W, rankC)

    # ---- Fallback: schoolbook (rank n*m*p) ----
    schoolbook_rank = n * m * p
    if best_rank is None or best_rank >= schoolbook_rank:
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        ok, _ = verify_numpy(U, V, W, n, m, p)
        if ok:
            record(U, V, W, schoolbook_rank)

    # ---- Cleanup: snap to integer/half-integer and remove near-zeros ----
    def clean(mat):
        out = np.zeros_like(mat)
        for idx in np.ndindex(mat.shape):
            v = mat[idx]
            if abs(v) < 1e-10:
                out[idx] = 0.0
            elif abs(v - round(v)) < 1e-9:
                out[idx] = float(round(v))
            elif abs(v - round(v*2)/2) < 1e-9:
                out[idx] = round(v*2)/2
            else:
                out[idx] = float(v)
        return out

    U_out = clean(best_U)
    V_out = clean(best_V)
    W_out = clean(best_W)

    # Final verification after cleanup
    ok, _ = verify_numpy(U_out, V_out, W_out, n, m, p, tol=1e-8)
    if not ok:
        # Fall back to a guaranteed-valid schoolbook (integers, always valid)
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        U_out, V_out, W_out = U, V, W

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
