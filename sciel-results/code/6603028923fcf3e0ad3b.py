"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Fallback: schoolbook rank 50 (always valid)
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def to_half(x):
        return round(x * 2.0) / 2.0

    def verify(U_mat, V_mat, W_mat):
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for c in range(U_mat.shape[1]):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Try rank r: solve for W given fixed U, V structure
    # U[i*m+j, c] = 1 if c == i*m+j (when i*m+j < r)
    # V[j*p+k, c] = 1 if c == j*p+k (when j*p+k < r)
    def try_rank(r):
        if time.time() > deadline - 5.0:
            return None
        
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        for i in range(n):
            for j in range(m):
                c = i * m + j
                if c < r:
                    U[i * m + j, c] = 1.0
        for j in range(m):
            for k in range(p):
                c = j * p + k
                if c < r:
                    V[j * p + k, c] = 1.0

        # Build linear system: for each (i,k,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        num_eq = n * p * m
        num_unk = p * n * r
        A_mat = np.zeros((num_eq, num_unk), dtype=float)
        b_vec = np.ones(num_eq, dtype=float)
        eq = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for c in range(r):
                        val = U[i * m + j, c] * V[j * p + k, c]
                        if val != 0.0:
                            w = (k * n + i) * r + c
                            A_mat[eq, w] = val
                    eq += 1

        try:
            sol, residuals, rank_lev, svals = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            return None

        W = sol.reshape(p * n, r)

        # Check exactness
        max_err = verify(U, V, W)
        if max_err < 1e-9:
            # Round to half-integers
            U_out = [[to_half(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_out = [[to_half(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_out = [[to_half(W[i, c]) for c in range(r)] for i in range(p * n)]
            
            # Re-verify after rounding
            U_r = np.array(U_out, dtype=float)
            V_r = np.array(V_out, dtype=float)
            W_r = np.array(W_out, dtype=float)
            max_err2 = verify(U_r, V_r, W_r)
            if max_err2 < 1e-9:
                return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    # Try ranks from 25 down to 10
    for r in range(25, 9, -1):
        if time.time() > deadline - 10.0:
            break
        result = try_rank(r)
        if result is not None:
            return result

    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
