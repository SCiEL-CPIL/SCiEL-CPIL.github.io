"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 130.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 48."""
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

    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def round_to_half(vals):
        """Round list of floats to nearest half-integer, return as int or float."""
        out = []
        for x in vals:
            r = round(x * 2) / 2
            if r == int(r):
                out.append(int(r))
            else:
                out.append(r)
        return out

    def round_mat(M):
        """Round a 2D list of floats to half-integers."""
        return [round_to_half(row) for row in M]

    def compute_residual(U, V, W, rank):
        """Vectorized residual computation. Returns max |1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]|."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized dot product
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    res = abs(1.0 - s)
                    if res > max_res:
                        max_res = res
        return max_res

    def try_optimize(rank, budget_sec=10.0, seed=0):
        """Try to find a rank-`rank` decomposition using alternating least squares.
           Returns (U, V, W) as lists of lists, or None."""
        if elapsed() > DEADLINE - budget_sec:
            return None

        np.random.seed(seed)

        # Better initialization: random Gaussian scaled
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5

        prev_res = float('inf')
        for it in range(500):
            if elapsed() > DEADLINE - 5:
                return None

            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # Build p x rank matrix B where B[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    B = V[j * p:(j + 1) * p, :] * W[np.arange(p) * n + i, :]
                    try:
                        u, *_ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass

            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # Build n x rank matrix B where B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    B = np.zeros((n, rank), dtype=np.float64)
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, *_ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass

            # Update W: for each (i,k), solve linear system
            for i in range(n):
                for k in range(p):
                    # Build m x rank matrix B where B[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    B = np.zeros((m, rank), dtype=np.float64)
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, *_ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass

            # Check convergence
            max_res = compute_residual(U, V, W, rank)
            
            if max_res < 1e-10:
                # Try rounding to half-integers and verify
                U_r = round_mat(U.tolist())
                V_r = round_mat(V.tolist())
                W_r = round_mat(W.tolist())
                if verify_decomp(U_r, V_r, W_r, rank):
                    return U_r, V_r, W_r
                # Also try rounding to integers
                U_ri = [[int(round(x)) for x in row] for row in U.tolist()]
                V_ri = [[int(round(x)) for x in row] for row in V.tolist()]
                W_ri = [[int(round(x)) for x in row] for row in W.tolist()]
                if verify_decomp(U_ri, V_ri, W_ri, rank):
                    return U_ri, V_ri, W_ri

            # Early termination if stuck
            if max_res > prev_res * 0.999 and it > 50:
                break
            prev_res = max_res

            # Periodic rounding check
            if it % 100 == 99:
                U_r = round_mat(U.tolist())
                V_r = round_mat(V.tolist())
                W_r = round_mat(W.tolist())
                if verify_decomp(U_r, V_r, W_r, rank):
                    return U_r, V_r, W_r

        return None

    # First: get schoolbook as fallback
    U_sch, V_sch, W_sch = schoolbook()

    # Try to find lower rank: 47, 46, ...
    # Use multiple seeds for each rank
    for target_rank in range(47, max(n*m, m*p, p*n) - 1, -1):
        if elapsed() > DEADLINE - 15:
            break
        for seed in [42 + target_rank, 1000 + target_rank, 2000 + target_rank]:
            if elapsed() > DEADLINE - 10:
                break
            found = try_optimize(target_rank, budget_sec=5.0, seed=seed)
            if found is not None:
                U, V, W = found
                if verify_decomp(U, V, W, target_rank):
                    return {"U": U, "V": V, "W": W}

    # Return schoolbook
    return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
