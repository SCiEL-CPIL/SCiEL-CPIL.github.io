import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Exact iff for every (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    R = len(U[0])
    # Precompute per-r outer of U and V
    # Use exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _try_rank17(n, m, p, seed, max_iters):
    # Unknowns: 17 vectors u_r in R^6, v_r in R^9, w_r in R^6.
    # Equations: 18 (one per (i,j,k)): sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
    # Fix w_1 = e_{k=0,i=0} to break scale symmetry; then 17 free w's.
    # For a given (u_1, v_1, w_2..w_17), the remaining 16 pairs (u_r, v_r)
    # can be solved via linear least squares over the 18 residual equations.
    #
    # Approach: iterate. Initialize u_1, v_1, w_2..w_17 randomly.
    # Then solve for (u_2..u_17, v_2..v_17) given w's fixed, via linear LSQ.
    # Update w's by gradient step toward satisfying equations. Repeat.

    rng = random.Random(seed)
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i, j, k)] = 1

    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # We'll represent: u_1 (6), v_1 (9), w_2..w_17 (16*6=96)
    # and solve for u_2..u_17 (16*6=96), v_2..v_17 (16*9=144)
    # Total unknowns in LSQ: 96 + 144 = 240, equations: 18. Underdetermined.
    # Use minimum-norm solution via pinv.

    def solve_uv_given_w(u1, v1, w_rest):
        # w_rest: list of 16 vectors of dim 6 (for r=2..17)
        # Equations: for each (i,j,k):
        #   u1[i,j]*v1[j,k]*w1[k,i] + sum_{r=2..17} u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
        # w1 is fixed = e_{(0,0)} i.e. w1[k*n+i] = 1 iff k==0 and i==0.
        w1 = [Fraction(0)] * (p * n)
        w1[w_idx(0, 0)] = Fraction(1)
        # Build residual for each (i,j,k) after subtracting the r=1 term.
        # For r>=2: term = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # We have variables u_r (r=2..17, dim 6) and v_r (r=2..17, dim 9).
        # The term is bilinear in (u_r, v_r). But we can linearize:
        # For each r, define a_r = v_r[j,k] * w_r[k,i] (depends on r, j, k, i)
        # and b_r = u_r[i,j] * w_r[k,i] (depends on r, i, j, k)
        # Actually it's a product u_r[i,j] * (v_r[j,k]*w_r[k,i]).
        # This is NOT linear in (u_r, v_r) jointly.
        #
        # Alternative: fix v_r and solve for u_r linearly, then fix u_r and solve for v_r.
        # We'll do alternating: given v_r, solve u_r linearly; given u_r, solve v_r linearly.

        # But the LSQ over all 16 r simultaneously is bilinear. Let's do it differently:
        # Treat each r independently? No, they're coupled through the sum.
        #
        # Simpler: use a rank-1 decomposition approach.
        # For the 16 free rank-1 terms, we need:
        #   sum_{r=2..17} (u_r ⊗ v_r ⊗ w_r)[i,j,k] = target[i,j,k] - (u1⊗v1⊗w1)[i,j,k]
        # where (u⊗v⊗w)[i,j,k] = u[i,j]*v[j,k]*w[k,i].
        #
        # Given w_2..w_17 fixed, define for each r: A_r[i,j,k] = w_r[k,i]
        # Then we need sum_r u_r[i,j]*v_r[j,k]*A_r[i,j,k] = RHS[i,j,k].
        # This is still bilinear in (u_r, v_r).
        #
        # Let's just do a simple iterative approach:
        # 1. Initialize u_r, v_r randomly.
        # 2. Given v_r and w_r, solve for u_r linearly (18 eqs, 96 unknowns -> min norm).
        # 3. Given u_r and w_r, solve for v_r linearly (18 eqs, 144 unknowns -> min norm).
        # 4. Update w_r via gradient descent.
        # 5. Repeat until convergence or deadline.

        return None  # Not used directly; we do the iteration in the main loop

    # Main iterative solver
    R = 17
    # Initialize
    u = [[rng.uniform(-1, 1) for _ in range(n * m)] for _ in range(R)]
    v = [[rng.uniform(-1, 1) for _ in range(m * p)] for _ in range(R)]
    w = [[rng.uniform(-1, 1) for _ in range(p * n)] for _ in range(R)]
    # Fix w[0] = e_{(0,0)}
    w[0] = [0.0] * (p * n)
    w[0][w_idx(0, 0)] = 1.0

    # Precompute target as flat array
    tgt = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[i * m * p + j * p + k] = 1.0

    def compute_residual(u, v, w):
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    res[i * m * p + j * p + k] = s - 1.0
        return res

    def solve_u_given_vw(v, w):
        # For each (i,j,k): sum_r u_r[i,j] * (v_r[j,k]*w_r[k,i]) = 1
        # Variables: u_r[i,j] for r=1..17, i in range(n), j in range(m) -> 17*6 = 102
        # But u_0[i,j] is also a variable. However we fixed w_0, not u_0.
        # All u_r are free.
        # Build A matrix: 18 eqs x 102 unknowns
        # Unknown index: r*(n*m) + i*m + j
        A = np.zeros((n * m * p, R * n * m))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, r * n * m + i * m + j] = v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    rhs[eq] = 1.0
                    eq += 1
        # Min-norm solution
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        u_new = [[0.0] * (n * m) for _ in range(R)]
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    u_new[r][i * m + j] = sol[r * n * m + i * m + j]
        return u_new

    def solve_v_given_uw(u, w):
        # For each (i,j,k): sum_r v_r[j,k] * (u_r[i,j]*w_r[k,i]) = 1
        # Variables: v_r[j,k] for r=1..17, j in range(m), k in range(p) -> 17*9 = 153
        A = np.zeros((n * m * p, R * m * p))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, r * m * p + j * p + k] = u[r][u_idx(i, j)] * w[r][w_idx(k, i)]
                    rhs[eq] = 1.0
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        v_new = [[0.0] * (m * p) for _ in range(R)]
        for r in range(R):
            for j in range(m):
                for k in range(p):
                    v_new[r][j * p + k] = sol[r * m * p + j * p + k]
        return v_new

    def update_w(u, v, w, lr):
        # Gradient of sum_{i,j,k} (sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] - 1)^2 w.r.t. w_r[k,i]
        # = 2 * sum_{i,j,k} res[i,j,k] * u_r[i,j]*v_r[j,k] * (delta_{k,i} matches)
        # For a given w_r[k0,i0], it appears in equation (i0, j, k0) for all j.
        # d/dw_r[k0,i0] = 2 * sum_j res[i0, j, k0] * u_r[i0, j] * v_r[j, k0]
        w_new = [row[:] for row in w]
        for r in range(1, R):  # w[0] is fixed
            for k0 in range(p):
                for i0 in range(n):
                    grad = 0.0
                    for j in range(m):
                        res_val = 0.0
                        for rr in range(R):
                            res_val += u[rr][u_idx(i0, j)] * v[rr][v_idx(j, k0)] * w[rr][w_idx(k0, i0)]
                        res_val -= 1.0
                        grad += res_val * u[r][u_idx(i0, j)] * v[r][v_idx(j, k0)]
                    w_new[r][w_idx(k0, i0)] -= lr * grad
        return w_new

    best_res = None
    best_state = None
    for it in range(max_iters):
        if deadline_exceeded():
            break
        # Solve u given v, w
        u_new = solve_u_given_vw(v, w)
        if u_new is None:
            break
        u = u_new
        # Solve v given u, w
        v_new = solve_v_given_uw(u, w)
        if v_new is None:
            break
        v = v_new
        # Update w
        w = update_w(u, v, w, 0.01)
        # Recompute residual
        res = compute_residual(u, v, w)
        norm = float(np.linalg.norm(res))
        if best_res is None or norm < best_res:
            best_res = norm
            best_state = (u[:], v[:], [row[:] for row in w])
            if norm < 1e-8:
                break

    if best_state is None:
        return None

    u, v, w = best_state
    # Convert to Fraction and verify
    U_frac = [[Fraction(round(x, 12)) for x in row] for row in u]
    V_frac = [[Fraction(round(x, 12)) for x in row] for row in v]
    W_frac = [[Fraction(round(x, 12)) for x in row] for row in w]

    # Round to nearest half-integer if close
    def round_half(x):
        h = x * 2
        r = round(h)
        if abs(h - r) < 1e-6:
            return Fraction(r, 2)
        return x

    U_frac = [[round_half(x) for x in row] for row in U_frac]
    V_frac = [[round_half(x) for x in row] for row in V_frac]
    W_frac = [[round_half(x) for x in row] for row in W_frac]

    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U_frac]
        V_list = [[float(x) for x in row] for row in V_frac]
        W_list = [[float(x) for x in row] for row in W_frac]
        return {"U": U_list, "V": V_list, "W": W_list}

    return None


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)

    # Try a few random seeds for rank-17 search
    for seed in range(20):
        if deadline_exceeded():
            break
        result = _try_rank17(n, m, p, seed, max_iters=200)
        if result is not None:
            return result

    return fallback


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
