"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Build the list of nonzero tensor entries
    # T[a, b, c] = 1 where a = k*n+i, b = i*m+j, c = j*p+k
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                entries.append((a, b, c, 1))

    # Precompute index lists for fast ALS
    # For each b, list of (a, c) pairs
    b_to_ac = {}
    for a, b, c, t in entries:
        if b not in b_to_ac:
            b_to_ac[b] = []
        b_to_ac[b].append((a, c))

    # For each c, list of (a, b) pairs
    c_to_ab = {}
    for a, b, c, t in entries:
        if c not in c_to_ab:
            c_to_ab[c] = []
        c_to_ab[c].append((a, b))

    # For each a, list of (b, c) pairs
    a_to_bc = {}
    for a, b, c, t in entries:
        if a not in a_to_bc:
            a_to_bc[a] = []
        a_to_bc[a].append((b, c))

    def verify_decomp(U, V, W, tol=1e-8):
        """Verify that the decomposition exactly matches the tensor."""
        for a, b, c, t in entries:
            val = np.dot(U[b, :], V[c, :] * W[a, :])
            if abs(val - t) > tol:
                return False
        return True

    def verify_exact(U, V, W):
        """Verify with exact rational arithmetic (for small integer/half-integer entries)."""
        for a, b, c, t in entries:
            s = Fraction(0)
            for r_idx in range(U.shape[1]):
                u_val = Fraction(U[b, r_idx]).limit_denominator(1000)
                v_val = Fraction(V[c, r_idx]).limit_denominator(1000)
                w_val = Fraction(W[a, r_idx]).limit_denominator(1000)
                s += u_val * v_val * w_val
            if s != Fraction(t):
                return False
        return True

    def polish(U, V, W, max_iter=100, tol=1e-12):
        """High-precision polish pass using ALS with tighter tolerance."""
        r = U.shape[1]
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Update U
            for b in range(n * m):
                ac_list = b_to_ac.get(b, [])
                if not ac_list:
                    continue
                A_mat = np.zeros((len(ac_list), r))
                t_vec = np.ones(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A_mat[idx, :] = V[c, :] * W[a, :]
                if len(ac_list) >= r:
                    try:
                        U[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

            # Update V
            for c in range(m * p):
                ab_list = c_to_ab.get(c, [])
                if not ab_list:
                    continue
                A_mat = np.zeros((len(ab_list), r))
                t_vec = np.ones(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A_mat[idx, :] = U[b, :] * W[a, :]
                if len(ab_list) >= r:
                    try:
                        V[c, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

            # Update W
            for a in range(p * n):
                bc_list = a_to_bc.get(a, [])
                if not bc_list:
                    continue
                A_mat = np.zeros((len(bc_list), r))
                t_vec = np.ones(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A_mat[idx, :] = U[b, :] * V[c, :]
                if len(bc_list) >= r:
                    try:
                        W[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

        return U, V, W

    def solve_rank(r, max_iter=80, seed=0):
        """Try to find a rank-r decomposition using vectorized ALS."""
        if time.time() > deadline - 40:
            return None

        rng = np.random.RandomState(seed)

        # Initialize with small random values
        U = rng.randn(n * m, r) * 0.01
        V = rng.randn(m * p, r) * 0.01
        W = rng.randn(p * n, r) * 0.01

        prev_err = np.inf
        for iteration in range(max_iter):
            if time.time() > deadline - 25:
                return None

            # Update U: for each b, solve linear system
            for b in range(n * m):
                ac_list = b_to_ac.get(b, [])
                if not ac_list:
                    continue
                A_mat = np.zeros((len(ac_list), r))
                t_vec = np.ones(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A_mat[idx, :] = V[c, :] * W[a, :]
                if len(ac_list) >= r:
                    try:
                        U[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

            # Update V: for each c, solve linear system
            for c in range(m * p):
                ab_list = c_to_ab.get(c, [])
                if not ab_list:
                    continue
                A_mat = np.zeros((len(ab_list), r))
                t_vec = np.ones(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A_mat[idx, :] = U[b, :] * W[a, :]
                if len(ab_list) >= r:
                    try:
                        V[c, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

            # Update W: for each a, solve linear system
            for a in range(p * n):
                bc_list = a_to_bc.get(a, [])
                if not bc_list:
                    continue
                A_mat = np.zeros((len(bc_list), r))
                t_vec = np.ones(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A_mat[idx, :] = U[b, :] * V[c, :]
                if len(bc_list) >= r:
                    try:
                        W[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass

            # Check convergence
            max_err = 0.0
            for a, b, c, t in entries:
                val = np.dot(U[b, :], V[c, :] * W[a, :])
                err = abs(val - t)
                if err > max_err:
                    max_err = err

            if max_err < 1e-6:
                # Polish
                U, V, W = polish(U, V, W, max_iter=50, tol=1e-12)
                if verify_decomp(U, V, W, tol=1e-8):
                    return U, V, W
                return None

            if abs(max_err - prev_err) < 1e-10 and max_err < 1e-3:
                # Stuck, try polish
                U, V, W = polish(U, V, W, max_iter=30, tol=1e-12)
                if verify_decomp(U, V, W, tol=1e-8):
                    return U, V, W
                return None

            prev_err = max_err

        # Final check
        if verify_decomp(U, V, W, tol=1e-8):
            U, V, W = polish(U, V, W, max_iter=30, tol=1e-12)
            if verify_decomp(U, V, W, tol=1e-8):
                return U, V, W

        return None

    # Try ranks from lower to higher
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But for matrix multiplication, the rank is at least max(n, m, p) * min(n, m, p) in general
    # Actually the trivial lower bound is max(n*m, m*p, p*n) for the tensor to be non-zero
    # Let's start from a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)
    # Actually for matrix mult, rank >= max(n*m, m*p, p*n) is not quite right
    # The tensor has n*m*p non-zero entries, so rank >= n*m*p / max_possible_per_term
    # Each rank-1 term can cover at most min(n*m, m*p, p*n) entries
    # So rank >= n*m*p / min(n*m, m*p, p*n) = max(n, m, p)
    # But we need a better bound. Let's just start from a reasonable value.
    start_rank = max(30, lower_bound)

    best_result = None
    for r in range(start_rank, min(rank_school + 1, 120)):
        if time.time() > deadline - 30:
            break

        # Try multiple seeds for better chance of convergence
        found = False
        for seed in range(r, r + 3):
            if time.time() > deadline - 25:
                break
            result = solve_rank(r, max_iter=60, seed=seed)
            if result is not None:
                best_result = result
                found = True
                break

        if found:
            break

    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school

    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for x in row:
                # Round to nearest half-integer
                half = round(x * 2) / 2
                if abs(x - half) < 1e-6:
                    row_list.append(int(half) if half == int(half) else half)
                else:
                    row_list.append(float(x))
            result.append(row_list)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
