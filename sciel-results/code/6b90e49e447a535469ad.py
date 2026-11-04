import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def check_time():
        return time.time() - t0 < DEADLINE

    # Build the list of 576 constraints: (U_idx, V_idx, W_idx) that must sum to 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i * m + j, j * p + k, k * n + i))
    assert len(constraints) == 576

    # Build the "known" part of the tensor: for each constraint index c, the
    # coefficient is 1. We want sum over rank terms of U[u,c]*V[v,c]*W[w,c] = 1
    # for the matching (u,v,w) and 0 for all other (u,v,w) triples.
    #
    # We'll use a random rank-r decomposition and solve for the entries.
    # The linear system: for each of the 576 "on" constraints, we have one equation.
    # But we also need all OFF-diagonal entries to be 0. There are 16*36*36 - 576
    # such entries, which is too many. Instead, we rely on the structure:
    # if the on-diagonal are satisfied AND the tensor has the correct "support",
    # it's valid. But that's not guaranteed.
    #
    # Better approach: only enforce the 576 on-diagonal = 1 constraints.
    # Then validate the full tensor. If off-diagonals leak, the validation fails.
    # With random entries, off-diagonal leakage is unlikely to be exactly 0,
    # so we need a smarter approach.
    #
    # Actually, let's use the standard trick: the matmul tensor T has T[u,v,w]=1
    # iff (u,v,w) is a valid (i,j,k) triple, else 0. We need sum_c u_c[v]*v_c[w]*w_c[u]
    # to equal T. We set up 576 equations (the "1" entries) and solve for the
    # unknowns. If the solution also makes all "0" entries zero, great.
    #
    # With r rank terms, we have r*(16+36+36) = 88r unknowns.
    # For r=576: 50688 unknowns, 576 equations -> very underdetermined.
    # We can fix some variables to reduce the system.
    #
    # Strategy: Fix U and V to random values, then solve for W.
    # W has 36*r unknowns. The 576 equations give 576 linear equations in 36r unknowns.
    # For r >= 16, 36r >= 576, so the system is underdetermined -> always solvable.
    # Then validate. If validation passes, we have a valid decomposition.
    #
    # But we want INTEGER/HALF-INTEGER entries. Random rationals won't be.
    # So we need to search for a solution with rational entries.
    #
    # Alternative: use the known 69-term construction as a starting point,
    # then try to find a lower-rank one by random perturbation + solving.
    #
    # Let me try a different approach: directly search for a low-rank
    # decomposition using the fact that we can fix U and V to simple values
    # (like 0/1) and solve for W.

    # --- Fallback: schoolbook ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = Fraction(1)
                    V[j * p + k][c] = Fraction(1)
                    W[k * n + i][c] = Fraction(1)
                    c += 1
        return U, V, W, r

    # --- Strassen 4x4x4 + schoolbook 4x4x5 (rank ~69) as guaranteed fallback ---
    def strassen_4x4_plus_schoolbook_4x5():
        S = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0]),
            ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),
            ([1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),
        ]
        terms_4x4 = {}
        for outer in range(7):
            oa, ob, oc = S[outer]
            for inner in range(7):
                ia, ib, ic = S[inner]
                for bi in range(4):
                    if oa[bi] == 0:
                        continue
                    bi_r, bi_c = divmod(bi, 2)
                    for ai in range(4):
                        if ia[ai] == 0:
                            continue
                        ai_r, ai_c = divmod(ai, 2)
                        a_row = 2 * bi_r + ai_r
                        a_col = 2 * bi_c + ai_c
                        U_idx = a_row * 4 + a_col
                        for bj in range(4):
                            if ob[bj] == 0:
                                continue
                            bj_r, bj_c = divmod(bj, 2)
                            for bji in range(4):
                                if ib[bji] == 0:
                                    continue
                                bji_r, bji_c = divmod(bji, 2)
                                b_row = 2 * bj_r + bji_r
                                b_col = 2 * bj_c + bji_c
                                V_idx_4x4 = b_row * 4 + b_col
                                for ci in range(4):
                                    if oc[ci] == 0:
                                        continue
                                    ci_r, ci_c = divmod(ci, 2)
                                    for ci2 in range(4):
                                        if ic[ci2] == 0:
                                            continue
                                        ci2_r, ci2_c = divmod(ci2, 2)
                                        c_row = 2 * ci_r + ci2_r
                                        c_col = 2 * ci_c + ci2_c
                                        W_idx_4x4 = c_row * 4 + c_col
                                        coeff = oa[bi] * ia[ai] * ob[bj] * ib[bji] * oc[ci] * ic[ci2]
                                        if coeff != 0:
                                            key = (U_idx, V_idx_4x4, W_idx_4x4)
                                            terms_4x4[key] = terms_4x4.get(key, 0) + coeff
        terms_4x4 = {k: v for k, v in terms_4x4.items() if v != 0}
        r_4x4 = len(terms_4x4)
        r_4x5 = 4 * 4 * 5
        r_total = r_4x4 + r_4x5

        U = [[0] * r_total for _ in range(nm)]
        V = [[0] * r_total for _ in range(mp)]
        W = [[0] * r_total for _ in range(pn)]
        c = 0
        for (u_idx, v_4x4, w_4x4), coeff in terms_4x4.items():
            b_row = v_4x4 // 4
            b_col = v_4x4 % 4
            v_global = b_row * 9 + b_col
            w_global = w_4x4
            U[u_idx][c] = Fraction(coeff)
            V[v_global][c] = Fraction(1)
            W[w_global][c] = Fraction(1)
            c += 1
        for i in range(4):
            for j in range(4):
                for k_local in range(5):
                    k_global = 4 + k_local
                    U[i * 4 + j][c] = Fraction(1)
                    V[j * 9 + k_global][c] = Fraction(1)
                    W[k_global * 4 + i][c] = Fraction(1)
                    c += 1
        return U, V, W, r_total

    def validate(U, V, W, r):
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = Fraction(0)
                    for c in range(r):
                        uc = Uj[c]
                        vc = Vk[c]
                        wc = Wk[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            tot += uc * vc * wc
                    if tot != Fraction(1):
                        return False
        # Check off-diagonal entries are 0
        for u_idx in range(nm):
            for v_idx in range(mp):
                for w_idx in range(pn):
                    # Check if this is a valid (i,j,k) triple
                    i = u_idx // m
                    j = u_idx % m
                    j2 = v_idx // p
                    k = v_idx % p
                    k2 = w_idx // n
                    i2 = w_idx % n
                    if i == j2 == i2 and j == k2:
                        continue  # this is a valid entry, skip
                    tot = Fraction(0)
                    Uj = U[u_idx]
                    Vk = V[v_idx]
                    Wk = W[w_idx]
                    for c in range(r):
                        uc = Uj[c]
                        vc = Vk[c]
                        wc = Wk[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            tot += uc * vc * wc
                    if tot != Fraction(0):
                        return False
        return True

    def validate_fast(U, V, W, r):
        """Float-based fast validation."""
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = 0.0
                    for c in range(r):
                        uc = Uj[c]
                        if uc == 0:
                            continue
                        vc = Vk[c]
                        if vc == 0:
                            continue
                        wc = Wk[c]
                        if wc == 0:
                            continue
                        tot += float(uc) * float(vc) * float(wc)
                    if abs(tot - 1.0) > 1e-9:
                        return False
        return True

    # Get the guaranteed 69-term decomposition
    U, V, W, r = strassen_4x4_plus_schoolbook_4x5()
    if not validate(U, V, W, r):
        U, V, W, r = schoolbook()
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    best_U, best_V, best_W, best_r = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W], r

    # ============================================================
    # Randomized search: try to find a lower-rank decomposition.
    # Fix U and V to random 0/1 patterns, solve for W via linear algebra.
    # ============================================================
    import numpy as np
    from fractions import Fraction as Fr

    def try_random_rank(r_target, seed):
        """Try to find a rank-r_target decomposition by fixing U,V randomly and solving for W."""
        rng = random.Random(seed)
        # U: 16 x r_target, entries in {0, 1, -1}
        # V: 36 x r_target, entries in {0, 1, -1}
        # W: 36 x r_target, to be solved
        U_rand = [[Fr(rng.choice([0, 1, -1])) for _ in range(r_target)] for _ in range(nm)]
        V_rand = [[Fr(rng.choice([0, 1, -1])) for _ in range(r_target)] for _ in range(mp)]

        # Build the linear system for W.
        # For each constraint (u, v, w) in constraints:
        #   sum_c U[u,c] * V[v,c] * W[w,c] = 1
        # This is a linear system in W[w,c] (36*r_target unknowns, 576 equations).
        # If 36*r_target > 576, it's underdetermined -> use least squares or
        # pick a particular solution.

        num_unknowns = pn * r_target
        if num_unknowns < 576:
            return None  # not enough degrees of freedom

        # Build matrix A (576 x num_unknowns) and vector b (576,)
        # A[eq, unknown] = U[u, c] * V[v, c] where unknown = w*n + ... wait
        # unknown index: w_idx * r_target + c
        # eq index: constraint index
        A = np.zeros((576, num_unknowns), dtype=float)
        b = np.ones(576, dtype=float)

        for eq_idx, (u_idx, v_idx, w_idx) in enumerate(constraints):
            for c in range(r_target):
                uc = float(U_rand[u_idx][c])
                vc = float(V_rand[v_idx][c])
                if uc == 0 or vc == 0:
                    continue
                unknown_idx = w_idx * r_target + c
                A[eq_idx, unknown_idx] = uc * vc
                b[eq_idx] = 1.0

        # Solve the underdetermined system: minimize ||W|| subject to A*W = b
        # Use np.linalg.lstsq
        try:
            W_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None

        # Check if the solution satisfies the on-diagonal constraints
        if residuals.size > 0 and residuals[0] > 1e-6:
            return None

        # Convert to Fraction (round to nearest half-integer)
        W_frac = [[Fr(0)] * r_target for _ in range(pn)]
        for w_idx in range(pn):
            for c in range(r_target):
                val = W_sol[w_idx * r_target + c]
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                W_frac[w_idx][c] = Fr(rounded, 2)

        # Validate
        U_list = [[x for x in row] for row in U_rand]
        V_list = [[x for x in row] for row in V_rand]
        if validate_fast(U_list, V_list, W_frac, r_target):
            if validate(U_list, V_list, W_frac, r_target):
                return U_list, V_list, W_frac, r_target
        return None

    # Try various ranks from low to high
    # Lower bound: max(16*4, 4*9, 9*4) = 36, but realistically higher.
    # Try ranks from 40 up to best_r - 1
    for r_target in range(40, best_r):
        if not check_time():
            break
        found = False
        for seed in range(50):
            if not check_time():
                break
            result = try_random_rank(r_target, seed)
            if result is not None:
                U, V, W, r = result
                if r < best_r:
                    best_U, best_V, best_W, best_r = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W], r
                found = True
                break
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue

    # Also try the approach of fixing W randomly and solving for U,V
    def try_random_rank_W_fixed(r_target, seed):
        """Fix W randomly, solve for U and V."""
        rng = random.Random(seed + 10000)
        W_rand = [[Fr(rng.choice([0, 1, -1])) for _ in range(r_target)] for _ in range(pn)]

        # Unknowns: U (16*r_target) + V (36*r_target) = 52*r_target
        num_unknowns = (nm + mp) * r_target
        if num_unknowns < 576:
            return None

        A = np.zeros((576, num_unknowns), dtype=float)
        b = np.ones(576, dtype=float)

        for eq_idx, (u_idx, v_idx, w_idx) in enumerate(constraints):
            for c in range(r_target):
                wc = float(W_rand[w_idx][c])
                if wc == 0:
                    continue
                # A[eq, u_unknown] * U[u,c] * V[v,c] -> this is bilinear, not linear!
                # Can't set up a linear system for both U and V simultaneously.
                # Skip this approach.
                pass
        return None

    # Try greedy elimination on the current best
    def try_eliminate_column(U, V, W, r, col):
        r_new = r - 1
        if r_new <= 0:
            return None
        U_new = [row[:col] + row[col + 1:] for row in U]
        V_new = [row[:col] + row[col + 1:] for row in V]
        W_new = [row[:col] + row[col + 1:] for row in W]
        if validate_fast(U_new, V_new, W_new, r_new):
            if validate(U_new, V_new, W_new, r_new):
                return U_new, V_new, W_new, r_new
        return None

    improved = True
    while improved and check_time():
        improved = False
        for col in range(best_r):
            if not check_time():
                break
            result = try_eliminate_column(best_U, best_V, best_W, best_r, col)
            if result is not None:
                best_U, best_V, best_W, best_r = result
                improved = True
                break

    # Final validation
    if not validate(best_U, best_V, best_W, best_r):
        U, V, W, r = schoolbook()
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    return {"U": [[float(x) for x in row] for row in best_U],
            "V": [[float(x) for x in row] for row in best_V],
            "W": [[float(x) for x in row] for row in best_W]}
