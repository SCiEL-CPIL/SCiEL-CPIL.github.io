import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    N_U = n * m  # 8
    N_V = m * p  # 28
    N_W = p * n  # 14

    def elapsed():
        return time.time() - start_time

    # ---------- Build the known rank-50 Strassen + schoolbook solution ----------
    def build_base_solution():
        rank = 50
        U = np.zeros((N_U, rank), dtype=np.int64)
        V = np.zeros((N_V, rank), dtype=np.int64)
        W = np.zeros((N_W, rank), dtype=np.int64)
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)

            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1

            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1

            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1

            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1

            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0

            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)

        assert col == rank
        return U, V, W

    def verify(U, V, W):
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Uf[i*m+j, :] * Vf[j*p+k, :] * Wf[k*n+i, :])
                    if abs(val - 1.0) > 1e-7:
                        return False
        return True

    U, V, W = build_base_solution()
    base_ok = verify(U, V, W)
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = U.shape[1]

    # ---------- Rank-minimization via exact rational row reduction ----------
    # The multiplication tensor T[i,j,k]=delta[i,k] is a 3-tensor. A rank-r
    # decomposition exists iff there are U (N_U x r), V (N_V x r), W (N_W x r)
    # with sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1.
    # Fixing U and V, the constraint on W is linear: for each (i,j,k),
    #   sum_t (U[i*m+j,t]*V[j*p+k,t]) * W[k*n+i,t] = 1.
    # Define A[(i,j,k), t] = U[i*m+j,t]*V[j*p+k,t]. We need A W = 1.
    # A solution W (over Q) exists iff rank(A) == rank([A | 1]).
    # We search for small-rank U,V (integer, from a structured candidate set)
    # that admit such a W.

    def try_find_lower_rank(U, V, W, time_budget):
        """Attempt to find a lower-rank decomposition. Returns (U,V,W) or None."""
        # Candidate basis vectors for U and V: each entry is a 0/1/-1 pattern
        # over the row indices. We restrict to a small structured set to keep
        # the search tractable: for each row of U, pick a subset of columns to be
        # nonzero. We try reducing rank by zeroing columns of the base solution
        # and checking if the remaining columns still span a valid decomposition.

        base_rank = U.shape[1]
        # Try to drop columns from the base solution one at a time (greedy).
        # If a column is redundant (its contribution can be expressed via others),
        # we can drop it.
        # This is a heuristic: we check if removing column c still allows a valid
        # decomposition by solving for W with the remaining columns.

        def can_decompose_with_columns(U_sub, V_sub):
            r = U_sub.shape[1]
            # Build A matrix: (n*m*p) x r
            A = np.zeros((n*m*p, r), dtype=object)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = (i*m + j)*p + k
                        for t in range(r):
                            A[row_idx, t] = Fraction(U_sub[i*m+j, t]) * Fraction(V_sub[j*p+k, t])
            # Check if system A W = 1 is consistent over Q.
            # Use Gaussian elimination on [A | 1].
            augmented = []
            for row in range(n*m*p):
                augmented.append([A[row, t] for t in range(r)] + [Fraction(1)])
            # Row reduce
            rank_A = 0
            rank_aug = 0
            for col in range(r + 1):
                # Find pivot
                pivot = None
                for row in range(rank_aug, len(augmented)):
                    if augmented[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                augmented[rank_aug], augmented[pivot] = augmented[pivot], augmented[rank_aug]
                # Normalize
                pv = augmented[rank_aug][col]
                for c in range(col, r + 1):
                    augmented[rank_aug][c] /= pv
                # Eliminate
                for row in range(len(augmented)):
                    if row != rank_aug and augmented[row][col] != 0:
                        factor = augmented[row][col]
                        for c in range(col, r + 1):
                            augmented[row][c] -= factor * augmented[rank_aug][c]
                rank_aug += 1
            # Check consistency: any row with all zeros in A but nonzero in 1 -> inconsistent
            for row in range(len(augmented)):
                a_part = all(augmented[row][c] == 0 for c in range(r))
                b_part = augmented[row][r] != 0
                if a_part and b_part:
                    return None
            # Solve for W (one particular solution)
            W_sol = [Fraction(0)] * r
            for row in range(rank_aug):
                # Find leading column
                lead = None
                for c in range(r):
                    if augmented[row][c] != 0:
                        lead = c
                        break
                if lead is not None:
                    W_sol[lead] = augmented[row][r]
            return W_sol

        # Greedy column removal
        U_cur = U.copy()
        V_cur = V.copy()
        cols_to_try = list(range(base_rank))
        removed = []
        for c in cols_to_try:
            if elapsed() > time_budget:
                break
            U_test = np.delete(U_cur, c, axis=1)
            V_test = np.delete(V_cur, c, axis=1)
            W_sol = can_decompose_with_columns(U_test, V_test)
            if W_sol is not None:
                # Column c is redundant; remove it
                U_cur = U_test
                V_cur = V_test
                removed.append(c)
                # Update W to the new solution
                # Re-verify
                if verify(U_cur, V_cur, np.array(W_sol, dtype=np.int64)):
                    W = np.array(W_sol, dtype=np.int64)
                    best_rank = U_cur.shape[1]
                    best = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W.tolist()}

        if removed:
            return U_cur, V_cur, np.array(W_sol, dtype=np.int64)
        return None

    # Try rank reduction if we have time
    if elapsed() < deadline - 20.0:
        result = try_find_lower_rank(U, V, W, deadline - 5.0)
        if result is not None:
            U2, V2, W2 = result
            if verify(U2, V2, W2):
                best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                best_rank = U2.shape[1]

    # Final verification
    if not verify(np.array(best["U"]), np.array(best["V"]), np.array(best["W"])):
        # Fallback to base solution
        U, V, W = build_base_solution()
        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best
