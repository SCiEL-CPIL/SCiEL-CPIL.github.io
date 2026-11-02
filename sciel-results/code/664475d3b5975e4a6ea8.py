"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = nm * mp  # n*m*p

    # The multiplication tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Strategy: For a given rank R, fix U and V (random small integers), then solve
    # for W. The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    #
    # This is a system of n*m*p equations in p*n*R unknowns (W entries).
    # We can group by i: for fixed i, we have m*p equations in p*R unknowns (W[k*n+i, c] for k,c).
    # Actually, let's think differently.
    #
    # For each i, define A_i[k, c] = V[j*p+k, c] * ... no, that's not right.
    #
    # Let me restructure. For fixed i, the equations are:
    #   sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1, for all j in [0,m), k in [0,p)
    #
    # Let W_i[k, c] = W[k*n+i, c]. Then:
    #   sum_c U_i[j, c] * V_jk[c] * W_i[k, c] = 1
    # where U_i[j,c] = U[i*m+j, c] and V_jk[c] = V[j*p+k, c].
    #
    # This is still coupled across c. Let's use a different approach:
    #
    # The full system: We have n*m*p equations. Unknowns are W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,R).
    # Total unknowns: p*n*R.
    #
    # For a random U and V, the system may be overdetermined (n*m*p > p*n*R when m > R... no, n*m*p vs p*n*R = n*p*R, so m*p*n vs n*p*R, i.e., m vs R. If R < m, overdetermined).
    #
    # Better approach: Use the "slice" method. Fix U and V. Then for each column c, define:
    #   S_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c]
    # We need sum_c S_c[i,j,k] * W[k*n+i, c] = 1.
    #
    # This is a linear system in the W variables. Let's build it.
    # Variables: W[k*n+i, c] for i in range(n), k in range(p), c in range(R). Total: n*p*R variables.
    # Equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1. Total: n*m*p equations.
    #
    # We can solve this using least squares or check consistency.

    def try_rank(R, num_trials=20):
        """Try to find a rank-R decomposition. Returns (U, V, W) or None."""
        random.seed(42 + R)
        for trial in range(num_trials):
            # Random U and V with small integer entries
            U = np.zeros((nm, R), dtype=object)
            V = np.zeros((mp, R), dtype=object)
            for idx in range(nm):
                for c in range(R):
                    U[idx, c] = random.randint(-2, 2)
            for idx in range(mp):
                for c in range(R):
                    V[idx, c] = random.randint(-2, 2)

            # Build linear system for W
            # Variables indexed by (i, k, c) -> var_idx = (i * p + k) * R + c
            # Equations indexed by (i, j, k) -> eq_idx = i * m * p + j * p + k
            num_vars = n * p * R
            num_eqs = n * m * p

            # Build matrix A (num_eqs x num_vars) and vector b (num_eqs,)
            A = np.zeros((num_eqs, num_vars), dtype=object)
            b = np.ones(num_eqs, dtype=object)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(R):
                            var_idx = (i * p + k) * R + c
                            A[eq_idx, var_idx] = U[i * m + j, c] * V[j * p + k, c]

            # Solve using Gaussian elimination with fractions
            # Augmented matrix [A | b]
            aug = np.zeros((num_eqs, num_vars + 1), dtype=object)
            for r in range(num_eqs):
                for c in range(num_vars):
                    aug[r, c] = A[r, c]
                aug[r, num_vars] = b[r]

            # Gaussian elimination
            row = 0
            for col in range(num_vars):
                if time.time() > deadline:
                    return None
                # Find pivot
                pivot = None
                for r in range(row, num_eqs):
                    if aug[r, col] != 0:
                        pivot = r
                        break
                if pivot is None:
                    continue
                # Swap rows
                if pivot != row:
                    aug[row], aug[pivot] = aug[pivot], aug[row]
                # Normalize
                pivot_val = aug[row, col]
                for c in range(col, num_vars + 1):
                    aug[row, c] = aug[row, c] / pivot_val
                # Eliminate
                for r in range(num_eqs):
                    if r != row and aug[r, col] != 0:
                        factor = aug[r, col]
                        for c in range(col, num_vars + 1):
                            aug[r, c] = aug[r, c] - factor * aug[row, c]
                row += 1
                if row == num_vars:
                    break

            # Check for inconsistency
            consistent = True
            for r in range(row, num_eqs):
                if aug[r, num_vars] != 0:
                    consistent = False
                    break
            if not consistent:
                continue

            # Check that all variables are determined (no free variables that would make it non-unique)
            # Actually, we just need any solution. Let's extract W.
            # For free variables, set them to 0.
            W_flat = np.zeros(num_vars, dtype=object)
            for r in range(row):
                # Find which variable this row corresponds to
                # (In reduced row echelon form, each pivot row has a 1 in some column)
                pass

            # Simpler: just extract from the augmented matrix assuming it's in RREF
            # Actually, let me just verify by computing W from the solution
            # For simplicity, let's use a different extraction:
            # In RREF, for each pivot row r, the pivot column is where the 1 is.
            W_flat = np.zeros(num_vars, dtype=object)
            for r in range(row):
                # Find pivot column
                for c in range(num_vars):
                    if aug[r, c] == 1:
                        W_flat[c] = aug[r, num_vars]
                        break
                # Non-pivot variables are 0

            # Reshape W
            W = np.zeros((pn, R), dtype=object)
            for i in range(n):
                for k in range(p):
                    for c in range(R):
                        var_idx = (i * p + k) * R + c
                        W[k * n + i, c] = W_flat[var_idx]

            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s = s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return (U, V, W)

        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But tensor rank can be higher. Try from 30 upward.
    lower_bound = max(nm, mp, pn)
    for R in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        result = try_rank(R, num_trials=10)
        if result is not None:
            best = result
            best_rank = R
            break  # Found minimal rank, stop

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
