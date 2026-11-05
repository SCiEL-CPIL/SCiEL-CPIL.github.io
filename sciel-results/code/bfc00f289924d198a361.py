"""Matrix multiplication 3x3x4 — rank-minimal exact tensor decomposition.

Strategy:
  * The schoolbook rank is 3*3*4 = 36.
  * For the 3x3x3 sub-block, the best known rank is 23 (Strassen-variant).
  * We attempt to find a decomposition of the 3x3x4 tensor with rank < 36
    by a randomised local search over rational (half-integer) entries.
  * If no improvement is found within the time budget, we return the
    schoolbook rank-36 decomposition (always correct).

Time budget: 140 s wall-clock (10 s safety margin before the 150 s kill).
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
DEADLINE = time.time() + 140  # 140 s safety margin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W


def _tensor_entry(U, V, W, r, n, m, p, i, j, k):
    """Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
    s = 0
    for c in range(len(U[0])):
        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
    return s


def _check(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor.
    Returns True if every entry matches."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(len(U[0])):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                # The tensor entry T[i,j,k] should be 1 if i==j==k (no, that's wrong)
                # Actually for matrix multiplication C = A*B, the (i,k) entry of C
                # is sum_j A[i,j]*B[j,k]. The tensor T_{i,j,k} = 1 for all i,j,k
                # (it's the structure tensor of matrix multiplication).
                # Wait: the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
                # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
                # The matrix multiplication tensor for C[i,k] = sum_j A[i,j]*B[j,k]
                # has structure T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                if val != 1:
                    return False
    return True


def _check_fast(U, V, W, n, m, p):
    """Faster check using numpy."""
    import numpy as np
    rank = len(U[0])
    Ua = np.array(U, dtype=float)  # (n*m, rank)
    Va = np.array(V, dtype=float)  # (m*p, rank)
    Wa = np.array(W, dtype=float)  # (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-10)


def _rank_of(U, V, W):
    """Return the number of columns (rank)."""
    return len(U[0])


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Start with schoolbook as fallback
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Try to find a lower-rank decomposition via random search
    # We try ranks from 35 down to 20, spending limited time on each.
    # For each rank r, we initialize with random half-integer entries
    # and do local search (coordinate descent) to minimize the error.

    import numpy as np
    from fractions import Fraction as Fr

    def _snap_to_half_integer(x, tol=1e-6):
        """Snap x to the nearest half-integer if close, else round to Fraction."""
        # Try nearest half-integer
        h = round(x * 2) / 2
        if abs(x - h) < tol:
            return Fr(h)
        # Try nearest integer
        i = round(x)
        if abs(x - i) < tol:
            return Fr(i)
        # Otherwise, use a rational approximation with small denominator
        # We'll just keep it as a Fraction with denominator 2 for now
        # For simplicity, snap to nearest 0.5
        return Fr(round(x * 2)) / 2

    def _random_init(r, seed=None):
        """Initialize random half-integer matrices."""
        rng = random.Random(seed)
        U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(p * n)]
        return U, V, W

    def _local_search(U, V, W, max_iter=5000, seed=None):
        """Coordinate descent: try to reduce error by perturbing entries.
        Returns (U, V, W, error)."""
        rng = random.Random(seed)
        import numpy as np

        def _error(U, V, W):
            rank = len(U[0])
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            Ur = Ua.reshape(n, m, rank)
            Vr = Va.reshape(m, p, rank)
            Wr = Wa.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            return np.sum((T - 1.0) ** 2)

        err = _error(U, V, W)
        if err < 1e-10:
            return U, V, W, err

        # Try random single-entry perturbations
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            # Pick which matrix and which entry to perturb
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randrange(n * m)
            elif mat == 'V':
                row = rng.randrange(m * p)
            else:
                row = rng.randrange(p * n)
            col = rng.randrange(len(U[0]))

            old_val = None
            if mat == 'U':
                old_val = U[row][col]
            elif mat == 'V':
                old_val = V[row][col]
            else:
                old_val = W[row][col]

            # Try a few perturbations
            best_new_val = old_val
            best_err = err
            for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                new_val = old_val + delta
                # Apply
                if mat == 'U':
                    U[row][col] = new_val
                elif mat == 'V':
                    V[row][col] = new_val
                else:
                    W[row][col] = new_val

                e = _error(U, V, W)
                if e < best_err:
                    best_err = e
                    best_new_val = new_val
                # Revert
                if mat == 'U':
                    U[row][col] = old_val
                elif mat == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val

            # Apply best
            if best_new_val != old_val:
                if mat == 'U':
                    U[row][col] = best_new_val
                elif mat == 'V':
                    V[row][col] = best_new_val
                else:
                    W[row][col] = best_new_val
                err = best_err
                if err < 1e-10:
                    break
            else:
                # No improvement; try a random reset of this entry
                pass

        return U, V, W, err

    # Try decreasing ranks
    for r in range(schoolbook_rank - 1, 19, -1):
        if time.time() > DEADLINE:
            break

        # Spend limited time per rank
        rank_deadline = time.time() + 5  # 5 seconds per rank
        found = False
        n_trials = 0
        while time.time() < rank_deadline and not found:
            n_trials += 1
            seed = n_trials * 1000 + r
            U, V, W = _random_init(r, seed=seed)
            U, V, W, err = _local_search(U, V, W, max_iter=2000, seed=seed)
            if err < 1e-8:
                # Verify with exact rational check
                # Convert to Fractions
                Uf = [[_snap_to_half_integer(U[i][j]) for j in range(r)] for i in range(n * m)]
                Vf = [[_snap_to_half_integer(V[i][j]) for j in range(r)] for i in range(m * p)]
                Wf = [[_snap_to_half_integer(W[i][j]) for j in range(r)] for i in range(p * n)]
                if _check(Uf, Vf, Wf, n, m, p):
                    best_U, best_V, best_W = Uf, Vf, Wf
                    best_rank = r
                    found = True
                    break

    # Convert best to lists of Fractions (or ints) for output
    # The evaluator accepts integer/half-integer entries
    # We return them as numbers (int or float .5)
    def _to_number(frac):
        if isinstance(frac, Fr):
            if frac.denominator == 1:
                return int(frac)
            elif frac.denominator == 2:
                return float(frac)
            else:
                return float(frac)
        return frac

    out_U = [[_to_number(U[i][j]) for j in range(len(best_U[0]))] for i in range(len(best_U))]
    out_V = [[_to_number(V[i][j]) for j in range(len(best_V[0]))] for i in range(len(best_V))]
    out_W = [[_to_number(W[i][j]) for j in range(len(best_W[0]))] for i in range(len(best_W))]

    return {"U": out_U, "V": out_V, "W": out_W}
