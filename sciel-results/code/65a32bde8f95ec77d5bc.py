"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 12.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5

    def _remaining():
        return DEADLINE - SAFETY_MARGIN - (time.time() - start_time)

    def _verify(U, V, W, rank):
        """Verify exact decomposition. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def _try_merge_pair(U, V, W, rank, r1, r2):
        """
        Try to merge two rank-1 terms (columns r1, r2) into a single term.
        Uses rational arithmetic (Gaussian elimination) for an exact solution.
        Returns (new_U, new_V, new_W, new_rank) or None.
        """
        w1 = W[:, r1].astype(float)
        w2 = W[:, r2].astype(float)
        u1 = U[:, r1].astype(float)
        u2 = U[:, r2].astype(float)
        v1 = V[:, r1].astype(float)
        v2 = V[:, r2].astype(float)

        # Target tensor M[i,j,k] = u1*v1*w1 + u2*v2*w2 (term-by-term)
        M = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i, j, k] = (u1[i*m+j]*v1[j*p+k]*w1[k*n+i] +
                                  u2[i*m+j]*v2[j*p+k]*w2[k*n+i])

        # Candidate combined W vectors. We try several; for each we solve for
        # u_new, v_new via exact rational Gaussian elimination.
        candidates = []
        candidates.append(w1)
        candidates.append(w2)
        candidates.append(w1 + w2)
        candidates.append(w1 - w2)
        # Also try a small grid of linear combos a*w1 + b*w2 for small integers.
        for a in (-2, -1, 1, 2):
            for b in (-2, -1, 1, 2):
                cand = a*w1 + b*w2
                if np.any(np.abs(cand) > 1e-12):
                    candidates.append(cand)

        # Deduplicate
        seen = set()
        uniq = []
        for c in candidates:
            key = tuple(np.round(c, 9))
            if key not in seen:
                seen.add(key)
                uniq.append(c)

        for w_new in uniq:
            if np.all(np.abs(w_new) < 1e-12):
                continue
            if not _solve_merge(M, w_new):
                continue
            # _solve_merge returns (u_new, v_new) on success
            u_new, v_new = _solve_merge(M, w_new)
            if u_new is None:
                continue

            new_rank = rank - 1
            new_U = np.zeros((n*m, new_rank))
            new_V = np.zeros((m*p, new_rank))
            new_W = np.zeros((p*n, new_rank))

            col_map = [r for r in range(rank) if r != r1 and r != r2]
            for new_col, old_col in enumerate(col_map):
                new_U[:, new_col] = U[:, old_col]
                new_V[:, new_col] = V[:, old_col]
                new_W[:, new_col] = W[:, old_col]

            new_U[:, new_rank-1] = u_new
            new_V[:, new_rank-1] = v_new
            new_W[:, new_rank-1] = w_new

            return new_U, new_V, new_W, new_rank

        return None

    def _solve_merge(M, w_new):
        """
        Given target tensor M and a fixed combined W vector w_new, solve for
        u_new (n*m) and v_new (m*p) such that
            u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i] == M[i,j,k]  for all i,j,k.
        Uses exact rational Gaussian elimination.
        Returns (u_new, v_new) as float arrays, or (None, None) if impossible.
        """
        # Build rational w_new
        wr = [Fraction(int(round(x * 10**9)), 10**9) if abs(x) > 1e-12 else Fraction(0) for x in w_new]
        # Build rational M
        Mr = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Mr[(i, j, k)] = Fraction(int(round(M[i, j, k] * 10**9)), 10**9)

        # Variables: u[i,j] for all (i,j) [n*m], v[j,k] for all (j,k) [m*p]
        # Equations: for each (i,j,k) with wr[k*n+i] != 0:
        #   u[i,j] * v[j,k] = M[i,j,k] / wr[k*n+i]
        # This is bilinear. Strategy: fix a reference to linearize.
        #
        # Approach: pick a reference index to anchor. We'll use a systematic
        # linear solve by fixing v values from one reference i and solving u,
        # then checking consistency.
        #
        # Simpler robust approach: set up as a system where we guess the
        # structure. Since each equation is u[i,j]*v[j,k] = c[i,j,k], we can
        # parameterize: for a fixed j0, express v[j,k] in terms of u.
        #
        # Let's do: choose reference row i0 and column k0.
        # For a fixed j, the equations over k: u[i,j]*v[j,k] = c[i,j,k].
        # For fixed i, v[j,k] = c[i,j,k]/u[i,j].
        # Consistency: c[i,j,k]/u[i,j] must be independent of i for each (j,k).
        #
        # Linear method: take logs is not valid for exact. Instead:
        # For each (j,k), define v[j,k]. For each (i,j), define u[i,j].
        # The equation c[i,j,k] = u[i,j]*v[j,k]*w[k,i].
        #
        # We'll fix v[j,k] for k = k_ref (a reference k with w nonzero) using
        # one reference i, then solve.
        #
        # Concretely: pick i_ref. For each (j,k) with w[k*n+i_ref]!=0:
        #   v[j,k] = c[i_ref,j,k] / (u[i_ref,j] * w[k,i_ref])
        # This still has u[i_ref,j]. 
        #
        # Alternative clean linearization: For each j, the submatrix over (i,k):
        #   C_j[i,k] = c[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # Define A_j[i,k] = c[i,j,k]/w[k,i] (where w[k,i]!=0). Then
        #   A_j[i,k] = u[i,j] * v[j,k]   (rank-1 matrix in (i,k))
        # So for each j, A_j must be a rank-1 matrix (over the (i,k) grid where
        # w is nonzero), and u[i,j], v[j,k] are its factors.
        #
        # So: for each j, build A_j (n x p). Check rank 1 and extract factors.
        # Then cross-check that u[i,j] is consistent across j (it's per (i,j), so
        # no cross-j constraint on u) and v[j,k] consistent across i (per (j,k)).
        # Actually u[i,j] only appears in A_j, and v[j,k] only in A_j. So each j
        # is independent! Great.
        #
        # For each j: A_j[i,k] = c[i,j,k]/w[k,i] for w[k,i]!=0.
        #   Need A_j = u_col_j (n-vector) outer v_row_j (p-vector).
        #   Extract: if A_j has a nonzero entry at (i0,k0):
        #     u[i0,j] * v[j,k0] = A_j[i0,k0]
        #     For all i: u[i,j] = A_j[i,k0]/v[j,k0]
        #     For all k: v[j,k] = A_j[i0,k]/u[i0,j]
        #   Consistency check: A_j[i,k] == u[i,j]*v[j,k] for all i,k.

        u_new = np.zeros(n*m)
        v_new = np.zeros(m*p)

        for j in range(m):
            # Build A_j as rational matrix n x p; entries where w[k,i]==0 are
            # unconstrained (skip).
            A = [[None]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    if wr[k*n+i] != 0:
                        A[i][k] = Mr[(i, j, k)] / wr[k*n+i]
                    else:
                        A[i][k] = None  # unconstrained

            # Find a reference (i0,k0) with a defined value
            i0 = k0 = None
            for i in range(n):
                for k in range(p):
                    if A[i][k] is not None and A[i][k] != 0:
                        i0, k0 = i, k
                        break
                if i0 is not None:
                    break

            if i0 is None:
                # All defined entries are zero (or none defined). Then u[i,j]*v[j,k]=0
                # for all defined. We can set u[i,j]=0 for all i (then v free=0).
                # But must be consistent: if any defined A is nonzero -> fail.
                for i in range(n):
                    for k in range(p):
                        if A[i][k] is not None and A[i][k] != 0:
                            return None, None
                # Set u[i,j]=0, v[j,k]=0
                for i in range(n):
                    u_new[i*m+j] = 0.0
                for k in range(p):
                    v_new[j*p+k] = 0.0
                continue

            # Extract factors
            # Let s = A[i0,k0]. Choose u[i0,j] = 1, then v[j,k0] = s.
            # But we need consistency; choose u[i0,j] = A[i0,k0], v[j,k0]=1 to keep
            # integers. Actually set v[j,k0] = 1, u[i0,j] = A[i0,k0].
            u_i0 = A[i0][k0]
            # v[j,k0] = 1
            # For all i: u[i,j] = A[i,k0] / v[j,k0] = A[i,k0]
            # For all k: v[j,k] = A[i0,k] / u[i0,j] = A[i0,k] / u_i0
            u_vec = [None]*n
            v_vec = [None]*p
            u_vec[i0] = u_i0
            v_vec[k0] = Fraction(1)
            for i in range(n):
                if i == i0:
                    continue
                if A[i][k0] is None:
                    # u[i,j] unconstrained by k0; leave for consistency check
                    u_vec[i] = None
                else:
                    u_vec[i] = A[i][k0]  # since v[j,k0]=1
            for k in range(p):
                if k == k0:
                    continue
                if A[i0][k] is None:
                    v_vec[k] = None
                else:
                    if u_i0 == 0:
                        # can't divide; fail unless A[i0][k]==0
                        if A[i0][k] != 0:
                            return None, None
                        v_vec[k] = Fraction(0)
                    else:
                        v_vec[k] = A[i0][k] / u_i0

            # Fill in any None u_vec[i] using another k where both defined
            for i in range(n):
                if u_vec[i] is None:
                    for k in range(p):
                        if A[i][k] is not None and v_vec[k] is not None and v_vec[k] != 0:
                            u_vec[i] = A[i][k] / v_vec[k]
                            break
                    if u_vec[i] is None:
                        # No constraint found; set 0 (will be checked)
                        u_vec[i] = Fraction(0)
            for k in range(p):
                if v_vec[k] is None:
                    for i in range(n):
                        if A[i][k] is not None and u_vec[i] is not None and u_vec[i] != 0:
                            v_vec[k] = A[i][k] / u_vec[i]
                            break
                    if v_vec[k] is None:
                        v_vec[k] = Fraction(0)

            # Consistency check over all defined (i,k)
            ok = True
            for i in range(n):
                for k in range(p):
                    if A[i][k] is not None:
                        if u_vec[i] * v_vec[k] != A[i][k]:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                return None, None

            # Store
            for i in range(n):
                u_new[i*m+j] = float(u_vec[i])
            for k in range(p):
                v_new[j*p+k] = float(v_vec[k])

        # Final float verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                    if abs(prod - M[i, j, k]) > 1e-9:
                        return None, None
        return u_new, v_new

    def _make_rank43():
        """
        Construct a low-rank decomposition by iteratively merging pairs of
        schoolbook terms using exact rational solving.
        """
        U, V, W, rank = _make_schoolbook()

        # Repeatedly try to merge any pair of columns until no merge works.
        max_rounds = 20
        for _ in range(max_rounds):
            if _remaining() < 5.0:
                break
            found = False
            for r1, r2 in combinations(range(rank), 2):
                if _remaining() < 5.0:
                    break
                result = _try_merge_pair(U, V, W, rank, r1, r2)
                if result is not None:
                    U, V, W, rank = result
                    found = True
                    break
            if not found:
                break

        return U, V, W, rank

    # Try to construct a low-rank decomposition
    U, V, W, rank = _make_rank43()

    # Verify the result
    if _verify(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        U, V, W, rank = _make_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
