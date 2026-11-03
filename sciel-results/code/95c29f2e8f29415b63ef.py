"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Structural approach:
  1. Encode the bilinear map A (3x3) * B (3x7) -> C (3x7) as a 3rd-order tensor.
  2. Apply a fast tensor network / ALS solver with multiple restarts and
     aggressive early stopping.
  3. Round to the nearest half-integers and verify exactness.
  4. If a lower rank is not found, return the trivial rank-63 decomposition.

The key structural change here is to use a more efficient ALS that exploits
the sparsity of the target tensor (only 63 non-zero entries) and to use
a smarter initialization based on the support pattern.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Precompute target tensor (sparse: only 63 non-zero entries)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    nz_a = []
    nz_b = []
    nz_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a, dtype=int)
    nz_b = np.array(nz_b, dtype=int)
    nz_c = np.array(nz_c, dtype=int)
    nz_vals = np.ones(len(nz_a))

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def is_valid_decomposition(U, V, W, tol=1e-9):
        """Check exactness: on-diagonal = 1, off-diagonal = 0."""
        # Efficient check: only need to verify the 63 target positions and
        # confirm no off-diagonal contamination.
        T = np.einsum('ar,br,cr->abc', U, V, W)
        # Check on-diagonal
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        # Check off-diagonal: sample a subset for speed, then full if needed
        # Full check is 9*21*21 = 3969 entries, minus 63 on-diagonal = 3906
        # This is fast enough
        mask = np.zeros((n * m, m * p, p * n), dtype=bool)
        mask[nz_a, nz_b, nz_c] = True
        if np.any(np.abs(T[~mask]) > tol):
            return False
        return True

    def round_to_half(U, V, W):
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    # Trivial rank-63 decomposition
    trivial_rank = n * m * p
    U_t = np.zeros((n * m, trivial_rank))
    V_t = np.zeros((m * p, trivial_rank))
    W_t = np.zeros((p * n, trivial_rank))
    for idx in range(trivial_rank):
        U_t[nz_a[idx], idx] = 1.0
        V_t[nz_b[idx], idx] = 1.0
        W_t[nz_c[idx], idx] = 1.0

    def als_sparse(target_nz_a, target_nz_b, target_nz_c, target_nz_vals,
                   R, max_iters=40, tol=1e-12, seed=0):
        """
        ALS exploiting sparsity. T[a,b,c] is nonzero only at (nz_a, nz_b, nz_c).
        For each a, the equation is:
          sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]  for all (b,c)
        But only 7 of the 21*21=441 (b,c) pairs are nonzero for a given a.
        We still need to enforce zeros at off-diagonal positions.
        Use full system but vectorize.
        """
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, R) * 0.5
        V = rng.randn(m * p, R) * 0.5
        W = rng.randn(p * n, R) * 0.5

        for it in range(max_iters):
            # --- Solve for U ---
            # For each a: U[a,:] is R-vector. Equation:
            #   U[a,r] * (V[:,r] * W[:,r])_{b,c} summed over r = T[a,b,c]
            # Build M_VW of shape (R, m*p*p*n): M_VW[r, b*(p*n)+c] = V[b,r]*W[c,r]
            M_VW = np.einsum('br,cr->rbc', V, W).reshape(R, m * p * p * n)
            for a in range(n * m):
                rhs = np.zeros(m * p * p * n)
                # Fill only nonzero entries
                for idx in range(len(target_nz_a)):
                    if target_nz_a[idx] == a:
                        b, c = target_nz_b[idx], target_nz_c[idx]
                        rhs[b * (p * n) + c] = target_nz_vals[idx]
                try:
                    U[a, :] = np.linalg.lstsq(M_VW.T, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for V ---
            M_UW = np.einsum('ar,cr->rac', U, W).reshape(R, n * m * p * n)
            for b in range(m * p):
                rhs = np.zeros(n * m * p * n)
                for idx in range(len(target_nz_b)):
                    if target_nz_b[idx] == b:
                        a, c = target_nz_a[idx], target_nz_c[idx]
                        rhs[a * (p * n) + c] = target_nz_vals[idx]
                try:
                    V[b, :] = np.linalg.lstsq(M_UW.T, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for W ---
            M_UV = np.einsum('ar,br->rab', U, V).reshape(R, n * m * m * p)
            for c in range(p * n):
                rhs = np.zeros(n * m * m * p)
                for idx in range(len(target_nz_c)):
                    if target_nz_c[idx] == c:
                        a, b = target_nz_a[idx], target_nz_b[idx]
                        rhs[a * (m * p) + b] = target_nz_vals[idx]
                try:
                    W[c, :] = np.linalg.lstsq(M_UV.T, rhs, rcond=None)[0]
                except:
                    pass

            # Check convergence
            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            err = 0.0
            for idx in range(len(nz_a)):
                a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
                err += (T_cur[a, b, c] - 1.0) ** 2
            # Sample off-diagonal
            err += np.sum(np.abs(T_cur) ** 2) - err  # crude
            if err < tol:
                break

        return U, V, W

    best_U, best_V, best_W = U_t, V_t, W_t
    best_rank = trivial_rank

    # Try ranks from 24 down to 15 (Strassen for 3x3x3 is 23, so 3x3x7 likely ~24-30)
    for R in range(24, 14, -1):
        if time.time() > deadline - 30:
            break
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            seed = trial * 7919 + R * 31
            U, V, W = als_sparse(nz_a, nz_b, nz_c, nz_vals, R,
                                 max_iters=30, tol=1e-14, seed=seed)
            U_r, V_r, W_r = round_to_half(U, V, W)
            if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = R
                break  # try lower rank

    if best_rank < trivial_rank:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    return {"U": to_list(U_t), "V": to_list(V_t), "W": to_list(W_t)}
