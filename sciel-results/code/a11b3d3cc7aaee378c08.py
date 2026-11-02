"""Matrix multiplication 4x4x6: rank-minimal exact tensor decomposition.

Strategy:
  1. Start from the known rank-20 block-decomposition (4 copies of the
     rank-5 2x2x3 tensor).
  2. Use a randomized / structured search to try to reduce the rank to 19
     or lower by finding a linear dependency among the 20 rank-1 terms
     that is consistent across U, V, W simultaneously.
  3. If a lower-rank decomposition is found, verify it exactly.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def _verify(U, V, W, n, m, p):
    """Check that the tensor decomposition is exact (all entries = 1)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    T = np.einsum('ir,kr,jr->ijk', Ua, Wa, Va)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if abs(T[i, k, j] - 1.0) > 1e-9:
                    return False
    return True


def _build_rank20():
    """Build the rank-20 decomposition from 4 copies of the 2x2x3 rank-5 tensor."""
    # Known rank-5 decomposition of 2x2x3 multiplication tensor.
    # Local convention: Ua[a*2+b], Va[b*3+c], Wc[c*2+a]
    Ua = [[1, 1, 1, 0, 1],
          [0, 1, 0, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 0, 1, 1, 1]]
    Va = [[1, 1, 0, 1, 0],
          [0, 1, 0, 0, 1],
          [1, 0, 1, 0, 1],
          [1, 0, 0, 1, 0],
          [1, 1, 1, 0, 1]]
    Wc = [[1, 1, 1, 1, 1],
          [1, 0, 0, 1, 1],
          [1, 1, 0, 0, 1]]

    U20 = [[0] * 20 for _ in range(16)]
    V20 = [[0] * 20 for _ in range(24)]
    W20 = [[0] * 20 for _ in range(24)]

    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * 5
            for a in range(2):
                for b in range(2):
                    for t in range(5):
                        U20[(bi * 2 + a) * 4 + (bj * 2 + b)][base + t] = Ua[a * 2 + b][t]
            for b in range(2):
                for c in range(3):
                    for t in range(5):
                        V20[(bj * 2 + b) * 6 + c][base + t] = Va[b * 3 + c][t]
            for c in range(3):
                for a in range(2):
                    for t in range(5):
                        W20[c * 4 + (bi * 2 + a)][base + t] = Wc[c * 2 + a][t]

    return U20, V20, W20


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding a column that can be eliminated.
    Uses a structured search over possible linear dependencies.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]

    # Try each column as a candidate for elimination
    for elim_col in range(r):
        if time.time() > deadline - 10.0:
            break

        # Build matrix of other columns
        other_cols = [c for c in range(r) if c != elim_col]
        U_other = Ua[:, other_cols]  # 16 x (r-1)
        V_other = Va[:, other_cols]  # 24 x (r-1)
        W_other = Wa[:, other_cols]  # 24 x (r-1)

        # Try to find alpha such that:
        #   U_other @ alpha = Ua[:, elim_col]
        #   V_other @ alpha = Va[:, elim_col]
        #   W_other @ alpha = Wa[:, elim_col]
        # This is an overdetermined system. We solve each separately and check consistency.

        try:
            alpha_U, res_U, rank_U, _ = np.linalg.lstsq(U_other, Ua[:, elim_col], rcond=None)
            alpha_V, res_V, rank_V, _ = np.linalg.lstsq(V_other, Va[:, elim_col], rcond=None)
            alpha_W, res_W, rank_W, _ = np.linalg.lstsq(W_other, Wa[:, elim_col], rcond=None)
        except Exception:
            continue

        # Check if the residuals are small (column is approximately in the span)
        if res_U[0] > 1e-6 or res_V[0] > 1e-6 or res_W[0] > 1e-6:
            continue

        # Check if the same alpha works for all three
        if (np.allclose(alpha_U, alpha_V, atol=1e-6) and
                np.allclose(alpha_U, alpha_W, atol=1e-6)):
            # Column elim_col is in the span of other columns with the same coefficients
            # Create rank-(r-1) matrices
            U_new = Ua[:, other_cols].copy()
            V_new = Va[:, other_cols].copy()
            W_new = Wa[:, other_cols].copy()

            # Round to half-integers
            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2

            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                return U_new.tolist(), V_new.tolist(), W_new.tolist(), r - 1

    return None


def _randomized_search(U, V, W, n, m, p, deadline, max_iters=5000):
    """
    Randomized search: perturb the rank-20 decomposition slightly and try
    to find a lower-rank approximation that rounds to an exact half-integer solution.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]

    target = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, k, j] = 1.0

    best_rank = r
    best_U = Ua.copy()
    best_V = Va.copy()
    best_W = Wa.copy()

    for it in range(max_iters):
        if time.time() > deadline - 15.0:
            break

        # Try to reduce rank by 1: randomly select a column to eliminate
        # and try to adjust the remaining columns to compensate.
        elim = np.random.randint(r)
        other_cols = [c for c in range(r) if c != elim]

        U_other = Ua[:, other_cols].copy()
        V_other = Va[:, other_cols].copy()
        W_other = Wa[:, other_cols].copy()

        # Add small random perturbation
        noise_scale = 0.1
        U_other += np.random.randn(*U_other.shape) * noise_scale
        V_other += np.random.randn(*V_other.shape) * noise_scale
        W_other += np.random.randn(*W_other.shape) * noise_scale

        # Try to solve for the eliminated column's contribution
        # We want: sum_{c in other} alpha_c * (U_c, V_c, W_c) = (U_elim, V_elim, W_elim)
        # This is a linear system in alpha.
        try:
            alpha_U, _, _, _ = np.linalg.lstsq(U_other, Ua[:, elim], rcond=None)
            alpha_V, _, _, _ = np.linalg.lstsq(V_other, Va[:, elim], rcond=None)
            alpha_W, _, _, _ = np.linalg.lstsq(W_other, Wa[:, elim], rcond=None)
        except Exception:
            continue

        # Check consistency
        if (np.allclose(alpha_U, alpha_V, atol=1e-3) and
                np.allclose(alpha_U, alpha_W, atol=1e-3)):
            # Found a consistent dependency!
            U_new = Ua[:, other_cols].copy()
            V_new = Va[:, other_cols].copy()
            W_new = Wa[:, other_cols].copy()

            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2

            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                return U_new.tolist(), V_new.tolist(), W_new.tolist(), r - 1

    return None


def construct():
    t0 = time.time()
    n, m, p = 4, 4, 6
    deadline = t0 + 130.0  # Safety margin

    # Start with rank-20
    U20, V20, W20 = _build_rank20()
    best = {"U": U20, "V": V20, "W": W20}
    best_rank = 20

    if not _verify(U20, V20, W20, n, m, p):
        # Fallback to schoolbook
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best

    # Try structured rank reduction
    if time.time() < deadline - 20.0:
        result = _try_reduce_rank(U20, V20, W20, n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            best = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = new_rank

    # Try randomized search
    if time.time() < deadline - 10.0 and best_rank > 19:
        result = _randomized_search(best["U"], best["V"], best["W"], n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            best = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = new_rank

    return best
