"""Rank search for 2x4x4 matrix multiplication via structured numerical tensor-rank fitting.

Strategy (different from block-wise Strassen composition):
  - Build the full multiplication tensor T[i,j,k] = 1 (n=2,m=4,p=4).
  - For candidate rank R, use a randomized/least-squares rank-R fitting procedure:
      fix one factor matrix, solve for the other two by linear least squares,
      then refine by alternating updates.
  - Snap to half-integers and verify an EXACT decomposition.
  - Return the lowest verified rank; fall back to the known rank-28 Strassen tiling.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _verify(U, V, W, n, m, p, R, atol=1e-6):
    """Return True if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Build tensor: for each r, outer product of reshaped U_r, V_r, W_r
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=atol))


def _snap_halfint(A):
    """Round to nearest half-integer."""
    return np.round(2.0 * A) / 2.0


def _try_rank(R, n, m, p, rng, max_iter=300, restarts=4):
    """Attempt to find a rank-R exact half-integer decomposition.
    Returns (U,V,W) as lists of lists on success, else None."""
    Nm = n * m
    Mp = m * p
    Pn = p * n

    # Precompute index maps
    # U index: (i*m + j)
    # V index: (j*p + k)
    # W index: (k*n + i)

    for _restart in range(restarts):
        U = rng.standard_normal((Nm, R))
        # Fix U, solve for V and W via least squares.
        # We have linear constraints: for each (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
        # This is bilinear in V and W. Use alternating: fix W, solve V linear; fix V, solve W linear.
        W = rng.standard_normal((Pn, R))

        for _it in range(max_iter):
            # --- Solve V given U, W ---
            # For each (j,k): sum_r V[j*p+k,r] * (sum_i U[i*m+j,r] W[k*n+i,r]) = 1
            # Define A_V[j,k,r] = sum_i U[i*m+j,r] W[k*n+i,r]
            # Then for each (j,k): A_V[j,k,:] . V[j*p+k,:] = 1
            # This is R equations? No: it's one equation (scalar) per (j,k) with R unknowns -> underdetermined.
            # Actually each (j,k) gives ONE scalar equation. So we have (m*p) equations for (m*p*R) unknowns.
            # Under-determined -> pick minimum-norm solution via least squares (Moore-Penrose) per row.
            # Better: treat as linear system. Let's build the full linear system over all (i,j,k).
            #
            # Full linear system in V (with U,W fixed):
            # For each (i,j,k): sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1
            # This is linear in V. Number of unknowns = m*p*R. Number of equations = n*m*p.
            # Build matrix M_V of shape (n*m*p, m*p*R) and rhs.
            #
            # Similarly for W.
            pass

        # The alternating least-squares on the full linear system:
        # Build V-system matrix once per iteration.
        # Index for equation (i,j,k) -> eq = (i*m + j)*p + k
        # Unknown for V -> (j*p + k)*R + r
        eq_count = n * m * p
        unk_V = m * p * R
        unk_W = p * n * R

        for _it in range(max_iter):
            # Solve for V (U,W fixed)
            M_V = np.zeros((eq_count, unk_V))
            rhs = np.ones(eq_count)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = (i * m + j) * p + k
                        for r in range(R):
                            M_V[eq, (j * p + k) * R + r] = U[i * m + j, r] * W[k * n + i, r]
            V_new = np.linalg.lstsq(M_V, rhs, rcond=None)[0].reshape(m * p, R)
            # Solve for W (U,V fixed)
            M_W = np.zeros((eq_count, unk_W))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = (i * m + j) * p + k
                        for r in range(R):
                            M_W[eq, (k * n + i) * R + r] = U[i * m + j, r] * V[j * p + k, r]
            W_new = np.linalg.lstsq(M_W, rhs, rcond=None)[0].reshape(p * n, R)

            # Update
            V = V_new
            W = W_new

            # Check residual
            if _it % 5 == 0:
                T = np.einsum('ijr,jkr,kir->ijk',
                              U.reshape(n, m, R), V.reshape(m, p, R), W.reshape(p, n, R))
                if np.max(np.abs(T - 1.0)) < 1e-7:
                    # Snap to half-integers and verify
                    Us = _snap_halfint(U)
                    Vs = _snap_halfint(V)
                    Ws = _snap_halfint(W)
                    if _verify(Us, Vs, Ws, n, m, p, R):
                        return (Us.tolist(), Vs.tolist(), Ws.tolist())
                    # also try without snap (maybe already integer/half-integer)
                    if _verify(U, V, W, n, m, p, R):
                        return (U.tolist(), V.tolist(), W.tolist())
        # end iter
        # end restart: re-randomize U
        U = rng.standard_normal((Nm, R))

    return None


def _strassen_block():
    U_loc = np.zeros((2, 2, 7))
    V_loc = np.zeros((2, 2, 7))
    W_loc = np.zeros((2, 2, 7))
    # r=0 (M1)
    U_loc[0, 0, 0] = 1; U_loc[1, 1, 0] = 1
    V_loc[0, 0, 0] = 1; V_loc[1, 1, 0] = 1
    W_loc[0, 0, 0] = 1; W_loc[1, 0, 0] = 1; W_loc[0, 1, 0] = 1; W_loc[1, 1, 0] = -1
    # r=1 (M2)
    U_loc[1, 0, 1] = 1; U_loc[1, 1, 1] = 1
    V_loc[0, 0, 1] = 1
    W_loc[0, 1, 1] = 1; W_loc[1, 1, 1] = -1
    # r=2 (M3)
    U_loc[0, 0, 2] = 1
    V_loc[1, 0, 2] = 1; V_loc[1, 1, 2] = -1
    W_loc[1, 0, 2] = 1; W_loc[1, 1, 2] = 1
    # r=3 (M4)
    U_loc[1, 1, 3] = 1
    V_loc[0, 1, 3] = 1; V_loc[0, 0, 3] = -1
    W_loc[0, 0, 3] = 1; W_loc[0, 1, 3] = 1
    # r=4 (M5)
    U_loc[0, 0, 4] = 1; U_loc[0, 1, 4] = 1
    V_loc[1, 1, 4] = 1
    W_loc[0, 0, 4] = -1; W_loc[1, 0, 4] = 1
    # r=5 (M6)
    U_loc[1, 0, 5] = 1; U_loc[0, 0, 5] = -1
    V_loc[0, 0, 5] = 1; V_loc[1, 0, 5] = 1
    W_loc[1, 1, 5] = 1
    # r=6 (M7)
    U_loc[0, 1, 6] = 1; U_loc[1, 1, 6] = -1
    V_loc[0, 1, 6] = 1; V_loc[1, 1, 6] = 1
    W_loc[0, 0, 6] = 1
    return U_loc, V_loc, W_loc


def _rank28_strassen(n, m, p):
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    U_loc, V_loc, W_loc = _strassen_block()
    blocks = [(0, 0, 0), (2, 0, 0), (0, 2, 2), (2, 2, 2)]
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        r_off = b_idx * 7
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    U[i * m + (a_off + j), r_off + r] = U_loc[i, j, r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V[(b_off + j) * p + (c_off + k), r_off + r] = V_loc[j, k, r]
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W[(c_off + k) * n + i, r_off + r] = W_loc[k, i, r]
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rng = np.random.default_rng(12345)

    # Try to find a rank below 28 via numerical fitting.
    # Start at the most promising target (27) and go up if it fails quickly.
    for R in [27, 26, 25]:
        if time.time() > deadline - 40:
            break
        res = _try_rank(R, n, m, p, rng, max_iter=120, restarts=2)
        if res is not None:
            return {"U": res[0], "V": res[1], "W": res[2]}

    # Fallback: known rank-28 Strassen tiling (guaranteed exact).
    U, V, W = _rank28_strassen(n, m, p)
    return {"U": U, "V": V, "W": W}
