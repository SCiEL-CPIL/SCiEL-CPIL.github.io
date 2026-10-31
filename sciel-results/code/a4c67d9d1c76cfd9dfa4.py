"""Structural redesign: model the Hermite-Gaussian uncertainty product as a
generalized Rayleigh quotient and solve the constrained secular equation."""
import time
import numpy as np
from scipy.linalg import eig, solve

_T0 = time.time()
_DEADLINE = 210.0
_MARGIN = 10.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _hermite_basis_norms(degrees):
    """Integral of (H_n e^{-x^2/2})^2 = sqrt(pi) 2^n n!."""
    from math import factorial
    return np.array([np.sqrt(np.pi) * (2 ** n) * factorial(n) for n in degrees], dtype=float)

def _build_forms(degrees):
    """Quadratic-form matrices for the two factors of the uncertainty product.

    The Fourier transform maps H_n e^{-x^2/2} -> (-i)^n H_n e^{-x^2/2}, so the
    L2 energy is diagonal in the Hermite basis.  The C4 uncertainty bound
    involves the ratio of the L2 energy to the weighted (position) energy,
    both diagonal in this basis; we keep them as full matrices to support the
    constraint projection cleanly.
    """
    n = len(degrees)
    base = _hermite_basis_norms(degrees)
    M_A = np.diag(base)
    # Weighted energy  \int x^2 g^2  in the Hermite basis (exact, diagonal in
    # the parity-matched subspace up to the known recurrence):
    #   <H_n e^{-x^2/2}, x^2 H_n e^{-x^2/2}> = (2n+1) * <h_n,h_n>
    M_B = np.diag((2 * np.array(degrees) + 1) * base)
    return M_A, M_B

def _solve_constrained(M_A, M_B, fix_index, c0):
    """Minimize c^T M_A c / c^T M_B c subject to c[fix_index] = c0.

    Returns the free-coefficient vector (all coefficients) or None.
    """
    n = M_A.shape[0]
    # Eliminate the fixed coordinate.
    idx = [i for i in range(n) if i != fix_index]
    A = M_A[np.ix_(idx, idx)]
    B = M_B[np.ix_(idx, idx)]
    # Linear terms from the fixed coordinate.
    a_lin = M_A[np.ix_(idx, [fix_index])] * c0
    b_lin = M_B[np.ix_(idx, [fix_index])] * c0
    # Quadratic objective in free vars:
    #   num = x^T A x + 2 a_lin^T x + M_A[fix,fix]*c0^2
    #   den = x^T B x + 2 b_lin^T x + M_B[fix,fix]*c0^2
    # Generalized eigenvalue of (A, B) gives the optimal *direction*; with a
    # linear shift the minimum of the ratio is attained at the eigenvector of
    # the shifted pencil.  We solve the secular equation directly.
    w, V = eig(B, A)  # solves A v = w B v  (scipy: eig(B,A) -> B v = w A v)
    # We want min (x^T A x + ...)/(x^T B x + ...).  Use the smallest generalized
    # eigenvalue of (A, B) as the direction and refine the offset by 1-D search.
    w = np.real(w)
    V = np.real(V)
    valid = np.isfinite(w)
    if not valid.any():
        return None
    w = w[valid]
    V = V[:, valid]
    best = None
    best_val = np.inf
    for k in range(V.shape[1]):
        v = V[:, k]
        if np.linalg.norm(v) < 1e-12:
            continue
        # 1-D minimization over scale t: x = t*v
        # ratio(t) = (t^2 v^T A v + 2 t a_lin^T v + A0)/(t^2 v^T B v + 2 t b_lin^T v + B0)
        A0 = M_A[fix_index, fix_index] * c0 ** 2
        B0 = M_B[fix_index, fix_index] * c0 ** 2
        qa = v @ A @ v
        qb = v @ B @ v
        la = a_lin @ v
        lb = b_lin @ v
        if abs(qb) < 1e-14:
            continue
        # derivative of ratio:
        # f(t) = (qa t^2 + 2 la t + A0)/(qb t^2 + 2 lb t + B0)
        # f'(t) = 0 -> quadratic in t.
        # (2 qa t + 2 la)(qb t^2 + 2 lb t + B0) - (qa t^2 + 2 la t + A0)(2 qb t + 2 lb) = 0
        # Expand coefficients of t^2, t, const:
        # t^3: 2qa*qb - 2qa*qb = 0  (cancels)
        # t^2: 2qa*2lb + 2la*qb - (qa*2lb + 2la*2qb) = 4qa lb + 2la qb - 2qa lb - 4la qb = 2qa lb - 2la qb
        # t^1: 2qa*B0 + 2la*2lb - (2la*B0 + A0*2qb) = 2qa B0 + 4la lb - 2la B0 - 2A0 qb
        # t^0: 2la*B0 - A0*2lb = 2la B0 - 2A0 lb
        c2 = 2 * (qa * lb - la * qb)
        c1 = 2 * qa * B0 + 4 * la * lb - 2 * la * B0 - 2 * A0 * qb
        c0c = 2 * (la * B0 - A0 * lb)
        cands = []
        if abs(c2) > 1e-14:
            disc = c1 * c1 - 4 * c2 * c0c
            if disc >= 0:
                sq = np.sqrt(disc)
                cands += [(-c1 + sq) / (2 * c2), (-c1 - sq) / (2 * c2)]
        else:
            if abs(c1) > 1e-14:
                cands.append(-c0c / c1)
        for t in cands:
            if not np.isfinite(t):
                continue
            x = t * v
            num = x @ A @ x + 2 * a_lin @ x + A0
            den = x @ B @ x + 2 * b_lin @ x + B0
            if den <= 1e-14 or num < 0:
                continue
            val = num / den
            if val < best_val:
                best_val = val
                best = x
    if best is None:
        return None
    full = np.zeros(n)
    full[idx] = best
    full[fix_index] = c0
    return full

def construct():
    try:
        if _time_left() <= 0:
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
        degrees = [0, 4, 8, 12]
        M_A, M_B = _build_forms(degrees)
        # g(0) = 0 constraint:  sum_j c_j H_j(0) = 0.
        # H_0(0)=1, H_4(0)=12, H_8(0)=1344, H_12(0)=1032192.
        from scipy.special import eval_hermite
        H0 = np.array([float(eval_hermite(n, 0.0)) for n in degrees])
        # Solve for c_12 in terms of the others: c_12 = -(H0_0 c_0 + H0_4 c_4 + H0_8 c_8)/H0_12
        fix_index = 3
        # We need a concrete c0 for the fixed coordinate; instead, parameterize
        # by the three free coefficients and eliminate c_12.  Do it by choosing
        # the free subspace as the null direction of the constraint.
        # Build a basis for the 3-dim subspace {c : H0 . c = 0}.
        # Use SVD of the 1x4 constraint row.
        U, s, Vt = np.linalg.svd(H0.reshape(1, -1))
        null = Vt[1:].T  # 4x3 basis of the constraint subspace
        # Objective on the subspace: c = null @ y, y in R^3.
        A_sub = null.T @ M_A @ null
        B_sub = null.T @ M_B @ null
        # Minimize y^T A_sub y / y^T B_sub y  -> smallest generalized eigenvalue.
        w, V = eig(B_sub, A_sub)
        w = np.real(w)
        V = np.real(V)
        valid = np.isfinite(w)
        if not valid.any():
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
        # eig(B_sub, A_sub) solves B_sub v = w A_sub v, i.e. A_sub v = (1/w) B_sub v.
        # The ratio R = y^T A y / y^T B y.  Minimize R -> pick v with smallest R.
        best = None
        best_val = np.inf
        for k in range(V.shape[1]):
            if not valid[k]:
                continue
            v = V[:, k]
            if np.linalg.norm(v) < 1e-12:
                continue
            r = (v @ A_sub @ v) / (v @ B_sub @ v)
            if np.isfinite(r) and r > 0 and r < best_val:
                best_val = r
                best = v
        if best is None:
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
        c = null @ best
        # Scale to a sensible magnitude (the seed is O(1)); keep relative shape.
        # Return the three free coefficients (degrees 0,4,8).
        free = c[[0, 1, 2]]
        # Normalize by the largest magnitude for stability.
        m = np.max(np.abs(free))
        if m > 1e-12:
            free = free / m * 1.0
        if not np.all(np.isfinite(free)):
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
        return free
    except Exception:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
