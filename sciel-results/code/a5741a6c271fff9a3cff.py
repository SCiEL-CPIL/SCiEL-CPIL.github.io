# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE_SECONDS = 615.0  # safety margin under the 630 s kill


def _build_harmonic_matrix():
    """
    Build the real orthonormal degree-10 spherical harmonic evaluation matrix.
    Returns H of shape (Ntheta, Nphi, 21) where column j corresponds to m = j-10.
    Uses 70-point Gauss-Legendre in cos(theta) and 130 equally spaced phi.
    """
    from scipy.special import eval_legendre

    k = 10
    Ntheta = 70
    Nphi = 130

    x, w = np.polynomial.legendre.leggauss(Ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
        for j in range(lo, hi + 1):
            den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (1.0 / den))

    for m in range(-k, k + 1):
        j = m + k
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    weights = np.outer(w, np.full(Nphi, dphi))
    return H, weights


def _ratio_and_grad(c, H, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    """
    f = np.tensordot(c, H, axes=([0], [2]))  # (Ntheta, Nphi)
    f2 = f * f
    f4 = f2 * f2
    A = np.sum(f4 * weights)  # ||f||_4^4
    B = np.sum(f2 * weights)  # ||f||_2^2

    if A <= 0 or B <= 0:
        return np.inf, np.zeros_like(c)

    R = A ** 0.25 / np.sqrt(B)

    A_neg34 = A ** (-0.75)
    B_neg12 = B ** (-0.5)
    A_pos14 = A ** 0.25
    B_neg32 = B ** (-1.5)

    coeff = A_neg34 * f2 * B_neg12 - A_pos14 * B_neg32
    fw_coeff = f * weights * coeff
    grad = np.tensordot(fw_coeff, H, axes=([0, 1], [0, 1]))

    return R, grad


def _var_f2_obj_grad(c, H_flat, w_flat, n):
    """
    Smooth surrogate: minimize Var(f^2) = E[f^4] - (E[f^2])^2 subject to E[f^2] = 1.
    With constraint E[f^2]=1, this equals E[f^4] - 1, so minimize E[f^4].
    But to keep it smooth and unconstrained, we minimize:
        J(c) = E[f^4] / (E[f^2])^2 - 1  (kurtosis excess, scale-invariant)
    Actually simpler: fix ||c||_2 = 1 (Euclidean) and minimize E[f^4].
    Since harmonics are orthonormal under the measure, if c is unit Euclidean
    then E[f^2] = c^T (H^T W H) c = c^T G c where G is the Gram matrix.
    For orthonormal basis, G = I, so E[f^2] = ||c||^2 = 1.
    Thus minimizing E[f^4] subject to ||c||=1 is equivalent.
    
    J(c) = sum f^4 w / sum w  (since E[f^2]=1 when ||c||=1 for orthonormal basis)
    Actually the quadrature weights sum to 4*pi, so E[f^2] = (1/(4pi)) sum f^2 w.
    If ||c||_2 = 1 and basis is orthonormal, then sum f^2 w = 4*pi, so E[f^2] = 1.
    Then E[f^4] = (1/(4pi)) sum f^4 w.
    
    dJ/dc = (1/(4pi)) * 4 f^3 * H^T w = (4/(4pi)) H^T (f^3 w) = (1/pi) H^T (f^3 w)
    But we need to project onto the tangent space of the sphere ||c||=1.
    """
    f = H_flat @ c
    f2 = f * f
    f4 = f2 * f2
    total_w = np.sum(w_flat)  # should be 4*pi
    E_f4 = np.dot(f4, w_flat) / total_w
    E_f2 = np.dot(f2, w_flat) / total_w
    
    # Objective: minimize E[f^4] (with ||c||=1 constraint handled by optimizer)
    # Gradient of E[f^4] w.r.t. c:
    # dE_f4/dc_j = (1/total_w) * sum_i 4*f_i^3 * w_i * H_ij
    grad_raw = (4.0 / total_w) * (H_flat.T @ (f * f2 * w_flat))
    
    # Project gradient onto tangent space of unit sphere
    grad_proj = grad_raw - np.dot(grad_raw, c) * c
    
    return E_f4, grad_proj


def _make_initializations(k, rng, H, weights):
    """Generate diverse starting points."""
    n = 2 * k + 1
    inits = []

    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())

    # Pairs
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v.copy())

    # All ones
    inits.append(np.ones(n))

    # Even m only
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    inits.append(v.copy())

    # Random
    for _ in range(30):
        v = rng.normal(size=n)
        v /= (np.linalg.norm(v) + 1e-15)
        inits.append(v)

    # Sparse
    for _ in range(15):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 6), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= (np.linalg.norm(v) + 1e-15)
        inits.append(v)

    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    Returns:
        coeffs: np.ndarray of shape (21,) with c_m ordered m = -10..10.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    rng = np.random.default_rng(42)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    # Precompute H as contiguous array for speed
    H_flat = H.reshape(-1, n)  # (Ntheta*Nphi, 21)
    w_flat = weights.reshape(-1)  # (Ntheta*Nphi,)

    def _ratio_grad_fast(c):
        f = H_flat @ c  # (N,)
        f2 = f * f
        f4 = f2 * f2
        A = np.dot(f4, w_flat)
        B = np.dot(f2, w_flat)
        if A <= 0 or B <= 0:
            return np.inf, np.zeros(n)
        R = A ** 0.25 / np.sqrt(B)
        A_neg34 = A ** (-0.75)
        B_neg12 = B ** (-0.5)
        A_pos14 = A ** 0.25
        B_neg32 = B ** (-1.5)
        coeff = A_neg34 * f2 * B_neg12 - A_pos14 * B_neg32
        fw_coeff = f * w_flat * coeff
        grad = H_flat.T @ fw_coeff
        return R, grad

    inits = _make_initializations(k, rng, H, weights)

    best_c = np.ones(n, dtype=np.float64)
    best_c /= np.linalg.norm(best_c)
    best_R = _ratio_grad_fast(best_c)[0]

    def _lbfgsb_on_sphere(c0):
        """Use L-BFGS-B to minimize E[f^4] on the unit sphere."""
        if remaining() < 15.0:
            return c0, np.inf

        def obj_and_grad(x):
            # Project back to sphere
            xn = x / (np.linalg.norm(x) + 1e-15)
            E_f4, grad_proj = _var_f2_obj_grad(xn, H_flat, w_flat, n)
            return E_f4, grad_proj

        try:
            res = minimize(
                obj_and_grad,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            c_opt = res.x / (np.linalg.norm(res.x) + 1e-15)
            R_opt = _ratio_grad_fast(c_opt)[0]
            return c_opt, R_opt
        except Exception:
            return c0, np.inf

    # L-BFGS-B from multiple initializations
    for init_idx, c0 in enumerate(inits):
        if remaining() < 15.0:
            break

        c0_norm = c0 / (np.linalg.norm(c0) + 1e-15)
        c_opt, R_opt = _lbfgsb_on_sphere(c0_norm)

        if np.isfinite(R_opt) and R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()

    # Final polishing with L-BFGS-B from best
    if remaining() > 30.0:
        c_opt, R_opt = _lbfgsb_on_sphere(best_c)
        if np.isfinite(R_opt) and R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()

    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c


# EVOLVE-BLOCK-END
