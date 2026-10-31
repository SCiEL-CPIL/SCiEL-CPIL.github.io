"""Analytic Hermite-Gaussian uncertainty bound: exact generalized eigenvalue + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _compute_moments(a0, a4, a8):
    """Compute m0, m2 (position) and n0, n2 (frequency) moments exactly."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k where P(x) = sum p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12

    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n - 1] * (2 * n - 1) / (2.0 * n) * 2.0  # I[n] = I[n-1] * (2n-1)/2
        # Actually: I[n] = (2n-1)!! * sp / 2^n, I[n]/I[n-1] = (2n-1)/2
        # Let me recompute properly
    # Recompute I properly
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m_k = sum_{i,j} p_i p_j I[k + i + j] for k=0,1 (m0, m2)
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            m0 += pp * I[i + j]
            m2 += pp * I[1 + i + j]

    # Fourier: Q(xi) coefficients q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k + i + j]
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            n0 += 2.0 * np.pi * qq * I[i + j]
            n2 += 2.0 * np.pi * qq * I[1 + i + j]

    return m0, m2, n0, n2


def _bound_and_grad(a0, a4, a8):
    """Compute the uncertainty bound and its gradient."""
    m0, m2, n0, n2 = _compute_moments(a0, a4, a8)

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    # Numerical gradient (robust, avoids algebraic errors)
    eps = 1e-8
    grad = np.zeros(3)
    for c in range(3):
        params_plus = np.array([a0, a4, a8], dtype=float)
        params_minus = np.array([a0, a4, a8], dtype=float)
        params_plus[c] += eps
        params_minus[c] -= eps
        bp, _, _, _ = _compute_moments(*params_plus)
        bm, _, _, _ = _compute_moments(*params_minus)
        # Recompute full bound
        def _b(params):
            mm0, mm2, nn0, nn2 = _compute_moments(*params)
            if mm0 <= 0 or nn0 <= 0:
                return 1e18
            return np.sqrt((mm2/mm0) * (nn2/nn0))
        grad[c] = (_b(params_plus) - _b(params_minus)) / (2 * eps)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    """
    Minimize the uncertainty bound using:
    1. Exact generalized eigenvalue solve for the Rayleigh quotient minimum
    2. Multi-start L-BFGS-B for robustness
    3. High-precision polish
    """
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    # --- Step 1: Exact generalized eigenvalue solve ---
    # The bound squared = (7/3) * (n2/n0) since vF = n2/n0 = 7/3 (constant from g(0)=0)
    # Actually vF is NOT constant in general. Let me verify:
    # n0 and n2 both depend on the coefficients, so vF = n2/n0 varies.
    # But the product bound^2 = vf * vF = (m2/m0)*(n2/n0)
    # This is NOT a simple Rayleigh quotient in 3D because it's a product of two ratios.
    # However, we can still use eigenvalue as a good initial guess.

    # Build quadratic form matrices for n0 and n2 in terms of (a0, a4, a8)
    # n0 = x^T N0 x, n2 = x^T N2 x where x = (a0, a4, a8)^T
    # We need to extract these by evaluating at basis vectors

    def _quad_coeffs(func, n=3):
        """Extract symmetric matrix M from quadratic form f(x) = x^T M x."""
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    e = np.zeros(n)
                    e[i] = 1.0
                    M[i, i] = func(e)
                else:
                    e1 = np.zeros(n)
                    e1[i] = 1.0
                    e2 = np.zeros(n)
                    e2[j] = 1.0
                    e12 = np.zeros(n)
                    e12[i] = 1.0
                    e12[j] = 1.0
                    # f(e1+e2) - f(e1) - f(e2) = 2*M[i,j]
                    M[i, j] = (func(e12) - func(e1) - func(e2)) / 2.0
                    M[j, i] = M[i, j]
        return M

    def _n0_of(x):
        m0v, m2v, n0v, n2v = _compute_moments(x[0], x[1], x[2])
        return n0v

    def _n2_of(x):
        m0v, m2v, n0v, n2v = _compute_moments(x[0], x[1], x[2])
        return n2v

    def _m0_of(x):
        m0v, m2v, n0v, n2v = _compute_moments(x[0], x[1], x[2])
        return m0v

    def _m2_of(x):
        m0v, m2v, n0v, n2v = _compute_moments(x[0], x[1], x[2])
        return m2v

    N0 = _quad_coeffs(_n0_of)
    N2 = _quad_coeffs(_n2_of)
    M0 = _quad_coeffs(_m0_of)
    M2 = _quad_coeffs(_m2_of)

    # The objective is sqrt((x^T M2 x / x^T M0 x) * (x^T N2 x / x^T N0 x))
    # This is a ratio of products, not a simple Rayleigh quotient.
    # But we can use the product matrix approach or just use eigenvalue as init.

    # Try generalized eigenvalue of (N2, N0) as a heuristic init
    try:
        eigvals, eigvecs = eigh(N2, N0)
        # Smallest eigenvalue direction
        v_min = eigvecs[:, 0]
        # Scale to reasonable magnitude
        norm_v = np.linalg.norm(v_min)
        if norm_v > 1e-15:
            v_scaled = v_min / norm_v * 0.5
            val_scaled, _ = _bound_and_grad(*v_scaled)
            if np.isfinite(val_scaled) and val_scaled < best_val:
                best_val = val_scaled
                best = v_scaled.copy()
    except Exception:
        pass

    # Also try eigenvalue of (M2, M0)
    try:
        eigvals2, eigvecs2 = eigh(M2, M0)
        v_min2 = eigvecs2[:, 0]
        norm_v2 = np.linalg.norm(v_min2)
        if norm_v2 > 1e-15:
            v_scaled2 = v_min2 / norm_v2 * 0.5
            val_scaled2, _ = _bound_and_grad(*v_scaled2)
            if np.isfinite(val_scaled2) and val_scaled2 < best_val:
                best_val = val_scaled2
                best = v_scaled2.copy()
    except Exception:
        pass

    # --- Step 2: Multi-start L-BFGS-B ---
    rng = np.random.default_rng(42)

    starts = [best.copy(), seed.copy()]

    # Add structured starts
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        starts.append(best + rng.normal(0.0, scale, 3))
        starts.append(seed + rng.normal(0.0, scale, 3))

    # Add some diverse starts
    for i in range(15):
        starts.append(rng.normal(0.0, 1.0 / (i + 1), 3))

    for s in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # --- Step 3: High-precision polish ---
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Final verification
    final_val, _ = _bound_and_grad(*best)
    if not np.isfinite(final_val):
        best = seed.copy()

    return np.array(best)
# EVOLVE-BLOCK-END
