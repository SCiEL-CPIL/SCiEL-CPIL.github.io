"""Rewritten program: analytic gradient-based optimization of Hermite coefficients for uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature with high accuracy
    n_quad = 300
    t_nodes, t_weights = hermgauss(n_quad)
    x_nodes = np.sqrt(2.0) * t_nodes
    x_weights = np.sqrt(2.0) * t_weights

    # Precompute powers of x
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2

    # Precompute Hermite polynomials at quadrature nodes
    H0 = np.ones_like(x_nodes)
    H4 = 16.0 * x4 - 48.0 * x2 + 12.0
    H8 = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12 = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
           - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    # Weighted Hermite polynomials for fast computation
    wH0 = x_weights * H0
    wH4 = x_weights * H4
    wH8 = x_weights * H8
    wH12 = x_weights * H12

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_bound_and_gradient(coeffs):
        """Compute the C4 upper bound and its gradient analytically."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        # P(x) and F(P)(x) at quadrature nodes
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12  # i^12 = -1

        # L2 norms
        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10, np.zeros(3)

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        # L4 norms
        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10, np.zeros(3)

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        # The bound on C4
        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

        # Compute gradient using analytic derivatives
        # d(bound)/d(a_k) for k in {0, 4, 8}
        # bound = (n4g * n4fg) / (n2g * n2fg)
        # Let's compute partial derivatives

        # d(norm2_g_sq)/d(a_k) = 2 * sum(x_weights * P * H_k)
        # d(norm2_fg_sq)/d(a_k) = 2 * sum(x_weights * FP * H_k)
        # d(norm4_g_4)/d(a_k) = 4 * sum(x_weights * P^3 * H_k)
        # d(norm4_fg_4)/d(a_k) = 4 * sum(x_weights * FP^3 * H_k)

        # But a12 also depends on a_k, so we need chain rule
        # da12/da0 = 1/10395, da12/da4 = -3/10395, da12/da8 = 105/10395

        da12_da = np.array([1.0, -3.0, 105.0]) / 10395.0

        # Effective derivative of P and FP w.r.t. a_k including a12 dependence
        # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        # dP/da_k = H_k + (da12/da_k)*H12
        # FP = a0*H0 + a4*H4 + a8*H8 - a12*H12
        # dFP/da_k = H_k - (da12/da_k)*H12

        H = np.array([H0, H4, H8])  # shape (3, n_quad)
        dP_da = H + da12_da[:, None] * H12[None, :]
        dFP_da = H - da12_da[:, None] * H12[None, :]

        # Derivatives of norms
        d_norm2_g_sq_da = 2.0 * (x_weights[None, :] * P[None, :] * dP_da).sum(axis=1)
        d_norm2_fg_sq_da = 2.0 * (x_weights[None, :] * FP[None, :] * dFP_da).sum(axis=1)
        d_norm4_g_4_da = 4.0 * (x_weights[None, :] * P[None, :]**3 * dP_da).sum(axis=1)
        d_norm4_fg_4_da = 4.0 * (x_weights[None, :] * FP[None, :]**3 * dFP_da).sum(axis=1)

        # d(norm2_g)/da_k = d_norm2_g_sq_da / (2 * norm2_g)
        d_norm2_g_da = d_norm2_g_sq_da / (2.0 * norm2_g)
        d_norm2_fg_da = d_norm2_fg_sq_da / (2.0 * norm2_fg)
        d_norm4_g_da = d_norm4_g_4_da / (4.0 * norm4_g)
        d_norm4_fg_da = d_norm4_fg_4_da / (4.0 * norm4_fg)

        # bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)
        # log(bound) = log(norm4_g) + log(norm4_fg) - log(norm2_g) - log(norm2_fg)
        # d(bound)/da_k = bound * (d_norm4_g_da/norm4_g + d_norm4_fg_da/norm4_fg - d_norm2_g_da/norm2_g - d_norm2_fg_da/norm2_fg)

        grad = bound * (d_norm4_g_da / norm4_g + d_norm4_fg_da / norm4_fg
                        - d_norm2_g_da / norm2_g - d_norm2_fg_da / norm2_fg)

        return bound, grad

    def objective(coeffs):
        """Compute the upper bound on C4 for the given Hermite coefficients."""
        val, _ = compute_bound_and_gradient(coeffs)
        return val

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Multi-start Nelder-Mead with diverse initial guesses
    np.random.seed(42)
    starts = [seed.copy()]
    # Add diverse starting points
    for _ in range(15):
        start = seed + np.random.randn(3) * np.array([0.5, 0.05, 0.005])
        starts.append(start)

    from scipy.optimize import minimize

    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Gradient-based refinement using conjugate gradient
    if time.time() < deadline - 10.0:
        try:
            def grad_func(coeffs):
                _, grad = compute_bound_and_gradient(coeffs)
                return grad

            result = minimize(
                objective,
                best_coeffs,
                jac=grad_func,
                method='CG',
                options={'maxiter': 500, 'gtol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final tight Nelder-Mead refinement
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
