"""Optimized Hermite coefficient search for uncertainty inequality bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.integrate import quad


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes
    n_quad = 500
    t_nodes, t_weights = hermgauss(n_quad)
    x_nodes = np.sqrt(2.0) * t_nodes
    x_weights = np.sqrt(2.0) * t_weights

    # Precompute powers
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2

    # Hermite polynomials at quadrature nodes
    H0 = np.ones_like(x_nodes)
    H4 = 16.0 * x4 - 48.0 * x2 + 12.0
    H8 = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12 = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
           - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_bound(coeffs):
        """Compute the C4 upper bound."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12  # i^12 = -1

        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        return (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

    def compute_gradient(coeffs):
        """Compute gradient of the C4 bound analytically."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12

        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return np.zeros(3)

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return np.zeros(3)

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

        da12_da = np.array([1.0, -3.0, 105.0]) / 10395.0

        H = np.array([H0, H4, H8])
        dP_da = H + da12_da[:, None] * H12[None, :]
        dFP_da = H - da12_da[:, None] * H12[None, :]

        d_norm2_g_sq_da = 2.0 * (x_weights[None, :] * P[None, :] * dP_da).sum(axis=1)
        d_norm2_fg_sq_da = 2.0 * (x_weights[None, :] * FP[None, :] * dFP_da).sum(axis=1)
        d_norm4_g_4_da = 4.0 * (x_weights[None, :] * P[None, :]**3 * dP_da).sum(axis=1)
        d_norm4_fg_4_da = 4.0 * (x_weights[None, :] * FP[None, :]**3 * dFP_da).sum(axis=1)

        d_norm2_g_da = d_norm2_g_sq_da / (2.0 * norm2_g)
        d_norm2_fg_da = d_norm2_fg_sq_da / (2.0 * norm2_fg)
        d_norm4_g_da = d_norm4_g_4_da / (4.0 * norm4_g)
        d_norm4_fg_da = d_norm4_fg_4_da / (4.0 * norm4_fg)

        grad = bound * (d_norm4_g_da / norm4_g + d_norm4_fg_da / norm4_fg
                        - d_norm2_g_da / norm2_g - d_norm2_fg_da / norm2_fg)

        return grad

    def objective(coeffs):
        return compute_bound(coeffs)

    def grad_func(coeffs):
        return compute_gradient(coeffs)

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Phase 1: Angular parameterization - exploit scale invariance
    # The C4 bound is homogeneous of degree 0 in the coefficients, so we can
    # parameterize directions on the unit sphere using two angles.
    def angles_to_coeffs(theta1, theta2):
        """Convert spherical angles to coefficient vector (normalized)."""
        sin_t1 = np.sin(theta1)
        cos_t1 = np.cos(theta1)
        sin_t2 = np.sin(theta2)
        cos_t2 = np.cos(theta2)
        # Spherical coordinates: x = sin(t1)*cos(t2), y = sin(t1)*sin(t2), z = cos(t1)
        return np.array([sin_t1 * cos_t2, sin_t1 * sin_t2, cos_t1])

    def bound_from_angles(theta1, theta2):
        """Compute bound from angular parameters."""
        direction = angles_to_coeffs(theta1, theta2)
        # Scale to match seed magnitude for numerical stability
        scale = np.linalg.norm(seed) / (np.linalg.norm(direction) + 1e-30)
        coeffs = direction * scale
        return compute_bound(coeffs)

    # High-resolution grid search over angular space
    if time.time() < deadline - 60.0:
        try:
            n_theta1 = 200
            n_theta2 = 200
            theta1_vals = np.linspace(-np.pi/2, np.pi/2, n_theta1)
            theta2_vals = np.linspace(-np.pi, np.pi, n_theta2)
            
            best_theta1 = theta1_vals[0]
            best_theta2 = theta2_vals[0]
            best_grid_val = best_val
            
            for i, t1 in enumerate(theta1_vals):
                if time.time() > deadline - 60.0:
                    break
                for j, t2 in enumerate(theta2_vals):
                    val = bound_from_angles(t1, t2)
                    if val < best_grid_val:
                        best_grid_val = val
                        best_theta1 = t1
                        best_theta2 = t2
            
            if best_grid_val < best_val:
                best_val = best_grid_val
                best_coeffs = angles_to_coeffs(best_theta1, best_theta2) * (np.linalg.norm(seed) / (np.linalg.norm(angles_to_coeffs(best_theta1, best_theta2)) + 1e-30))
        except Exception:
            pass

    # Phase 2: Refine angular minimum with local optimization
    if time.time() < deadline - 40.0:
        try:
            def angular_objective(angles):
                return bound_from_angles(angles[0], angles[1])
            
            # Start from best angular point or seed direction
            seed_dir = seed / (np.linalg.norm(seed) + 1e-30)
            # Convert seed direction to angles
            x, y, z = seed_dir
            r_xy = np.sqrt(x**2 + y**2)
            if r_xy > 1e-15:
                init_theta1 = np.arcsin(r_xy)
                init_theta2 = np.arctan2(y, x)
            else:
                init_theta1 = 0.0
                init_theta2 = 0.0
            
            ang_result = minimize(
                angular_objective,
                [init_theta1, init_theta2],
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            
            if ang_result.fun < best_val:
                best_val = ang_result.fun
                best_coeffs = angles_to_coeffs(ang_result.x[0], ang_result.x[1]) * (np.linalg.norm(seed) / (np.linalg.norm(angles_to_coeffs(ang_result.x[0], ang_result.x[1])) + 1e-30))
        except Exception:
            pass

    # Phase 3: Differential evolution in original space for fine-tuning
    if time.time() < deadline - 30.0:
        bounds = [
            (best_coeffs[0] - 1.0, best_coeffs[0] + 1.0),
            (best_coeffs[1] - 0.1, best_coeffs[1] + 0.1),
            (best_coeffs[2] - 0.01, best_coeffs[2] + 0.01),
        ]
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-14,
                mut=(0.5, 1.5),
                recomb=0.8,
                polish=False,
                init='latinhypercube',
                update_interval=2,
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Phase 4: Basin hopping for stochastic local search
    if time.time() < deadline - 20.0:
        try:
            def basin_objective(x):
                return compute_bound(x)

            def basin_gradient(x):
                return compute_gradient(x)

            def basin_local(x):
                res = minimize(
                    basin_objective,
                    x,
                    jac=basin_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                return res.x, res.fun, {'nfev': res.nfev}

            bh_result = basinhopping(
                basin_objective,
                best_coeffs,
                minimizer=basin_local,
                niter=30,
                stepsize=0.3,
                seed=42,
                T=1.0,
                accept_test=None,
            )
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_coeffs = bh_result.x.copy()
        except Exception:
            pass

    # Phase 5: High-precision L-BFGS-B polish
    if time.time() < deadline - 10.0:
        try:
            lb_result = minimize(
                objective,
                best_coeffs,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-16, 'ftol': 1e-18, 'maxls': 50}
            )
            if lb_result.fun < best_val:
                best_val = lb_result.fun
                best_coeffs = lb_result.x.copy()
        except Exception:
            pass

    # Phase 6: Even tighter final polish
    if time.time() < deadline - 2.0:
        try:
            lb_result2 = minimize(
                objective,
                best_coeffs,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-18, 'ftol': 1e-20, 'maxls': 100}
            )
            if lb_result2.fun < best_val:
                best_val = lb_result2.fun
                best_coeffs = lb_result2.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
