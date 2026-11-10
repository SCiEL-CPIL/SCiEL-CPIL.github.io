# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _compute_R(c: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute R = ||f||_4 / ||f||_2 where f = sum_m c_m Y_m.
    Y is shape (n_theta * n_phi, 21) precomputed basis values.
    weights is shape (n_theta * n_phi,) quadrature weights.
    """
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    if norm2_sq <= 0:
        return np.inf
    return np.sqrt(norm4_sq) / norm2_sq


def _compute_R_and_grad(c: np.ndarray, Y: np.ndarray, weights: np.ndarray):
    """
    Compute R and its gradient with respect to c.
    R = ||f||_4 / ||f||_2
    """
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    w2 = np.sum(weights * f2)
    w4 = np.sum(weights * f4)
    if w2 <= 0:
        return np.inf, np.zeros_like(c)
    norm2 = np.sqrt(w2)
    norm4 = np.sqrt(w4)
    R = norm4 / norm2

    # dR/dc = (1/norm2) * d(norm4)/dc - (norm4/norm2^2) * d(norm2)/dc
    # d(norm4)/dc = (1/norm4) * sum(weights * 2*f^3 * Y)
    # d(norm2)/dc = (1/norm2) * sum(weights * 2*f * Y)
    df4_dc = np.sum(weights[:, None] * (2 * f3[:, None] if False else 2 * f**3)[:, None] * Y, axis=0) / norm4
    df2_dc = np.sum(weights[:, None] * (2 * f)[:, None] * Y, axis=0) / norm2

    grad = df4_dc / norm2 - (norm4 / (norm2 * norm2)) * df2_dc
    # Simplify: R = norm4/norm2
    # dR = (1/norm2) d(norm4) - (norm4/norm2^2) d(norm2)
    # d(norm4)/dc = (1/(2*norm4)) * d(sum(w*f^4))/dc = (1/(2*norm4)) * sum(w*4*f^3*Y) = (2/norm4)*sum(w*f^3*Y)
    # d(norm2)/dc = (1/(2*norm2)) * d(sum(w*f^2))/dc = (1/(2*norm2)) * sum(w*2*f*Y) = (1/norm2)*sum(w*f*Y)
    # So grad = (1/norm2)*(2/norm4)*sum(w*f^3*Y) - (norm4/norm2^2)*(1/norm2)*sum(w*f*Y)
    #        = (2/(norm2*norm4))*sum(w*f^3*Y) - (norm4/norm2^3)*sum(w*f*Y)

    sum_wf3Y = np.sum(weights[:, None] * (f**3)[:, None] * Y, axis=0)
    sum_wfY = np.sum(weights[:, None] * f[:, None] * Y, axis=0)

    grad = (2.0 / (norm2 * norm4)) * sum_wf3Y - (norm4 / (norm2**3)) * sum_wfY
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing R = ||f||_4 / ||f||_2
    for real orthonormal degree-10 spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget, leave margin

    k = 10
    n_coeffs = 2 * k + 1  # 21

    # Precompute the basis Y_m at quadrature points
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # x = cos(theta), w_gl are weights for integral over [-1,1]
    # Surface integral: dOmega = dphi * d(cos(theta))
    # So weight for each (theta_i, phi_j) point is w_gl[i] * (2*pi/n_phi)
    # Total integral should be 4*pi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Create grid
    cos_theta = np.tile(x, (n_phi, 1)).T  # shape (n_theta, n_phi)
    phi_grid = np.tile(phi, (n_theta, 1))  # shape (n_theta, n_phi)
    weights_grid = np.tile(w_gl[:, None], (1, n_phi)) * dphi  # shape (n_theta, n_phi)

    # Flatten for matrix operations
    cos_theta_flat = cos_theta.ravel()  # shape (n_theta * n_phi,)
    phi_flat = phi_grid.ravel()
    weights_flat = weights_grid.ravel()

    # Precompute the 21 basis functions Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)

    from scipy.special import lpmv

    Y = np.zeros((n_theta * n_phi, n_coeffs), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * np.math.factorial(k - abs_m) / np.math.factorial(k + abs_m))
        P = lpmv(abs_m, k, cos_theta_flat)  # includes Condon-Shortley phase
        if m == 0:
            Y[:, idx] = N_m * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2) * N_m * P * np.cos(m * phi_flat)
        else:
            Y[:, idx] = np.sqrt(2) * N_m * P * np.sin(abs_m * phi_flat)

    # Initial guess: all ones
    best_c = np.ones(n_coeffs, dtype=np.float64)
    best_R = _compute_R(best_c, Y, weights_flat)

    # Strategy: Use differential evolution for global search, then L-BFGS-B for local refinement
    # First, try a few different initial points with L-BFGS-B
    rng = np.random.default_rng(42)

    # Try multiple random initializations with L-BFGS-B
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break

        if i == 0:
            c0 = np.ones(n_coeffs)
        else:
            c0 = rng.standard_normal(n_coeffs)
            c0 /= np.linalg.norm(c0) * 0.1  # Scale to reasonable magnitude

        # Use L-BFGS-B to minimize R
        def objective(c):
            R, _ = _compute_R_and_grad(c, Y, weights_flat)
            return R

        def grad(c):
            _, g = _compute_R_and_grad(c, Y, weights_flat)
            return g

        try:
            result = minimize(objective, c0, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < best_R:
                if result.fun < best_R:
                    best_R = result.fun
                    best_c = result.x.copy()
        except Exception:
            pass

    # Now use differential evolution for global search
    if time.time() < deadline - 120:
        # Define bounds: coefficients should be in reasonable range
        # The constraint is max|c| between 1e-3 and 1e5
        bounds = [(-10, 10)] * n_coeffs

        def de_objective(c):
            R, _ = _compute_R_and_grad(c, Y, weights_flat)
            return R

        try:
            de_result = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-10, polish=True,
                init='sobol',
                workers=1
            )
            if de_result.fun < best_R:
                best_R = de_result.fun
                best_c = de_result.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        def objective(c):
            R, _ = _compute_R_and_grad(c, Y, weights_flat)
            return R

        def grad(c):
            _, g = _compute_R_and_grad(c, Y, weights_flat)
            return g

        try:
            result = minimize(objective, best_c, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_R:
                best_R = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    # Validate the result
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(n_coeffs)

    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs) if max_abs > 0 else np.ones(n_coeffs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)

    return best_c


# EVOLVE-BLOCK-END
