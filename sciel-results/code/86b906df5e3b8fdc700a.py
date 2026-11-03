import numpy as np
import scipy.optimize as so
import scipy.special as sp
import scipy.integrate as si
import time
import math
import sys

# Hermite polynomials (physicists')
def H(n, x):
    return sp.eval_hermite(n, x)

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Coefficients for H0, H4, H8
    # c12 is derived from g(0)=0
    # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) * exp(-x^2/2)
    # H0(0)=1, H4(0)=-12, H8(0)=1344, H12(0)=-103968
    # c0 - 12 c4 + 1344 c8 - 103968 c12 = 0
    # c12 = (c0 - 12 c4 + 1344 c8) / 103968

    def get_c12(c0, c4, c8):
        return (c0 - 12 * c4 + 1344 * c8) / 103968.0

    def P_poly(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        return c0 * H(0, x) + c4 * H(4, x) + c8 * H(8, x) + c12 * H(12, x)

    def g_func(x, c0, c4, c8):
        return P_poly(x, c0, c4, c8) * np.exp(-0.5 * x**2)

    def g_hat_func(xi, c0, c4, c8):
        # Using the Hermite eigenfunction property:
        # \hat{g}(\xi) = \sum_{k} c_k (-i)^k \sqrt{2\pi} \phi_k(\xi)
        # where \phi_k(\xi) = H_k(\xi) e^{-\xi^2/2} / \sqrt{2^k k! \sqrt{\pi}}
        # For k in {0, 4, 8, 12}, (-i)^k = 1
        # \hat{g}(\xi) = \sqrt{2} e^{-\xi^2/2} \sum_k c_k H_k(\xi) / \sqrt{2^k k!}
        c12 = get_c12(c0, c4, c8)
        coeffs = [c0, c4, c8, c12]
        degrees = [0, 4, 8, 12]
        val = 0.0
        for k, ck in zip(degrees, coeffs):
            factor = ck * H(k, xi) / np.sqrt(2**k * math.factorial(k))
            val += factor
        return np.sqrt(2) * val * np.exp(-0.5 * xi**2)

    def find_sign_changes(x_grid, func_vals):
        """Find approximate locations where func changes sign."""
        changes = []
        for i in range(len(x_grid) - 1):
            if func_vals[i] * func_vals[i+1] < 0:
                # Linear interpolation for zero crossing
                x0, x1 = x_grid[i], x_grid[i+1]
                f0, f1 = func_vals[i], func_vals[i+1]
                x_cross = x0 - f0 * (x1 - x0) / (f1 - f0)
                changes.append(x_cross)
        return changes

    def compute_l1_norm(c0, c4, c8, is_g=True):
        """Compute L1 norm by finding sign changes and integrating piecewise."""
        if time_left() < 5.0:
            return np.nan

        # Use a fine grid to find sign changes
        x_grid = np.linspace(-15, 15, 20000)
        
        if is_g:
            vals = P_poly(x_grid, c0, c4, c8) * np.exp(-0.5 * x_grid**2)
        else:
            vals = g_hat_func(x_grid, c0, c4, c8)
        
        # Find sign changes
        sign_changes = find_sign_changes(x_grid, vals)
        
        # Sort and add boundaries
        points = [-15.0] + sign_changes + [15.0]
        points = sorted(set(points))
        
        # Integrate |vals| over each interval
        total = 0.0
        for i in range(len(points) - 1):
            a, b = points[i], points[i+1]
            if b <= a:
                continue
            if time_left() < 3.0:
                return np.nan
            try:
                if is_g:
                    def integrand(x):
                        return np.abs(P_poly(x, c0, c4, c8) * np.exp(-0.5 * x**2))
                else:
                    def integrand(x):
                        return np.abs(g_hat_func(x, c0, c4, c8))
                integral, _ = si.quad(integrand, a, b, limit=100, points=None)
                total += integral
            except Exception:
                return np.nan
        return total

    def compute_objective(vars):
        if time_left() < 10.0:
            return np.inf

        c0, c4, c8 = vars

        # Calculate c12
        c12 = get_c12(c0, c4, c8)

        # ||g||_2^2 using Gauss-Hermite quadrature
        # g(x)^2 = P(x)^2 exp(-x^2)
        x_gh, w_gh = np.polynomial.hermite.hermgauss(13)
        P_vals = c0 * H(0, x_gh) + c4 * H(4, x_gh) + c8 * H(8, x_gh) + c12 * H(12, x_gh)
        norm2_sq = np.sum(w_gh * P_vals**2)

        if norm2_sq < 1e-10:
            return np.inf

        # ||g||_1
        norm1 = compute_l1_norm(c0, c4, c8, is_g=True)
        if np.isnan(norm1) or norm1 < 0:
            return np.inf

        # ||g_hat||_1
        norm1_hat = compute_l1_norm(c0, c4, c8, is_g=False)
        if np.isnan(norm1_hat) or norm1_hat < 0:
            return np.inf

        return (norm1 * norm1_hat) / norm2_sq

    # Optimization
    # Initial guess from seed
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    x0 = np.array([c0_init, c4_init, c8_init])

    best_val = np.inf
    best_x = x0

    # Try multiple random starts and the seed
    np.random.seed(42)
    starts = [x0]

    # Add random perturbations with varying scales
    for scale in [0.05, 0.1, 0.2, 0.5]:
        for _ in range(3):
            pert = np.random.normal(0, scale, 3)
            starts.append(x0 + pert)

    # Also try some structured starts
    for c0_try in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for c4_try in [-0.1, -0.05, 0, 0.05, 0.1]:
            for c8_try in [-0.001, -0.0005, 0, 0.0005, 0.001]:
                starts.append(np.array([c0_try, c4_try, c8_try]))

    # Limit total starts based on time
    max_starts = min(len(starts), 50)
    starts = starts[:max_starts]

    for s in starts:
        if time_left() < 15.0:
            break
        try:
            # Use Nelder-Mead with reasonable tolerances
            res = so.minimize(compute_objective, s, method='Nelder-Mead',
                            options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final polish if time allows
    if time_left() > 30.0 and np.isfinite(best_val):
        try:
            res = so.minimize(compute_objective, best_x, method='Nelder-Mead',
                            options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x
