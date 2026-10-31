"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.integrate

_START_TIME = time.time()
_DEADLINE = 195.0  # 15s safety margin before 210s kill

# Precompute the analytic integrals needed for the bound.
# g(x) = exp(-x^2/2) * P(x) where P = c0*H0 + c4*H4 + c8*H8 + c12*H12
# c12 = -(c0 + 12*c4 + 1344*c8) / 467712
#
# ||g||_2^2 = integral exp(-x^2) * P(x)^2 dx
# ||g||_1 = integral exp(-x^2/2) * |P(x)| dx

def _hermite_vals(x):
    """Compute H0, H4, H8, H12 at points x (array)."""
    x = np.asarray(x, dtype=float)
    H0 = np.ones_like(x)
    H4 = 16.0*x**4 - 48.0*x**2 + 12.0
    H8 = 256.0*x**8 - 4608.0*x**6 + 23040.0*x**4 - 34560.0*x**2 + 1344.0
    H12 = 4096.0*x**12 - 184320.0*x**10 + 3225600.0*x**8 - 22809600.0*x**6 + 7077880.0*x**4 - 84672000.0*x**2 + 467712.0
    return H0, H4, H8, H12

# Precompute the L2 norm squared coefficients
# ||g||_2^2 = sqrt(pi) * (c0^2 * 1 + c4^2 * 384 + c8^2 * 10321920 + c12^2 * 1961987635200)
_L2_COEFFS = np.array([1.0, 384.0, 10321920.0, 1961987635200.0])
_L2_SCALE = np.sqrt(np.pi)

def _P_from_coeffs(c0, c4, c8, x):
    """Compute P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12 at points x."""
    c12 = -(c0 + 12.0*c4 + 1344.0*c8) / 467712.0
    H0, H4, H8, H12 = _hermite_vals(x)
    return c0*H0 + c4*H4 + c8*H8 + c12*H12

def _compute_L1(c0, c4, c8):
    """Compute ||g||_1 = integral exp(-x^2/2) |P(x)| dx using numerical integration."""
    def integrand(x):
        P = _P_from_coeffs(c0, c4, c8, np.array([x]))[0]
        return np.exp(-x**2 / 2.0) * abs(P)
    
    # Integrate over a range where the Gaussian is negligible
    # Use adaptive quadrature
    result, _ = scipy.integrate.quad(integrand, -15, 15, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result

def _compute_L2_sq(c0, c4, c8):
    """Compute ||g||_2^2 analytically."""
    c12 = -(c0 + 12.0*c4 + 1344.0*c8) / 467712.0
    coeffs = np.array([c0, c4, c8, c12])
    return _L2_SCALE * np.sum(_L2_COEFFS * coeffs**2)

def _compute_bound(c0, c4, c8):
    """Compute the upper bound J = (||g||_1)^2 / ||g||_2^2."""
    L2_sq = _compute_L2_sq(c0, c4, c8)
    if L2_sq < 1e-15:
        return 1e10
    L1 = _compute_L1(c0, c4, c8)
    return (L1 * L1) / L2_sq

def _compute_bound_grad(c0, c4, c8, eps=1e-7):
    """Compute bound and gradient using central finite differences."""
    f0 = _compute_bound(c0, c4, c8)
    grad = np.zeros(3)
    for i in range(3):
        h = eps * max(1.0, abs([c0, c4, c8][i]))
        params_plus = np.array([c0, c4, c8])
        params_minus = np.array([c0, c4, c8])
        params_plus[i] += h
        params_minus[i] -= h
        fp = _compute_bound(*params_plus)
        fm = _compute_bound(*params_minus)
        grad[i] = (fp - fm) / (2.0 * h)
    return f0, grad

def construct():
    """Optimize the three Hermite coefficients to minimize the uncertainty bound."""
    # Seed values
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed_coeffs.copy()
    best_score = _compute_bound(*seed_coeffs)
    
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    def objective_grad(params):
        f, g = _compute_bound_grad(params[0], params[1], params[2])
        return f, g
    
    # Primary optimization from seed
    try:
        res = scipy.optimize.minimize(
            objective,
            seed_coeffs,
            method='L-BFGS-B',
            jac=objective_grad,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Multi-start with random points
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 30:
        np.random.seed(42)
        n_starts = 30
        for i in range(n_starts):
            if time.time() - _START_TIME > _DEADLINE - 15:
                break
            # Generate random starting point
            scale = 1.0
            random_coeffs = np.array([
                np.random.uniform(-3.0, 3.0) * scale,
                np.random.uniform(-0.5, 0.5) * scale,
                np.random.uniform(-0.05, 0.05) * scale
            ])
            try:
                res_r = scipy.optimize.minimize(
                    objective,
                    random_coeffs,
                    method='L-BFGS-B',
                    jac=objective_grad,
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if res_r.fun < best_score:
                    best_score = res_r.fun
                    best_coeffs = res_r.x.copy()
            except Exception:
                pass
    
    # Final polish from best found
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 10:
        try:
            res_final = scipy.optimize.minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
