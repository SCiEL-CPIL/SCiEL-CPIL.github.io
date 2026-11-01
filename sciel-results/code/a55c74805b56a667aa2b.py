import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Safety margin before 210s


def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """

    # Physicist's Hermite polynomials (expanded):
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 1260x^8 - 7560x^6 + 12600x^4 - 5544x^2 + 945
    #
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    #      = (c12) x^12 + (c8 - 84*c12) x^10 + (210*c8 + 1260*c12) x^8
    #        + (-420*c8 - 7560*c12 + c4) x^6 + (105*c8 + 12600*c12 - 6*c4) x^4
    #        + (-420*c8 - 5544*c12 + 3*c4) x^2 + (c0 + 105*c8 + 945*c12)
    #
    # Constraint: P(0) = c0 + 105*c8 + 945*c12 = 0  =>  c12 = -(c0 + 105*c8)/945
    # Wait, the task says "the degree-12 weight is fixed so g(0) = 0" and the
    # seed uses c12 = -(c0 + 3*c4 + 105*c8)/945. Let me recheck.
    #
    # Actually, looking at the seed: c0=-1.13, c4=0.04, c8=1/3240≈0.00030864
    # c12 = -(-1.13 + 3*0.04 + 105*0.00030864)/945 = -(-1.13+0.12+0.0324)/945 = -(-0.9776)/945 ≈ 0.001034
    #
    # But P(0) = c0*1 + c4*3 + c8*105 + c12*945 = -1.13 + 0.12 + 0.0324 + 0.9776 ≈ 0.0000 ✓
    # So the constraint is: c0 + 3*c4 + 105*c8 + 945*c12 = 0
    # => c12 = -(c0 + 3*c4 + 105*c8) / 945

    def get_c12(c0, c4, c8):
        return -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0

    def compute_poly_coeffs(c0, c4, c8):
        """Return coefficients of P(x) as a list [a12, a10, a8, a6, a4, a2, a0]
        for P(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0"""
        c12 = get_c12(c0, c4, c8)
        a12 = c12
        a10 = c8 - 84.0 * c12
        a8 = 210.0 * c8 + 1260.0 * c12
        a6 = c4 - 420.0 * c8 - 7560.0 * c12
        a4 = 105.0 * c8 + 12600.0 * c12 - 6.0 * c4
        a2 = 3.0 * c4 - 420.0 * c8 - 5544.0 * c12
        a0 = c0 + 105.0 * c8 + 945.0 * c12  # This should be ~0 by constraint
        return np.array([a12, a10, a8, a6, a4, a2, a0])

    def eval_P(x, coeffs):
        """Evaluate P(x) using Horner's method on the even-power polynomial.
        coeffs = [a12, a10, a8, a6, a4, a2, a0]"""
        x2 = x * x
        # P(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
        #      = x^2 * (a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2) + a0
        # Use Horner in x^2:
        result = coeffs[0]
        for i in range(1, 7):
            result = result * x2 + coeffs[i]
        return result

    def objective(coeffs):
        c0, c4, c8 = coeffs
        poly_coeffs = compute_poly_coeffs(c0, c4, c8)

        def integrand(x):
            p_val = eval_P(x, poly_coeffs)
            g_val = p_val * np.exp(-0.5 * x * x)
            return (x ** 4) * np.abs(g_val)

        val, err = quad(integrand, 0, np.inf, limit=200, epsabs=1e-12, epsrel=1e-12)
        return 2.0 * val

    # Initial guesses
    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    # Multi-start with L-BFGS-B (faster and more precise than Nelder-Mead)
    np.random.seed(42)
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]

    num_starts = 80

    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break

        if i == 0:
            x0 = best_coeffs.copy()
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])

        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})

            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish on the best solution
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs
