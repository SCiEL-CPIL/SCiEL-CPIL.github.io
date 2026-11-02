import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start = time.time()
    deadline = 205.0

    # Physicists' Hermite polynomials (up to degree 12)
    H = [
        np.array([1.0]),
        np.array([0.0, 2.0]),
        np.array([1.0, 0.0, -2.0]),
        np.array([0.0, 4.0, 0.0, -4.0]),
        np.array([1.0, 0.0, -12.0, 0.0, 6.0]),
        np.array([0.0, 8.0, 0.0, -48.0, 0.0, 24.0]),
        np.array([1.0, 0.0, -30.0, 0.0, 180.0, 0.0, -30.0]),
        np.array([0.0, 16.0, 0.0, -240.0, 0.0, 960.0, 0.0, -160.0]),
        np.array([1.0, 0.0, -60.0, 0.0, 1680.0, 0.0, -1680.0, 0.0, 105.0]),
        np.array([0.0, 32.0, 0.0, -896.0, 0.0, 13440.0, 0.0, -6720.0, 0.0, 320.0]),
        np.array([1.0, 0.0, -132.0, 0.0, 5808.0, 0.0, -66528.0, 0.0, 33264.0, 0.0, -924.0]),
        np.array([0.0, 64.0, 0.0, -3584.0, 0.0, 107520.0, 0.0, -1971200.0, 0.0, 1662720.0, 0.0, -112640.0]),
        np.array([1.0, 0.0, -264.0, 0.0, 21456.0, 0.0, -665280.0, 0.0, 12454080.0, 0.0, -7654656.0, 0.0, 10395.0]),
    ]

    def poly_coeffs(c0, c4, c8):
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
        a = np.zeros(13)
        a[0] += c0 * H[0]
        a[4] += c4 * H[4]
        a[8] += c8 * H[8]
        a[12] += c12 * H[12]
        return a

    def moments(a):
        # I_k = ∫ x^k a(x) e^{-x^2} dx, exact via Gaussian quadrature (even moments)
        # H_k = ∫ x^k a'(x) e^{-x^2} dx, exact via integration by parts
        N = 400
        x, w = np.polynomial.hermite.hermgauss(N)
        expw = np.exp(-x * x) * w
        # a(x) values
        P = np.zeros_like(x)
        for k in range(13):
            if a[k] != 0.0:
                P += a[k] * x**k
        I = np.array([np.sum((x**k) * P * expw) for k in range(25)])
        # a'(x)
        Pd = np.zeros_like(x)
        for k in range(1, 13):
            if a[k] != 0.0:
                Pd += a[k] * k * x**(k - 1)
        Hm = np.array([np.sum((x**k) * Pd * expw) for k in range(25)])
        return I, Hm

    def objective(c):
        if time.time() > deadline - 2.0:
            return np.inf
        a = poly_coeffs(c[0], c[1], c[2])
        I, Hm = moments(a)
        E = I[0]
        if E <= 0 or not np.isfinite(E):
            return np.inf
        Ex2 = I[2]
        if Ex2 < 0 or not np.isfinite(Ex2):
            return np.inf
        # Frequency side: use f''(x) = a''(x)e^{-x^2/2} - a(x)e^{-x^2/2}
        # ∫(f'')^2 dx = a(0)^2 * sqrt(pi) - 2 H_0 + I_4
        a0 = float(np.sum(a))  # a(0)
        fpp2 = a0 * a0 * np.sqrt(np.pi) - 2.0 * Hm[0] + I[4]
        if fpp2 < 0 or not np.isfinite(fpp2):
            return np.inf
        Efp2 = fpp2 / (4.0 * np.pi**2)
        if Efp2 <= 0:
            return np.inf
        prod = np.sqrt(Ex2 / E) * np.sqrt(Efp2 / E)
        if not np.isfinite(prod):
            return np.inf
        return prod

    # Global search via differential evolution
    best_val = np.inf
    best_x = np.array([0.0, 0.0, 0.0])

    # Evaluate a few sensible points first
    for trial in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
        [0.0, 0.0, 0.01],
        [0.5, 0.0, 0.0],
        [0.0, 0.05, 0.005],
        [-0.5, 0.02, 0.001],
        [0.0, 0.0, 0.001],
    ]:
        v = objective(np.array(trial))
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_x = np.array(trial)

    def bounded_objective(c):
        if time.time() > deadline - 3.0:
            return np.inf
        return objective(c)

    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    try:
        res = differential_evolution(
            bounded_objective, bounds,
            seed=42, maxiter=60, popsize=15,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.9,
            polish=False, init='sobol',
            updating='deferred',
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Local refinement from the best found
    if time.time() < deadline - 5.0:
        try:
            r2 = minimize(bounded_objective, best_x, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(r2.fun) and r2.fun < best_val:
                best_val = r2.fun
                best_x = r2.x.copy()
        except Exception:
            pass

    # A few more local refinements from perturbed points
    if time.time() < deadline - 3.0:
        rng = np.random.RandomState(7)
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            cand = best_x + rng.normal(0, 0.05, 3)
            try:
                r3 = minimize(bounded_objective, cand, method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300, 'adaptive': True})
                if np.isfinite(r3.fun) and r3.fun < best_val:
                    best_val = r3.fun
                    best_x = r3.x.copy()
            except Exception:
                pass

    # If all failed, fall back to a default
    if not np.isfinite(best_val):
        best_x = np.array([0.0, 0.0, 0.0])

    return np.array(best_x[:3], dtype=float)
