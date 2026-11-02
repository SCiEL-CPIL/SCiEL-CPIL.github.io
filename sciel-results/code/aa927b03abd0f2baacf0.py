import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Precompute grid
x = np.linspace(-12, 12, 8001)
dx = x[1] - x[0]
gauss = np.exp(-x**2 / 2)

# Probabilists' Hermite polynomials He_0, He_4, He_8, He_12 evaluated on x
def he0(x): return np.ones_like(x)
def he4(x): return x**4 - 6*x**2 + 3
def he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

H0 = he0(x)
H4 = he4(x)
H8 = he8(x)
H12 = he12(x)

# Values at 0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def objective(coeffs):
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    l1 = np.sum(np.abs(g)) * dx
    return l1**2

def construct():
    t0 = time.time()
    deadline = t0 + 180  # safety margin before 210s kill

    # Seed coefficients (from the reference solution)
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)

    # Define bounds for differential evolution
    # The coefficients should be reasonably small
    bounds = [(-5, 5), (-5, 5), (-5, 5)]

    # First, try differential evolution for a global search
    if time.time() < deadline - 30:
        try:
            def de_objective(coeffs):
                return objective(coeffs)

            res_de = differential_evolution(
                de_objective, bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if time.time() < deadline - 15 and res_de.fun < best_val:
                best = res_de.x
                best_val = res_de.fun
        except Exception:
            pass

    # Then, refine with Nelder-Mead from the best found so far
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                objective, best, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if time.time() < deadline and res_nm.fun < best_val:
                best = res_nm.x
                best_val = res_nm.fun
        except Exception:
            pass

    # Additional multi-start with different random seeds
    if time.time() < deadline - 5:
        rng = np.random.default_rng(12345)
        for i in range(10):
            if time.time() > deadline - 2:
                break
            start = best + rng.normal(0, 0.3, size=3)
            try:
                res = minimize(
                    objective, start, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                if res.fun < best_val:
                    best = res.x
                    best_val = res.fun
            except Exception:
                pass

    return np.array(best)
