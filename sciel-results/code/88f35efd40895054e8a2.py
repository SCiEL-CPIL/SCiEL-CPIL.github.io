"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0 (Physicists' Hermite)
    h0_0 = eval_hermite(0, 0)
    h4_0 = eval_hermite(4, 0)
    h8_0 = eval_hermite(8, 0)
    h12_0 = eval_hermite(12, 0)

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid
    # Range [-10, 10] is sufficient for e^-x^2/2 decay
    # 20000 points for high accuracy of L1 norm
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]

    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    # Weights for exact L2 norm calculation: W_i = sqrt(pi) * 2^i * i!
    import math
    W0 = math.sqrt(math.pi) * 2**0 * math.factorial(0)
    W4 = math.sqrt(math.pi) * 2**4 * math.factorial(4)
    W8 = math.sqrt(math.pi) * 2**8 * math.factorial(8)
    W12 = math.sqrt(math.pi) * 2**12 * math.factorial(12)

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly * gauss

        # L2 norm squared (Exact)
        norm_sq = c0**2 * W0 + c4**2 * W4 + c8**2 * W8 + c12**2 * W12
        if norm_sq < 1e-15:
            return 1e10

        # L1 norm of g (Numerical)
        l1_g = np.sum(np.abs(g)) * dx

        # Uncertainty product bound ~ (L1 / sqrt(L2))^2 = L1^2 / L2
        return (l1_g * l1_g) / norm_sq

    # Seed values (from current program, but we will re-optimize)
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    np.random.seed(42)
    
    # Strategy 1: Multi-start Powell
    n_starts = 100
    max_iter_per_start = 1000
    
    for i in range(n_starts):
        if time.time() > deadline - 20.0:
            break

        if i == 0:
            x0 = [c0_init, c4_init, c8_init]
        else:
            # Random initialization in a reasonable range
            # Coefficients are expected to be small/moderate
            x0 = [
                np.random.uniform(-5, 5),
                np.random.uniform(-1, 1),
                np.random.uniform(-0.1, 0.1)
            ]

        try:
            res = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='Powell',
                options={'maxiter': max_iter_per_start, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True}
            )

            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    # Strategy 2: Polish with L-BFGS-B from the best found
    if time.time() < deadline - 10.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass
            
    # Strategy 3: One more polish with Nelder-Mead for robustness
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
