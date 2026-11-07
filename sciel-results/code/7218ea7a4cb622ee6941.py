"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_value_fast(c0, c1, c2):
    """Compute r^2/(2*pi) using fast numpy polynomial operations."""
    c0f, c1f, c2f = float(c0), float(c1), float(c2)
    c3f = -(c0f + 12.0 * c1f + 16.0 * c2f) / 23040.0

    # q(y) where y = x^2, coefficients from y^5 down to y^0
    q_coeffs = np.array([
        4096.0 * c3f,
        -46080.0 * c3f,
        290304.0 * c3f + 256.0 * c2f,
        -774144.0 * c3f - 1344.0 * c2f,
        860160.0 * c3f + 2688.0 * c2f + 16.0 * c1f,
        -322560.0 * c3f - 1344.0 * c2f - 48.0 * c1f
    ])

    # Find roots
    roots_y = np.roots(q_coeffs)
    real_roots_y = [r.real for r in roots_y if abs(r.imag) < 1e-9 and r.real > 1e-10]

    if not real_roots_y:
        return np.inf

    # Check sign change at each real positive root
    valid_roots = []
    for r in real_roots_y:
        eps = 1e-6 * max(1.0, r)
        q_neg = np.polyval(q_coeffs, r - eps)
        q_pos = np.polyval(q_coeffs, r + eps)
        if q_neg * q_pos < 0:
            valid_roots.append(r)

    if not valid_roots:
        return np.inf

    r_max_y = max(valid_roots)
    return r_max_y / (2.0 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin

    # Seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_value_fast(seed[0], seed[1], seed[2])
    best_c = seed.copy()

    # Random sampling
    np.random.seed(42)
    n_samples = 15000
    for i in range(n_samples):
        if time.time() > deadline - 30.0:
            break
        c0 = np.random.uniform(-5, 5)
        c1 = np.random.uniform(-5, 5)
        c2 = np.random.uniform(-5, 5)
        try:
            val = _compute_value_fast(c0, c1, c2)
            if val < best_val:
                best_val = val
                best_c = np.array([c0, c1, c2])
        except:
            pass

    # Local optimization from best
    if time.time() < deadline - 20.0:
        def objective(x):
            try:
                return _compute_value_fast(x[0], x[1], x[2])
            except:
                return 1e10

        res = minimize(objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x

    # Multi-start local optimization
    if time.time() < deadline - 10.0:
        for i in range(200):
            if time.time() > deadline - 5.0:
                break
            c0 = np.random.uniform(-3, 3)
            c1 = np.random.uniform(-3, 3)
            c2 = np.random.uniform(-3, 3)
            try:
                def objective(x):
                    try:
                        return _compute_value_fast(x[0], x[1], x[2])
                    except:
                        return 1e10
                res = minimize(objective, [c0, c1, c2], method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
            except:
                pass

    return np.array(best_c)
# EVOLVE-BLOCK-END
