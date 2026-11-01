"""Uncertainty bound minimization via unit-sphere reparameterization and adaptive stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_from_unit_sphere(c0, c4, c8):
    """
    Given a unit vector (c0, c4, c8) on the sphere, compute the uncertainty bound.
    We scale the coefficients by an optimal amplitude to minimize the bound.
    """
    # The bound is homogeneous: if we scale (a0,a4,a8) by t, the bound scales by |t|
    # So we need to find the optimal t. But actually the bound formula is:
    # bound = sqrt(vf * vF) where vf = m2/m0, vF = n2/n0
    # m and n are quadratic in the coefficients, so m2/m0 is homogeneous of degree 0
    # This means the bound is independent of scale!
    # Let's verify: m[k] = sum p_i p_j I[...] where p_k are linear in a's
    # So m[k] is quadratic in a's. m2/m0 is ratio of two quadratics = degree 0.
    # Therefore the bound depends only on the direction, not the magnitude.
    
    # So we can just evaluate at (c0, c4, c8) directly
    return _compute_bound(c0, c4, c8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization with adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    # Normalize seed to unit sphere
    r_seed = np.linalg.norm(seed)
    if r_seed < 1e-15:
        c_seed = np.array([1.0, 0.0, 0.0])
    else:
        c_seed = seed / r_seed

    best_bound = seed_bound
    best_c = c_seed.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration - random points on unit sphere
    n_explore = 500
    for _ in range(n_explore):
        if time.time() > _DEADLINE - 60:
            break
        # Generate random point on unit sphere
        v = rng.normal(0, 1, 3)
        v = v / np.linalg.norm(v)
        b = _bound_from_unit_sphere(v[0], v[1], v[2])
        if b < best_bound:
            best_bound = b
            best_c = v.copy()

    # Phase 2: Adaptive stochastic local search on the unit sphere
    # State: (c0, c4, c8) constrained to unit sphere
    state = best_c.copy()
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step size for the sphere
    step_size = 0.1
    temperature = 1.0
    cooling = 0.9998

    # Track acceptance for adaptive step sizing
    accept_count = 0
    total_count = 0

    max_iters = 40000
    iter_count = 0

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 10:
            break

        # Generate perturbation: add Gaussian noise and renormalize
        delta = rng.normal(0, step_size, 3)
        new_state = state + delta
        norm = np.linalg.norm(new_state)
        if norm < 1e-15:
            new_state = rng.normal(0, 1, 3)
            new_state = new_state / np.linalg.norm(new_state)
        else:
            new_state = new_state / norm

        new_val = _bound_from_unit_sphere(new_state[0], new_state[1], new_state[2])
        if not np.isfinite(new_val):
            new_val = 1e18

        delta_val = new_val - best_val
        total_count += 1

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            step_size *= 0.98
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-10 or (temperature > 1e-15 and np.exp(-delta_val / temperature) > rng.random()):
                state = new_state
                accept_count += 1
                step_size *= 1.02
            else:
                step_size *= 1.05

        # Adaptive step size adjustment every 100 iterations
        if total_count % 100 == 0:
            ratio = accept_count / total_count
            if ratio > 0.6:
                step_size *= 1.15
            elif ratio < 0.3:
                step_size *= 0.85
            # Clamp step size
            step_size = np.clip(step_size, 0.001, 1.0)
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

        iter_count += 1

    # Phase 3: Fine-tuning with smaller steps
    step_size = 0.01
    state = best_state.copy()
    temperature = 1e-3
    cooling = 0.999

    for _ in range(25000):
        if time.time() > _DEADLINE - 5:
            break

        delta = rng.normal(0, step_size, 3)
        new_state = state + delta
        norm = np.linalg.norm(new_state)
        if norm < 1e-15:
            new_state = rng.normal(0, 1, 3)
            new_state = new_state / np.linalg.norm(new_state)
        else:
            new_state = new_state / norm

        new_val = _bound_from_unit_sphere(new_state[0], new_state[1], new_state[2])
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            step_size *= 0.97
        else:
            if new_val - best_val < 1e-12 or (temperature > 1e-15 and np.exp(-(new_val - best_val) / temperature) > rng.random()):
                state = new_state
                step_size *= 1.01
            else:
                step_size *= 1.03

        temperature *= cooling
        if temperature < 1e-12:
            temperature = 1e-12

    # Return the best coefficients
    result = best_state.copy()

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
