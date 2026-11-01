"""Uncertainty bound minimization via spherical reparameterization and adaptive stochastic local search."""
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
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
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
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
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


def _bound_from_unit_sphere(c0, c4, c8, scale):
    """
    Compute bound for normalized direction (c0,c4,c8) with norm 1, scaled by 'scale'.
    """
    return _compute_bound(scale * c0, scale * c4, scale * c8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses direct spherical search with multi-scale exploration and adaptive local search.
    """
    rng = np.random.default_rng(42)

    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_norm = np.linalg.norm(seed)
    seed_dir = seed / seed_norm
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_scale = seed_norm
    best_dir = seed_dir.copy()

    # Phase 1: Multi-scale exploration on the unit sphere
    # Try a wide range of scales with many random directions
    scales = np.logspace(-2, 2, 50)  # from 0.01 to 100
    n_dirs = 200

    for s in scales:
        if time.time() > _DEADLINE - 80:
            break
        for _ in range(n_dirs):
            if time.time() > _DEADLINE - 80:
                break
            # Random point on unit sphere
            v = rng.standard_normal(3)
            v /= np.linalg.norm(v)
            b = _bound_from_unit_sphere(v[0], v[1], v[2], s)
            if b < best_bound:
                best_bound = b
                best_scale = s
                best_dir = v.copy()

    # Phase 2: Local search around best point with adaptive steps
    # Work in (log_scale, theta1, theta2) parameterization
    def to_spherical(c):
        r = np.linalg.norm(c)
        if r < 1e-15:
            return 0.0, 0.0, 0.0
        t1 = np.arctan2(c[1], c[0])
        t2 = np.arccos(np.clip(c[2] / r, -1.0, 1.0))
        return r, t1, t2

    def from_spherical(r, t1, t2):
        if r < 1e-15:
            return np.array([0.0, 0.0, 0.0])
        return np.array([
            r * np.sin(t2) * np.cos(t1),
            r * np.sin(t2) * np.sin(t1),
            r * np.cos(t2)
        ])

    # State: (log_scale, theta1, theta2)
    log_s_best = np.log(best_scale)
    _, t1_best, t2_best = to_spherical(best_dir * best_scale)
    state = np.array([log_s_best, t1_best, t2_best])
    best_state = state.copy()
    best_val = best_bound

    # Adaptive simulated annealing on the sphere
    steps = np.array([0.1, 0.1, 0.1])
    temperature = 0.1
    cooling = 0.9998
    accept_count = 0
    total_count = 0
    max_iters = 50000

    for iter_count in range(max_iters):
        if time.time() > _DEADLINE - 20:
            break

        # Generate perturbation
        delta = rng.standard_normal(3) * steps
        new_state = state + delta
        # Wrap theta1
        new_state[1] = (new_state[1] + np.pi) % (2 * np.pi) - np.pi
        # Clamp theta2
        new_state[2] = np.clip(new_state[2], 0.001, np.pi - 0.001)
        # Clamp log_scale
        new_state[0] = np.clip(new_state[0], -4.0, 3.0)

        # Evaluate
        new_c = from_spherical(np.exp(new_state[0]), new_state[1], new_state[2])
        new_val = _compute_bound(new_c[0], new_c[1], new_c[2])
        if not np.isfinite(new_val):
            new_val = 1e18

        delta_val = new_val - best_val
        total_count += 1

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            steps *= 0.97
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-10 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.01
            else:
                steps *= 1.03

        # Adaptive step size
        if total_count % 50 == 0:
            ratio = accept_count / total_count
            if ratio > 0.65:
                steps *= 1.15
            elif ratio < 0.25:
                steps *= 0.75
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-8:
            temperature = 1e-8

    # Phase 3: Fine-tuning with very small steps
    steps = np.array([0.005, 0.005, 0.005])
    state = best_state.copy()
    temperature = 1e-4
    cooling = 0.9995

    for _ in range(30000):
        if time.time() > _DEADLINE - 5:
            break

        delta = rng.standard_normal(3) * steps
        new_state = state + delta
        new_state[1] = (new_state[1] + np.pi) % (2 * np.pi) - np.pi
        new_state[2] = np.clip(new_state[2], 0.0001, np.pi - 0.0001)
        new_state[0] = np.clip(new_state[0], -4.0, 3.0)

        new_c = from_spherical(np.exp(new_state[0]), new_state[1], new_state[2])
        new_val = _compute_bound(new_c[0], new_c[1], new_c[2])
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.96
        else:
            if new_val - best_val < 1e-14 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                steps *= 1.005
            else:
                steps *= 1.02

        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

    # Convert back to (a0, a4, a8)
    r_final = np.exp(best_state[0])
    t1_final = best_state[1]
    t2_final = best_state[2]
    result = from_spherical(r_final, t1_final, t2_final)

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
