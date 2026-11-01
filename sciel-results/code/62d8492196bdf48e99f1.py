"""Uncertainty bound minimization via normalized-vector reparameterization and stochastic local search on the unit sphere."""
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

    # Precompute I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier transform coefficients
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


def _bound_from_normalized(c, scale):
    """
    Compute bound from normalized direction c (unit vector) and scale factor.
    The actual coefficients are a = scale * c.
    """
    a0, a4, a8 = scale * c[0], scale * c[1], scale * c[2]
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses normalized-vector reparameterization: search on unit sphere for direction,
    and 1D search for scale. Stochastic local search with adaptive steps.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(42)

    # Initialize: normalize seed direction
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        c_seed = seed / r_seed
    else:
        c_seed = np.array([1.0, 0.0, 0.0])

    # Phase 1: Broad exploration - sample random directions and scales
    # Sample many random unit vectors and test various scales
    n_dirs = 200
    scales = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]

    for _ in range(n_dirs):
        if time.time() > _DEADLINE - 80:
            break
        # Random unit vector in 3D
        v = rng.normal(0, 1, 3)
        nv = np.linalg.norm(v)
        if nv < 1e-15:
            continue
        c = v / nv

        for s in scales:
            if time.time() > _DEADLINE - 80:
                break
            b = _bound_from_normalized(c, s)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_params = s * c

    # Also explore neighborhood of seed direction
    for s in [0.5 * r_seed, 0.7 * r_seed, r_seed, 1.3 * r_seed, 1.5 * r_seed, 2.0 * r_seed]:
        if time.time() > _DEADLINE - 80:
            break
        for _ in range(50):
            # Perturb the direction
            delta = rng.normal(0, 0.2, 3)
            c_new = c_seed + delta
            cn = np.linalg.norm(c_new)
            if cn < 1e-15:
                continue
            c_new = c_new / cn
            b = _bound_from_normalized(c_new, s)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_params = s * c_new

    # Phase 2: Stochastic local search on the sphere + scale
    # State: (theta1, theta2, log_scale) where direction is on unit sphere
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    log_scale = np.log(r_best)
    c_best = best_params / r_best
    theta1_best = np.arctan2(c_best[1], c_best[0])
    theta2_best = np.arccos(np.clip(c_best[2], -1.0, 1.0))

    # State: [theta1, theta2, log_scale]
    state = np.array([theta1_best, theta2_best, log_scale])
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step sizes for each dimension
    steps = np.array([0.05, 0.05, 0.1])
    temperature = 0.1
    cooling = 0.9998
    accept_count = 0
    total_count = 0

    max_iters = 50000
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 15:
            break

        # Generate perturbation
        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta

        # Wrap theta1
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        # Clamp theta2 to avoid poles
        new_state[1] = np.clip(new_state[1], 0.01, np.pi - 0.01)
        # Clamp log_scale
        new_state[2] = np.clip(new_state[2], -2.0, 4.0)

        # Convert to direction and scale
        st2 = np.sin(new_state[1])
        c_new = np.array([
            st2 * np.cos(new_state[0]),
            st2 * np.sin(new_state[0]),
            np.cos(new_state[1])
        ])
        s_new = np.exp(new_state[2])

        new_val = _bound_from_normalized(c_new, s_new)
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
            # Accept with Metropolis probability
            if delta_val < 1e-12:
                state = new_state
                accept_count += 1
                steps *= 1.01
            elif np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.01
            else:
                steps *= 1.03

        # Adaptive step size based on acceptance rate
        if total_count % 200 == 0:
            ratio = accept_count / total_count
            if ratio > 0.7:
                steps *= 1.15
            elif ratio < 0.2:
                steps *= 0.75
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-8:
            temperature = 1e-8

    # Phase 3: Fine-tuning with smaller steps
    steps = np.array([0.005, 0.005, 0.01])
    state = best_state.copy()
    temperature = 1e-4
    cooling = 0.9995

    for _ in range(30000):
        if time.time() > _DEADLINE - 5:
            break

        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 0.001, np.pi - 0.001)
        new_state[2] = np.clip(new_state[2], -2.0, 4.0)

        st2 = np.sin(new_state[1])
        c_new = np.array([
            st2 * np.cos(new_state[0]),
            st2 * np.sin(new_state[0]),
            np.cos(new_state[1])
        ])
        s_new = np.exp(new_state[2])

        new_val = _bound_from_normalized(c_new, s_new)
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.95
        else:
            if new_val - best_val < 1e-14:
                state = new_state
                steps *= 1.005
            elif np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                steps *= 1.005
            else:
                steps *= 1.015

        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

    # Convert back to (a0, a4, a8)
    r_final = np.exp(best_state[2])
    t1_final = best_state[0]
    t2_final = best_state[1]
    st2 = np.sin(t2_final)
    c_final = np.array([
        st2 * np.cos(t1_final),
        st2 * np.sin(t1_final),
        np.cos(t2_final)
    ])
    result = r_final * c_final

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
