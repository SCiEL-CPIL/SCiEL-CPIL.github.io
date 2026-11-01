"""Uncertainty bound minimization via normalized-vector reparameterization and adaptive stochastic local search."""
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


def _bound_from_normalized(c):
    """
    Compute bound from normalized coefficient vector c = (c0, c4, c8) with ||c||=1.
    We search over directions; the optimal scale is found by line search.
    """
    # Try several scales to find the one that minimizes the bound
    # The bound is scale-invariant in a specific way: if we scale (a0,a4,a8) by t,
    # then a12 also scales by t, so P scales by t, and the variance ratio is scale-invariant.
    # Actually, let's verify: P_t(x) = t*P(x), so |P_t|^2 = t^2|P|^2, so m_k = t^2*m_k(P)
    # and n_k = t^2*n_k(P). So vf = m2/m0 and vF = n2/n0 are scale-invariant!
    # This means the bound depends only on the direction, not the magnitude.
    # So we can just use c directly as (a0, a4, a8) since the bound is scale-invariant.
    return _compute_bound(c[0], c[1], c[2])


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses normalized-vector reparameterization with adaptive stochastic local search
    directly on the unit sphere.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration - sample many random directions on the unit sphere
    # Since the bound is scale-invariant, we only need to search over directions
    n_samples = 2000
    for _ in range(n_samples):
        if time.time() > _DEADLINE - 120:
            break
        # Sample uniformly on the unit sphere in R^3
        v = rng.normal(0, 1, 3)
        norm = np.linalg.norm(v)
        if norm > 1e-15:
            c = v / norm
            b = _bound_from_normalized(c)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_params = c.copy()

    # Also try structured grid on the sphere
    # Use spherical coordinates for a finer grid
    n_theta1 = 30
    n_theta2 = 15
    for i in range(n_theta1):
        if time.time() > _DEADLINE - 120:
            break
        t1 = -np.pi + (2 * np.pi) * i / n_theta1
        for j in range(n_theta2):
            t2 = np.pi * (j + 0.5) / n_theta2
            st2 = np.sin(t2)
            c = np.array([st2 * np.cos(t1), st2 * np.sin(t1), np.cos(t2)])
            b = _bound_from_normalized(c)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_params = c.copy()

    # Try neighborhood of the seed direction
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        c_seed = seed / r_seed
        for _ in range(500):
            if time.time() > _DEADLINE - 120:
                break
            delta = rng.normal(0, 0.2, 3)
            c_new = c_seed + delta
            norm = np.linalg.norm(c_new)
            if norm > 1e-15:
                c_new = c_new / norm
                b = _bound_from_normalized(c_new)
                if np.isfinite(b) and b < best_bound:
                    best_bound = b
                    best_params = c_new.copy()

    # Phase 2: Adaptive stochastic local search on the unit sphere
    # State is the normalized vector c in R^3
    state = best_params.copy()
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step size (scalar, applied to all components)
    step_size = 0.1
    accept_count = 0
    total_count = 0
    temperature = 0.5
    cooling = 0.9998

    max_iters = 50000
    iter_count = 0

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 20:
            break

        # Generate perturbation in tangent space, then project back to sphere
        delta = rng.normal(0, 1, 3) * step_size
        new_state = state + delta
        norm = np.linalg.norm(new_state)
        if norm < 1e-15:
            new_state = state
        else:
            new_state = new_state / norm

        new_val = _bound_from_normalized(new_state)
        if not np.isfinite(new_val):
            new_val = 1e18

        delta_val = new_val - best_val
        total_count += 1

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            # Shrink step on improvement
            step_size *= 0.98
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-12 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                step_size *= 1.02
            else:
                step_size *= 1.05

        # Adaptive step size adjustment
        if total_count % 50 == 0:
            ratio = accept_count / total_count
            if ratio > 0.7:
                step_size *= 1.15
            elif ratio < 0.2:
                step_size *= 0.7
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-8:
            temperature = 1e-8

        iter_count += 1

    # Phase 3: Fine-tuning with very small steps
    step_size = 0.01
    state = best_state.copy()
    temperature = 1e-4
    cooling = 0.9995

    for _ in range(30000):
        if time.time() > _DEADLINE - 5:
            break

        delta = rng.normal(0, 1, 3) * step_size
        new_state = state + delta
        norm = np.linalg.norm(new_state)
        if norm < 1e-15:
            new_state = state
        else:
            new_state = new_state / norm

        new_val = _bound_from_normalized(new_state)
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            step_size *= 0.97
        else:
            if new_val - best_val < 1e-14 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                step_size *= 1.01
            else:
                step_size *= 1.03

        temperature *= cooling
        if temperature < 1e-14:
            temperature = 1e-14

    # The result is the normalized vector; we can return it as (a0, a4, a8)
    # since the bound is scale-invariant
    result = best_state.copy()

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound + 1e-10:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
