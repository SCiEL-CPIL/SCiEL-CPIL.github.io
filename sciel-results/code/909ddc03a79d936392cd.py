"""Uncertainty bound minimization via unit-sphere reparameterization and adaptive stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0

# Precompute integrals I[n] = integral x^{2n} exp(-x^2) dx
_SP = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _SP
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

# Precompute sign array for Fourier transform
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


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

    # m[k] = sum_{i,j} p_i p_j I[k+i+j] for k=0,1
    # Use matrix form: M = I_matrix @ (p @ p.T)
    # Build I_matrix[i,j] = I[i+j] for i,j in 0..6
    I_mat = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx = i + j
            if idx < 13:
                I_mat[i, j] = _I[idx]

    pp = p @ p  # element-wise square
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    # This is trace(I_k_mat * p p^T) where I_k_mat[i,j] = I[k+i+j]
    m0 = 0.0
    m1 = 0.0
    for i in range(7):
        for j in range(7):
            val = p[i] * p[j]
            if i + j < 13:
                m0 += val * _I[i + j]
            if i + j + 1 < 13:
                m1 += val * _I[i + j + 1]

    # Fourier: q = p * signs
    q = p * _SIGNS

    n0 = 0.0
    n1 = 0.0
    for i in range(7):
        for j in range(7):
            val = q[i] * q[j]
            if i + j < 13:
                n0 += val * _I[i + j]
            if i + j + 1 < 13:
                n1 += val * _I[i + j + 1]

    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m1 / m0
    vF = n1 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _sphere_to_coeffs(t1, t2, r):
    """Convert spherical coordinates to (a0, a4, a8)."""
    st2 = np.sin(t2)
    return np.array([r * st2 * np.cos(t1), r * st2 * np.sin(t1), r * np.cos(t2)])


def _coeffs_to_sphere(coeffs):
    """Convert (a0, a4, a8) to spherical coordinates (t1, t2, r)."""
    r = np.linalg.norm(coeffs)
    if r < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    t1 = np.arctan2(coeffs[1], coeffs[0])
    t2 = np.arccos(np.clip(coeffs[2] / r, -1.0, 1.0))
    return np.array([t1, t2, r])


def _local_search(state0, max_iters, step0, rng, deadline):
    """
    Run adaptive stochastic local search from initial state.
    state = [t1, t2, log_r]
    Returns (best_state, best_val).
    """
    state = state0.copy()
    best_state = state0.copy()
    best_val = _compute_bound(*_sphere_to_coeffs(state0[0], state0[1], np.exp(state0[2])))
    
    steps = np.array([step0, step0, step0 * 0.5])
    temperature = 1.0
    cooling = 0.9997
    accept_count = 0
    total_count = 0
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 0.005, np.pi - 0.005)
        new_state[2] = np.clip(new_state[2], -4.0, 5.0)
        
        coeffs = _sphere_to_coeffs(new_state[0], new_state[1], np.exp(new_state[2]))
        new_val = _compute_bound(*coeffs)
        
        if not np.isfinite(new_val):
            new_val = 1e18
        
        total_count += 1
        
        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            steps *= 0.985
        else:
            delta_val = new_val - best_val
            if delta_val < 1e-12 or (temperature > 1e-10 and np.exp(-delta_val / temperature) > rng.random()):
                state = new_state
                accept_count += 1
                steps *= 1.015
            else:
                steps *= 1.04
        
        if total_count % 50 == 0:
            ratio = accept_count / max(total_count, 1)
            if ratio > 0.5:
                steps *= 1.08
            elif ratio < 0.2:
                steps *= 0.85
            accept_count = 0
            total_count = 0
        
        temperature *= cooling
        if temperature < 1e-12:
            temperature = 1e-12
    
    return best_state, best_val


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization with adaptive stochastic local search.
    """
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad random initialization on sphere
    # Try various radii and random directions
    radii = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    n_samples = 80
    
    for r in radii:
        if time.time() > _DEADLINE - 80:
            break
        for _ in range(n_samples):
            if time.time() > _DEADLINE - 80:
                break
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(0.01, np.pi - 0.01)
            coeffs = _sphere_to_coeffs(t1, t2, r)
            b = _compute_bound(*coeffs)
            if b < best_bound:
                best_bound = b
                best_params = coeffs
    
    # Also try seed neighborhood with various radii
    r_seed = np.linalg.norm(seed)
    for r in [0.5 * r_seed, 0.7 * r_seed, r_seed, 1.3 * r_seed, 2.0 * r_seed]:
        if time.time() > _DEADLINE - 80:
            break
        for _ in range(40):
            dt1 = rng.normal(0, 0.5)
            dt2 = rng.normal(0, 0.5)
            t1 = np.arctan2(seed[1], seed[0]) + dt1
            t2 = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0)) + dt2
            t2 = np.clip(t2, 0.01, np.pi - 0.01)
            coeffs = _sphere_to_coeffs(t1, t2, r)
            b = _compute_bound(*coeffs)
            if b < best_bound:
                best_bound = b
                best_params = coeffs
    
    # Phase 2: Multiple local searches from best candidates
    state_best = _coeffs_to_sphere(best_params)
    log_r_best = np.log(max(state_best[2], 1e-10))
    state_best[2] = log_r_best
    
    # Run local search from best point
    best_state, best_val = _local_search(
        state_best, max_iters=25000, step0=0.15, rng=rng, deadline=_DEADLINE - 20
    )
    if best_val < best_bound:
        best_bound = best_val
        best_params = _sphere_to_coeffs(best_state[0], best_state[1], np.exp(best_state[2]))
    
    # Run additional local searches from other promising points
    additional_starts = []
    # Add a few more random starting points near the best
    for _ in range(5):
        t1 = best_state[0] + rng.normal(0, 0.3)
        t2 = best_state[1] + rng.normal(0, 0.3)
        log_r = best_state[2] + rng.normal(0, 0.3)
        t2 = np.clip(t2, 0.01, np.pi - 0.01)
        log_r = np.clip(log_r, -4.0, 5.0)
        additional_starts.append(np.array([t1, t2, log_r]))
    
    # Also try a few completely different regions
    for _ in range(3):
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(0.1, np.pi - 0.1)
        log_r = rng.uniform(-1.0, 2.0)
        additional_starts.append(np.array([t1, t2, log_r]))
    
    for start in additional_starts:
        if time.time() > _DEADLINE - 15:
            break
        s, v = _local_search(start, max_iters=5000, step0=0.1, rng=rng, deadline=_DEADLINE - 15)
        if v < best_bound:
            best_bound = v
            best_params = _sphere_to_coeffs(s[0], s[1], np.exp(s[2]))
    
    # Phase 3: Fine-tuning from the best point
    state_final = _coeffs_to_sphere(best_params)
    state_final[2] = np.log(max(state_final[2], 1e-10))
    
    best_state, best_val = _local_search(
        state_final, max_iters=30000, step0=0.03, rng=rng, deadline=_DEADLINE - 5
    )
    if best_val < best_bound:
        best_bound = best_val
        best_params = _sphere_to_coeffs(best_state[0], best_state[1], np.exp(best_state[2]))
    
    # Verify the result
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()
    
    return best_params
# EVOLVE-BLOCK-END
