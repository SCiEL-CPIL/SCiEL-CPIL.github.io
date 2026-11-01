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
    Compute bound from normalized direction (c0,c4,c8) on unit sphere and scale.
    The actual coefficients are (scale*c0, scale*c4, scale*c8).
    """
    a0 = scale * c0
    a4 = scale * c4
    a8 = scale * c8
    return _compute_bound(a0, a4, a8)


def _normalize(v):
    """Normalize vector to unit length, handling zero vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / norm


def _project_to_sphere(point, center, step):
    """
    Generate a new point on the unit sphere near 'center' by adding a random
    perturbation in the tangent space and re-normalizing.
    """
    # Generate random perturbation in R^3
    delta = rng_global.normal(0, step, 3)
    new_point = center + delta
    return _normalize(new_point)


def _find_optimal_scale(c0, c4, c8, search_range=(0.1, 50.0)):
    """
    For a given direction on the unit sphere, find the optimal scale by
    1D search over the scale parameter.
    """
    best_scale = search_range[0]
    best_val = _bound_from_unit_sphere(c0, c4, c8, search_range[0])
    
    # Coarse search
    n_coarse = 50
    scales = np.geomspace(search_range[0], search_range[1], n_coarse)
    for s in scales:
        if time.time() > _DEADLINE - 30:
            break
        val = _bound_from_unit_sphere(c0, c4, c8, s)
        if val < best_val:
            best_val = val
            best_scale = s
    
    # Fine search around best
    if best_scale > search_range[0] * 1.01:
        fine_start = max(search_range[0], best_scale * 0.8)
        fine_end = min(search_range[1], best_scale * 1.2)
        n_fine = 30
        fine_scales = np.linspace(fine_start, fine_end, n_fine)
        for s in fine_scales:
            if time.time() > _DEADLINE - 30:
                break
            val = _bound_from_unit_sphere(c0, c4, c8, s)
            if val < best_val:
                best_val = val
                best_scale = s
    
    return best_scale, best_val


rng_global = None


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization: search over direction (c0,c4,c8) with
    sum c_i^2 = 1, and optimize scale separately for each direction.
    """
    global rng_global
    rng_global = np.random.default_rng(42)
    
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)
    
    # Normalize seed to get initial direction
    seed_dir = _normalize(seed)
    seed_scale = np.linalg.norm(seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    best_direction = seed_dir.copy()
    best_scale = seed_scale
    
    # Phase 1: Broad exploration of directions on unit sphere
    n_directions = 200
    for i in range(n_directions):
        if time.time() > _DEADLINE - 80:
            break
        
        # Generate random direction on unit sphere
        v = rng_global.normal(0, 1, 3)
        direction = _normalize(v)
        
        # Find optimal scale for this direction
        scale, val = _find_optimal_scale(direction[0], direction[1], direction[2])
        
        if val < best_bound:
            best_bound = val
            best_direction = direction.copy()
            best_scale = scale
            best_params = np.array([scale * direction[0], scale * direction[1], scale * direction[2]])
    
    # Also explore neighborhood of seed direction
    for _ in range(100):
        if time.time() > _DEADLINE - 80:
            break
        
        # Perturb seed direction
        delta = rng_global.normal(0, 0.3, 3)
        direction = _normalize(seed_dir + delta)
        
        # Find optimal scale
        scale, val = _find_optimal_scale(direction[0], direction[1], direction[2])
        
        if val < best_bound:
            best_bound = val
            best_direction = direction.copy()
            best_scale = scale
            best_params = np.array([scale * direction[0], scale * direction[1], scale * direction[2]])
    
    # Phase 2: Adaptive stochastic local search on the unit sphere
    # State: direction on unit sphere (3D, normalized)
    state_dir = best_direction.copy()
    best_val = best_bound
    best_dir_state = state_dir.copy()
    best_scale_state = best_scale
    
    # Adaptive step size for sphere perturbations
    step_size = 0.1
    accept_count = 0
    total_count = 0
    temperature = 1.0
    cooling = 0.999
    
    max_iters = 25000
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > _DEADLINE - 20:
            break
        
        # Generate new direction via tangent-space perturbation
        new_dir = _project_to_sphere(state_dir, state_dir, step_size)
        
        # Find optimal scale for new direction
        new_scale, new_val = _find_optimal_scale(new_dir[0], new_dir[1], new_dir[2])
        
        if not np.isfinite(new_val):
            new_val = 1e18
        
        delta_val = new_val - best_val
        total_count += 1
        
        if new_val < best_val:
            state_dir = new_dir.copy()
            best_val = new_val
            best_dir_state = new_dir.copy()
            best_scale_state = new_scale
            accept_count += 1
            # Shrink step on improvement
            step_size *= 0.97
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-10 or np.exp(-delta_val / temperature) > rng_global.random():
                state_dir = new_dir.copy()
                accept_count += 1
                step_size *= 1.02
            else:
                step_size *= 1.05
        
        # Adaptive step size adjustment
        if total_count % 100 == 0:
            ratio = accept_count / total_count
            if ratio > 0.6:
                step_size *= 1.1
            elif ratio < 0.3:
                step_size *= 0.8
            accept_count = 0
            total_count = 0
        
        # Cool down
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10
        
        iter_count += 1
    
    # Phase 3: Fine-tuning with smaller steps
    step_size = 0.01
    state_dir = best_dir_state.copy()
    temperature = 1e-3
    cooling = 0.998
    
    for _ in range(15000):
        if time.time() > _DEADLINE - 5:
            break
        
        new_dir = _project_to_sphere(state_dir, state_dir, step_size)
        new_scale, new_val = _find_optimal_scale(new_dir[0], new_dir[1], new_dir[2])
        
        if not np.isfinite(new_val):
            new_val = 1e18
        
        if new_val < best_val:
            state_dir = new_dir.copy()
            best_val = new_val
            best_dir_state = new_dir.copy()
            best_scale_state = new_scale
            step_size *= 0.96
        else:
            if new_val - best_val < 1e-12 or np.exp(-(new_val - best_val) / temperature) > rng_global.random():
                state_dir = new_dir.copy()
                step_size *= 1.01
            else:
                step_size *= 1.03
        
        temperature *= cooling
        if temperature < 1e-12:
            temperature = 1e-12
    
    # Final result
    result = np.array([
        best_scale_state * best_dir_state[0],
        best_scale_state * best_dir_state[1],
        best_scale_state * best_dir_state[2]
    ])
    
    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed
        return seed.copy()
    
    return result
# EVOLVE-BLOCK-END
