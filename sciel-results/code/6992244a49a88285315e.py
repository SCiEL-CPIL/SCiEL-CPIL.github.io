"""Uncertainty bound minimization via normalized-vector reparameterization and adaptive stochastic local search on the unit sphere."""
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

    # Precompute moments I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # Compute m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
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

    # Compute n[k] = integral xi^{2k} |G(xi)|^2 dxi
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


def _bound_from_normalized(c0, c4, c8, scale):
    """
    Compute bound from normalized direction (c0, c4, c8) on unit sphere
    and a scale factor. The actual coefficients are scale * (c0, c4, c8).
    """
    a0 = scale * c0
    a4 = scale * c4
    a8 = scale * c8
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses normalized-vector reparameterization with adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Systematic exploration of the parameter space
    # Try a grid of scales and directions
    scales = np.array([0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    
    # Generate diverse directions on the unit sphere
    n_directions = 200
    for scale in scales:
        if time.time() > _DEADLINE - 80:
            break
        for _ in range(n_directions):
            if time.time() > _DEADLINE - 80:
                break
            # Random point on unit sphere
            v = rng.normal(0, 1, 3)
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                continue
            v = v / norm
            b = _bound_from_normalized(v[0], v[1], v[2], scale)
            if b < best_bound:
                best_bound = b
                best_params = np.array([scale * v[0], scale * v[1], scale * v[2]])

    # Phase 2: Local refinement around best point
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    
    # Work in (direction, log_scale) space
    dir_best = best_params / r_best
    log_scale_best = np.log(r_best)
    
    state = np.array([dir_best[0], dir_best[1], dir_best[2], log_scale_best])
    best_state = state.copy()
    best_val = best_bound
    
    # Adaptive step sizes for (c0, c4, c8, log_scale)
    steps = np.array([0.05, 0.05, 0.05, 0.05])
    accept_count = 0
    total_count = 0
    temperature = 0.5
    cooling = 0.9998
    
    max_iters = 40000
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > _DEADLINE - 20:
            break
        
        # Generate perturbation
        delta = rng.normal(0, 1, 4) * steps
        new_state = state + delta
        
        # Normalize the direction part (first 3 components)
        dir_part = new_state[:3]
        dir_norm = np.linalg.norm(dir_part)
        if dir_norm < 1e-15:
            dir_part = np.array([1.0, 0.0, 0.0])
        else:
            dir_part = dir_part / dir_norm
        
        # Clamp log_scale
        new_state[3] = np.clip(new_state[3], -2.0, 3.0)
        new_state[:3] = dir_part
        
        scale = np.exp(new_state[3])
        new_val = _bound_from_normalized(new_state[0], new_state[1], new_state[2], scale)
        
        if not np.isfinite(new_val):
            new_val = 1e18
        
        delta_val = new_val - best_val
        total_count += 1
        
        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            steps *= 0.98
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-10 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.02
            else:
                steps *= 1.04
        
        # Adaptive step size adjustment
        if total_count % 200 == 0:
            ratio = accept_count / total_count
            if ratio > 0.55:
                steps *= 1.15
            elif ratio < 0.25:
                steps *= 0.75
            accept_count = 0
            total_count = 0
        
        # Cool down
        temperature *= cooling
        if temperature < 1e-8:
            temperature = 1e-8
        
        iter_count += 1

    # Phase 3: Multiple restarts with local search
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Start from a perturbed version of the best solution
        if restart == 0:
            start_state = best_state.copy()
        else:
            # Random restart
            v = rng.normal(0, 1, 3)
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / norm
            scale = np.exp(rng.uniform(-1.0, 2.0))
            start_state = np.array([v[0], v[1], v[2], np.log(scale)])
        
        state = start_state.copy()
        steps = np.array([0.02, 0.02, 0.02, 0.02])
        temperature = 0.1
        cooling = 0.9995
        local_best_val = _bound_from_normalized(state[0], state[1], state[2], np.exp(state[3]))
        local_best_state = state.copy()
        
        for _ in range(5000):
            if time.time() > _DEADLINE - 5:
                break
            
            delta = rng.normal(0, 1, 4) * steps
            new_state = state + delta
            
            dir_part = new_state[:3]
            dir_norm = np.linalg.norm(dir_part)
            if dir_norm < 1e-15:
                dir_part = np.array([1.0, 0.0, 0.0])
            else:
                dir_part = dir_part / dir_norm
            
            new_state[3] = np.clip(new_state[3], -2.0, 3.0)
            new_state[:3] = dir_part
            
            scale = np.exp(new_state[3])
            new_val = _bound_from_normalized(new_state[0], new_state[1], new_state[2], scale)
            
            if not np.isfinite(new_val):
                new_val = 1e18
            
            if new_val < local_best_val:
                state = new_state
                local_best_val = new_val
                local_best_state = new_state.copy()
                steps *= 0.97
            else:
                if new_val - local_best_val < 1e-12 or np.exp(-(new_val - local_best_val) / temperature) > rng.random():
                    state = new_state
                    steps *= 1.01
                else:
                    steps *= 1.03
            
            temperature *= cooling
            if temperature < 1e-10:
                temperature = 1e-10
        
        if local_best_val < best_val:
            best_val = local_best_val
            best_state = local_best_state.copy()

    # Convert back to (a0, a4, a8)
    dir_final = best_state[:3]
    scale_final = np.exp(best_state[3])
    result = np.array([
        scale_final * dir_final[0],
        scale_final * dir_final[1],
        scale_final * dir_final[2]
    ])

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
