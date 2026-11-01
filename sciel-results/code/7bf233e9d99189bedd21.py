"""Uncertainty bound minimization via unit-sphere reparameterization with adaptive stochastic search."""
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

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1
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


def _compute_bound_fast(a0, a4, a8):
    """Fast version using precomputed constants."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # Precomputed I values
    sp = np.sqrt(np.pi)
    I = np.array([
        sp,
        sp * 0.5,
        sp * 0.75,
        sp * 1.3125,
        sp * 2.53125,
        sp * 5.265625,
        sp * 11.54296875,
        sp * 25.9716796875,
        sp * 58.436279296875,
        sp * 131.48162841796875,
        sp * 295.8336639404297,
        sp * 665.6257438659668,
        sp * 1497.6579237004253
    ])

    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

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


def _stochastic_local_search_on_sphere(c_dir, n_iters, initial_step, rng):
    """
    Perform stochastic local search on the unit sphere.
    c_dir is a unit vector in R^3.
    Returns the best direction found and its bound value.
    """
    c_curr = c_dir.copy()
    curr_val = _compute_bound_fast(*c_curr)
    
    best_c = c_dir.copy()
    best_val = curr_val
    
    step = initial_step
    temperature = 1.0
    cooling = 0.9998
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 15:
            break
        
        # Generate random perturbation on the sphere
        # Use small random vector and project to tangent space
        delta = rng.normal(0, 1, 3) * step
        c_new = c_curr + delta
        
        # Normalize to unit sphere
        norm = np.linalg.norm(c_new)
        if norm < 1e-15:
            continue
        c_new = c_new / norm
        
        new_val = _compute_bound_fast(*c_new)
        if not np.isfinite(new_val):
            new_val = 1e18
        
        delta_val = new_val - best_val
        
        if new_val < best_val:
            c_curr = c_new
            best_val = new_val
            best_c = c_new.copy()
            step *= 0.98
        else:
            # Accept with simulated annealing probability
            if delta_val < 1e-12 or np.exp(-delta_val / temperature) > rng.random():
                c_curr = c_new
                step *= 1.02
            else:
                step *= 1.05
        
        # Adaptive step size
        if it % 50 == 0 and it > 0:
            step = np.clip(step, 0.001, 1.0)
        
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10
    
    return best_c, best_val


def _optimize_radius(c_dir, r_min, r_max, n_samples, rng):
    """
    Find optimal radius r such that r * c_dir minimizes the bound.
    Uses golden section search.
    """
    c_dir = c_dir / np.linalg.norm(c_dir)
    
    def bound_at_r(r):
        return _compute_bound_fast(*(r * c_dir))
    
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2
    a, b = r_min, r_max
    
    # First, do a coarse grid search to find a good bracket
    grid_r = np.linspace(r_min, r_max, n_samples)
    grid_vals = np.array([bound_at_r(r) for r in grid_r])
    
    min_idx = np.argmin(grid_vals)
    if min_idx == 0:
        a, b = r_min, grid_r[1]
    elif min_idx == len(grid_r) - 1:
        a, b = grid_r[-2], r_max
    else:
        a, b = grid_r[max(0, min_idx - 1)], grid_r[min(len(grid_r) - 1, min_idx + 1)]
    
    # Refine with golden section
    for _ in range(30):
        if time.time() > _DEADLINE - 10:
            break
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = bound_at_r(x1)
        f2 = bound_at_r(x2)
        
        if f1 < f2:
            b = x2
        else:
            a = x1
    
    r_opt = (a + b) / 2
    val_opt = bound_at_r(r_opt)
    
    return r_opt, val_opt


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization with adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound_fast(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad exploration on the unit sphere
    # Generate diverse initial directions
    n_initial = 100
    for i in range(n_initial):
        if time.time() > _DEADLINE - 90:
            break
        
        # Random direction on sphere
        v = rng.normal(0, 1, 3)
        c_dir = v / np.linalg.norm(v)
        
        # Find optimal radius for this direction
        r_opt, val_opt = _optimize_radius(c_dir, 0.1, 50.0, 20, rng)
        
        if val_opt < best_bound:
            best_bound = val_opt
            best_params = r_opt * c_dir
    
    # Also try seed direction
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        c_seed = seed / r_seed
        r_opt, val_opt = _optimize_radius(c_seed, 0.1, 50.0, 30, rng)
        if val_opt < best_bound:
            best_bound = val_opt
            best_params = r_opt * c_seed
    
    # Phase 2: Stochastic local search from best direction
    if time.time() < _DEADLINE - 60:
        c_best_dir = best_params / np.linalg.norm(best_params)
        
        # Multiple rounds with different step sizes
        for round_num in range(3):
            if time.time() > _DEADLINE - 30:
                break
            
            initial_steps = [0.5, 0.2, 0.05]
            n_iters_list = [3000, 5000, 10000]
            
            c_curr, val_curr = _stochastic_local_search_on_sphere(
                c_best_dir,
                n_iters=n_iters_list[round_num],
                initial_step=initial_steps[round_num],
                rng=rng
            )
            
            # Find optimal radius for this direction
            r_opt, val_opt = _optimize_radius(c_curr, 0.1, 50.0, 30, rng)
            
            if val_opt < best_bound:
                best_bound = val_opt
                best_params = r_opt * c_curr
                c_best_dir = c_curr.copy()
    
    # Phase 3: Fine-tuning with very small steps
    if time.time() < _DEADLINE - 15:
        c_best_dir = best_params / np.linalg.norm(best_params)
        
        c_curr, val_curr = _stochastic_local_search_on_sphere(
            c_best_dir,
            n_iters=15000,
            initial_step=0.01,
            rng=rng
        )
        
        r_opt, val_opt = _optimize_radius(c_curr, 0.1, 50.0, 50, rng)
        
        if val_opt < best_bound:
            best_bound = val_opt
            best_params = r_opt * c_curr
    
    # Verify the result
    final_bound = _compute_bound_fast(*best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
    
    return best_params
# EVOLVE-BLOCK-END
