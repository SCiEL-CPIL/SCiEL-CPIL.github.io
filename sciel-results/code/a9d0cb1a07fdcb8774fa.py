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


def _bound_from_angles(theta1, theta2):
    """
    Compute bound for unit vector on sphere defined by spherical angles.
    a0 = sin(theta2) * cos(theta1)
    a4 = sin(theta2) * sin(theta1)
    a8 = cos(theta2)
    """
    st2 = np.sin(theta2)
    a0 = st2 * np.cos(theta1)
    a4 = st2 * np.sin(theta1)
    a8 = np.cos(theta2)
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Exploits scale invariance by searching only on the unit sphere S^2.
    """
    # Seed values for reference (normalized)
    seed = np.array([-1.13, 0.04, 0.000308642])
    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_dir = seed / norm_seed
    else:
        seed_dir = np.array([1.0, 0.0, 0.0])
    
    # Convert seed direction to angles
    r_seed = 1.0
    theta1_seed = np.arctan2(seed_dir[1], seed_dir[0])
    theta2_seed = np.arccos(np.clip(seed_dir[2], -1.0, 1.0))
    
    best_bound = _bound_from_angles(theta1_seed, theta2_seed)
    best_t1 = theta1_seed
    best_t2 = theta2_seed

    rng = np.random.default_rng(42)

    # Phase 1: Global random search on the sphere
    # Sample many random points
    N_global = 500
    for _ in range(N_global):
        if time.time() > _DEADLINE - 60:
            break
        # Uniform distribution on sphere
        z = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z*z)
        # Map to theta1, theta2
        # x = r * cos(phi), y = r * sin(phi), z = z
        # theta1 = atan2(y, x) = phi
        # theta2 = acos(z)
        t1 = phi
        t2 = np.arccos(z)
        
        b = _bound_from_angles(t1, t2)
        if b < best_bound:
            best_bound = b
            best_t1 = t1
            best_t2 = t2

    # Phase 2: Local search (Simulated Annealing / Random Walk) on the sphere
    # We optimize t1, t2
    state = np.array([best_t1, best_t2])
    best_state = state.copy()
    best_val = best_bound
    
    # Adaptive step sizes
    steps = np.array([0.05, 0.05])
    temperature = 1.0
    cooling = 0.999
    
    max_iters = 20000
    accept_count = 0
    total_count = 0
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Perturb
        delta = rng.normal(0, 1, 2) * steps
        new_state = state + delta
        
        # Wrap t1
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        # Clamp t2 to avoid poles where coordinate system degenerates
        new_state[1] = np.clip(new_state[1], 1e-6, np.pi - 1e-6)
        
        new_val = _bound_from_angles(new_state[0], new_state[1])
        
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
            # Accept with probability
            if delta_val < 1e-12 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.02
            else:
                steps *= 1.05
                
        # Adaptive step size adjustment
        if total_count % 100 == 0:
            ratio = accept_count / total_count
            if ratio > 0.6:
                steps *= 1.1
            elif ratio < 0.3:
                steps *= 0.8
            accept_count = 0
            total_count = 0
            
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

    # Phase 3: Fine tuning
    steps = np.array([0.005, 0.005])
    state = best_state.copy()
    temperature = 1e-3
    cooling = 0.999
    
    for _ in range(10000):
        if time.time() > _DEADLINE - 5:
            break
            
        delta = rng.normal(0, 1, 2) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 1e-8, np.pi - 1e-8)
        
        new_val = _bound_from_angles(new_state[0], new_state[1])
        if not np.isfinite(new_val):
            new_val = 1e18
            
        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.97
        else:
            if new_val - best_val < 1e-14 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                steps *= 1.01
            else:
                steps *= 1.03
                
        temperature *= cooling
        if temperature < 1e-12:
            temperature = 1e-12

    # Convert best state back to (a0, a4, a8)
    # We return the unit vector, which is a valid set of coefficients.
    # The scale doesn't matter, but we can return the unit vector.
    st2 = np.sin(best_state[1])
    result = np.array([
        st2 * np.cos(best_state[0]),
        st2 * np.sin(best_state[0]),
        np.cos(best_state[1])
    ])
    
    # Verify
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed direction
        return seed_dir.copy()
        
    return result
# EVOLVE-BLOCK-END
