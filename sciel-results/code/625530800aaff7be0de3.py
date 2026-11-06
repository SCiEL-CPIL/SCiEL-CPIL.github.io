import time
import numpy as np
from scipy.optimize import minimize

# Global timer and deadline management
_T0 = None
_HARD_DEADLINE = None

def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0

def _morse_energy(P, rho):
    """
    Compute exact Morse energy.
    E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0
    
    # Calculate all pair distances
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(r, np.inf)
    
    # Calculate Morse potential for each pair
    t = np.exp(rho * (1.0 - r))
    E_pair = t * (t - 2.0)
    
    # Sum over upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    return float(np.sum(E_pair[iu, ju]))

def _morse_gradient(P, rho):
    """
    Compute gradient of Morse energy.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return np.zeros_like(P)
        
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    
    t = np.exp(rho * (1.0 - r))
    # dE/dr = rho * t * (2*t - 2)
    dEdr = rho * t * (2.0 * t - 2.0)
    
    # inv_r is 1/r, 0 on diagonal
    inv_r = np.zeros_like(r)
    mask = r < np.inf
    inv_r[mask] = 1.0 / r[mask]
    
    # W_ij = - (dEdr_ij) / r_ij
    W = -dEdr * inv_r
    
    # grad_i = sum_j W_ij * (P_i - P_j)
    row_sum = np.sum(W, axis=1)
    grad = row_sum[:, np.newaxis] * P - W @ P
    
    return grad

def _enforce_min_dist(P, min_d=0.5):
    """
    Nudge atoms to ensure minimum distance.
    """
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    if n < 2:
        return P
        
    for _ in range(100):
        d = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        bad_pairs = np.argwhere(d < min_d)
        if len(bad_pairs) == 0:
            break
            
        for i, j in bad_pairs:
            if i == j: continue
            gap = min_d - d[i, j] + 0.01
            v = P[i] - P[j]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                # Random direction
                v = np.random.normal(size=3)
                norm_v = np.linalg.norm(v)
            v = v / norm_v
            P[i] += 0.5 * gap * v
            P[j] -= 0.5 * gap * v
    return P

def _local_minimize(P0, rho, max_iter=200, tol=1e-12):
    """
    Local minimization using L-BFGS-B.
    """
    n = P0.shape[0]
    
    def fun(x):
        P = x.reshape(n, 3)
        return _morse_energy(P, rho)
        
    def jac(x):
        P = x.reshape(n, 3)
        return _morse_gradient(P, rho).ravel()
        
    try:
        res = minimize(
            fun, P0.ravel(), 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol*100}
        )
        P_opt = res.x.reshape(n, 3)
        E_opt = _morse_energy(P_opt, rho)
        return E_opt, P_opt
    except Exception:
        return None, None

def _generate_icosahedron(n, rng):
    """
    Generate an icosahedral-like structure.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 13:
        P = np.vstack([np.zeros(3), verts[:n-1]])
        if n < 13:
            return P[:n]
        return P
        
    P = [np.zeros(3)]
    P.extend(list(verts)) # 13 atoms
    
    remaining = n - 13
    if remaining > 0:
        # Add atoms in a "fuzzy" second shell
        for _ in range(remaining):
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            r = 1.5 + rng.random() * 0.5
            pos = dir_vec * r + rng.normal(0, 0.2, 3)
            P.append(pos)
            
    return np.array(P)

def _generate_fcc(n, rng, jitter=0.1):
    """
    Generate an FCC cluster fragment.
    """
    a = 1.0
    R = (3.0 * n * a**3 / (16.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    pts = []
    max_idx = int(np.ceil(R / a)) + 2
    
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i*a, j*a, k*a])
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, k*a])
                p3 = np.array([(i+0.5)*a, j*a, (k+0.5)*a])
                p4 = np.array([i*a, (j+0.5)*a, (k+0.5)*a])
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
                        
    if len(pts) > n:
        pts_arr = np.array(pts)
        dists = np.linalg.norm(pts_arr, axis=1)
        idxs = np.argsort(dists)[:n]
        pts = pts_arr[idxs]
    elif len(pts) < n:
        for _ in range(n - len(pts)):
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                pts.append(p)
            else:
                pts.append(np.zeros(3))
                
    pts_arr = np.array(pts)
    pts_arr += rng.normal(0, jitter, pts_arr.shape)
    return pts_arr

def _generate_random(n, rng):
    """
    Generate a random spherical cluster.
    """
    R = (n / (4.0/3.0 * np.pi)) ** (1/3.0) * 1.5
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                pts.append(p * R)
                break
    return np.array(pts)

def _generate_seeds(n, rng):
    """
    Generate a diverse set of seeds.
    """
    seeds = []
    
    # 1. Icosahedral seed
    seeds.append(_generate_icosahedron(n, rng))
    
    # 2. FCC seed
    seeds.append(_generate_fcc(n, rng, jitter=0.05))
    seeds.append(_generate_fcc(n, rng, jitter=0.2))
    
    # 3. Random spherical
    seeds.append(_generate_random(n, rng))
    
    # 4. Distorted Icosahedron
    if n >= 13:
        ico = _generate_icosahedron(n, rng)
        ico += rng.normal(0, 0.3, ico.shape)
        seeds.append(ico)
        
    return seeds

def _langevin_step(P, grad, mass, dt, kT, rng):
    """
    Simple Langevin dynamics step.
    P_new = P + v*dt
    v_new = v*(1-gamma*dt) - (grad/mass)*dt + sqrt(2*gamma*kT*dt/mass)*noise
    Here we do a single integrated step for simplicity.
    """
    # v_current is not tracked explicitly in this simple form, we use implicit integration or small dt
    # Let's use: P += v*dt, v += (-grad/mass - gamma*v + noise)*dt
    # For simplicity in this context, let's just do a gradient step with noise
    # P_new = P - lr * grad + noise
    # This is effectively Langevin with high friction
    return P

def _langevin_anneal(P0, rho, rng, time_budget):
    """
    Simulated Annealing using Langevin dynamics.
    """
    t0 = time.time()
    n = P0.shape[0]
    P = P0.copy()
    v = np.zeros_like(P)
    
    # Parameters
    T0 = 1.0
    T_final = 0.001
    # Linear cooling schedule
    # Steps are estimated by time
    dt = 0.01
    gamma = 1.0 # Friction
    mass = 1.0
    
    # Initial step size
    # Estimate max force
    grad = _morse_gradient(P, rho)
    max_force = np.max(np.linalg.norm(grad, axis=1))
    if max_force < 1e-8:
        max_force = 1.0
    step_size = 0.1 / max_force
    
    while time.time() - t0 < time_budget:
        # Determine current temperature based on time fraction
        elapsed = time.time() - t0
        # Assume a target number of steps? No, time based.
        # Let's do a fixed number of "epochs"
        # Or just linear T decrease over time_budget
        frac = min(1.0, elapsed / time_budget)
        kT = T0 * (T_final / T0) ** frac
        
        # Noise amplitude: sqrt(2 * gamma * kT * dt / mass)
        noise_amp = np.sqrt(2 * gamma * kT * dt / mass)
        
        grad = _morse_gradient(P, rho)
        
        # Update velocity
        # v = (1 - gamma*dt) * v - (dt/mass) * grad + noise
        v *= (1 - gamma * dt)
        v -= (dt / mass) * grad
        v += rng.normal(0, noise_amp, size=P.shape)
        
        # Update position
        P += dt * v
        
        # Clamp velocity to prevent explosions
        v_norms = np.linalg.norm(v, axis=1)
        max_v = 0.1
        v[v_norms > max_v] *= (max_v / v_norms[v_norms > max_v, np.newaxis])
        
        # Occasionally enforce min distance?
        # It might disrupt the flow. Let's see.
        # If atoms get too close, the repulsion should push them apart.
        # But if dt is too large, they might jump over the repulsion core.
        # The Morse potential has a singularity at r=0 (in the derivative).
        # Let's add a check.
        d = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.1:
            # Too close, reduce step size or enforce
            # Enforce gently
            P = _enforce_min_dist(P, 0.1)
            v *= 0.1 # Damp velocity
            
    return P

def _basin_hopping(start_pos, rho, rng, time_budget, max_iter=50):
    """
    Basin hopping with adaptive step size.
    Returns (best_energy, best_pos).
    """
    t0 = time.time()
    best_E, best_P = _local_minimize(start_pos, rho)
    if best_E is None:
        best_E = _morse_energy(start_pos, rho)
        best_P = start_pos.copy()
        
    current_E, current_P = best_E, best_P.copy()
    step_size = 0.5
    T = 0.1 # Temperature for Metropolis acceptance
    
    for i in range(max_iter):
        if time.time() - t0 > time_budget:
            break
            
        # Perturb
        noise = rng.normal(0, step_size, current_P.shape)
        perturbed = current_P + noise
        
        # Relax
        E_new, P_new = _local_minimize(perturbed, rho)
        
        if E_new is None:
            continue
            
        delta_E = E_new - current_E
        
        # Metropolis criterion
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_E, current_P = E_new, P_new.copy()
            step_size *= 1.05 # Increase step if accepted
        else:
            step_size *= 0.95 # Decrease step if rejected
            
        # Update best
        if current_E < best_E:
            best_E, best_P = current_E, current_P.copy()
            
        # Adapt temperature/step
        if i % 20 == 0:
            T = max(0.01, T * 0.8)
            step_size = min(1.0, max(0.1, step_size))

    return best_E, best_P

def construct(instance):
    global _T0, _HARD_DEADLINE
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", 3.0)
    
    rng = np.random.default_rng(instance.get("seed", 42) + n)
    
    start_time = time.time()
    
    # Time budget management
    call_deadline = start_time + 40.0
    effective_deadline = min(call_deadline, _HARD_DEADLINE)
    
    best_E = np.inf
    best_P = None
    
    # Phase 1: Initial Seeds + Local Minimization
    seeds = _generate_seeds(n, rng)
    
    for seed in seeds:
        if time.time() > effective_deadline - 5.0:
            break
        E, P = _local_minimize(seed, rho)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    if best_P is None:
        best_P = _generate_random(n, rng)
        best_E = _morse_energy(best_P, rho)
        
    # Phase 2: Basin Hopping and Annealing
    time_remaining = effective_deadline - time.time()
    
    if time_remaining > 2.0:
        # Split time between Basin Hopping and Langevin Annealing
        # BH is usually more effective for discrete landscapes, but Annealing might explore more.
        # Let's try 60% BH, 40% Annealing
        
        t_bh = time_remaining * 0.6 - 1.0
        t_ann = time_remaining * 0.4 - 1.0
        
        # Run BH
        E_bh, P_bh = _basin_hopping(best_P, rho, rng, time_budget=max(0.1, t_bh))
        if E_bh < best_E:
            best_E = E_bh
            best_P = P_bh
        
        # Run Annealing from best so far
        if t_ann > 0.5 and time.time() < effective_deadline - 1.0:
             P_ann = _langevin_anneal(best_P, rho, rng, time_budget=t_ann)
             # Relax after annealing
             E_ann, P_ann_rel = _local_minimize(P_ann, rho)
             if E_ann is not None and E_ann < best_E:
                 best_E = E_ann
                 best_P = P_ann_rel
        
        # If time remains, try a second BH run from a different seed
        if time.time() < effective_deadline - 2.0:
             # Pick a random seed from the initial set
             alt_seed = _generate_fcc(n, rng, jitter=0.5)
             t_second = effective_deadline - time.time() - 1.0
             E_bh2, P_bh2 = _basin_hopping(alt_seed, rho, rng, time_budget=t_second)
             if E_bh2 < best_E:
                 best_E = E_bh2
                 best_P = P_bh2

    # Final Polish
    if time.time() < _HARD_DEADLINE - 0.5 and best_P is not None:
        E, P = _local_minimize(best_P, rho, max_iter=1000, tol=1e-14)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    # Enforce minimum distance
    best_P = _enforce_min_dist(best_P, 0.5)
    
    return best_P
