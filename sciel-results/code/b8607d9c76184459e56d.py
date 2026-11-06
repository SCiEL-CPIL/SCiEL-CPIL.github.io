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
        # Contract: six calls together < 300 s.
        # Safety margin before 300s kill.
        _HARD_DEADLINE = _T0 + 295.0
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

def _local_minimize(P0, rho, max_iter=200):
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
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
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
            # Just return the first n points (center + some vertices)
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
    # 6 calls, 300s total. ~50s per call.
    # We use a global hard deadline to be safe.
    call_deadline = start_time + 45.0
    effective_deadline = min(call_deadline, _HARD_DEADLINE)
    
    best_E = np.inf
    best_P = None
    
    # Phase 1: Initial Seeds + Local Minimization
    seeds = _generate_seeds(n, rng)
    
    for seed in seeds:
        if time.time() > effective_deadline - 5.0: # Leave time for BH
            break
        E, P = _local_minimize(seed, rho)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    if best_P is None:
        best_P = _generate_random(n, rng)
        best_E = _morse_energy(best_P, rho)
        
    # Phase 2: Basin Hopping from the best seed
    # We allocate the remaining time to basin hopping.
    time_remaining = effective_deadline - time.time()
    if time_remaining > 2.0:
        # Run basin hopping from the best structure found so far
        # We can also run it from other good seeds if time permits, but let's focus on the best.
        E_bh, P_bh = _basin_hopping(best_P, rho, rng, time_budget=time_remaining - 2.0)
        if E_bh < best_E:
            best_E = E_bh
            best_P = P_bh
            
        # If time remains, try a second BH run from a different seed (e.g. the second best)
        # But we didn't track second best. Let's just do one more BH run with a larger step size?
        # Or maybe a different seed.
        if time.time() < effective_deadline - 2.0:
             # Pick a random seed from the initial set (excluding the one we used for best_P if possible, or just random)
             alt_seed = _generate_fcc(n, rng, jitter=0.5)
             E_bh2, P_bh2 = _basin_hopping(alt_seed, rho, rng, time_budget=effective_deadline - time.time() - 2.0)
             if E_bh2 < best_E:
                 best_E = E_bh2
                 best_P = P_bh2

    # Final Polish
    if time.time() < _HARD_DEADLINE - 0.5 and best_P is not None:
        E, P = _local_minimize(best_P, rho, max_iter=500)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    # Enforce minimum distance
    best_P = _enforce_min_dist(best_P, 0.5)
    
    return best_P
