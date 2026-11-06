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
        # We use 290s to be safe against overhead.
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
        
    for _ in range(50):
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

def _generate_mackay(n, rng):
    """
    Generate Mackay icosahedral clusters (1, 13, 55, 147, 309...)
    For non-Mackay numbers, we take the largest Mackay subset and add random atoms.
    """
    # Mackay numbers: 1, 13, 55, 147, 309
    mackay_sizes = [1, 13, 55, 147, 309]
    
    # Find largest Mackay size <= n
    base_size = 1
    for size in mackay_sizes:
        if size <= n:
            base_size = size
        else:
            break
            
    # Generate the base Mackay cluster
    if base_size == 1:
        P = np.array([[0.0, 0.0, 0.0]])
    elif base_size == 13:
        P = _generate_icosahedron(13, rng)
    elif base_size == 55:
        # 55 is 1 + 12 + 42. 
        # We approximate by taking icosahedron and adding a second shell
        P = _generate_icosahedron(13, rng)
        # Add 42 atoms in second shell
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        for _ in range(42):
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            r = 1.9 + rng.random() * 0.2 # Approx radius for 2nd shell
            pos = dir_vec * r + rng.normal(0, 0.1, 3)
            P = np.vstack([P, pos])
    elif base_size == 147:
        P = _generate_mackay(55, rng)
        # Add 92 atoms in 3rd shell
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        for _ in range(92):
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            r = 2.8 + rng.random() * 0.3
            pos = dir_vec * r + rng.normal(0, 0.1, 3)
            P = np.vstack([P, pos])
    elif base_size == 309:
        P = _generate_mackay(147, rng)
        # Add 162 atoms in 4th shell
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        for _ in range(162):
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            r = 3.8 + rng.random() * 0.4
            pos = dir_vec * r + rng.normal(0, 0.1, 3)
            P = np.vstack([P, pos])

    # If n > base_size, add random atoms near the surface
    if n > base_size:
        remaining = n - base_size
        for _ in range(remaining):
            # Pick a random existing atom and place new one nearby
            idx = rng.integers(0, base_size)
            pos = P[idx] + rng.normal(0, 0.5, 3)
            P = np.vstack([P, pos])
            
    return P[:n]

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
    
    # 1. Mackay Icosahedral (Best for LJ/Morse clusters)
    seeds.append(_generate_mackay(n, rng))
    
    # 2. FCC
    seeds.append(_generate_fcc(n, rng, jitter=0.05))
    seeds.append(_generate_fcc(n, rng, jitter=0.2))
    
    # 3. Random Spherical
    seeds.append(_generate_random(n, rng))
    
    # 4. Distorted Icosahedron
    if n >= 13:
        ico = _generate_icosahedron(n, rng)
        ico += rng.normal(0, 0.3, ico.shape)
        seeds.append(ico)
        
    return seeds

def _basin_hopping(start_pos, rho, rng, time_budget, max_iter=1000):
    """
    Basin Hopping with adaptive step size and temperature.
    Returns (best_energy, best_pos).
    """
    t0 = time.time()
    best_E, best_P = _local_minimize(start_pos, rho)
    if best_E is None:
        best_E = _morse_energy(start_pos, rho)
        best_P = start_pos.copy()
        
    current_E, current_P = best_E, best_P.copy()
    step_size = 0.5
    T = 0.5 # Initial Temperature
    
    # Track acceptance for adaptive step size
    accepted = 0
    total = 0
    
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
        total += 1
        
        # Metropolis criterion
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_E, current_P = E_new, P_new.copy()
            accepted += 1
            
            # Update best
            if current_E < best_E:
                best_E, best_P = current_E, current_P.copy()
        else:
            pass
            
        # Adapt step size and temperature based on acceptance rate
        if total > 20:
            acc_rate = accepted / total
            # Target acceptance rate ~0.5
            if acc_rate > 0.7:
                step_size *= 1.1
            elif acc_rate < 0.3:
                step_size *= 0.9
                
            # Cool down temperature
            T *= 0.995
            T = max(0.01, T)
            
            # Reset counters
            accepted = 0
            total = 0
            
        # Clamp step size
        step_size = np.clip(step_size, 0.1, 2.0)

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
    
    # Phase 1: Generate diverse seeds and locally minimize each
    seeds = _generate_seeds(n, rng)
    minimized_seeds = []
    
    for seed in seeds:
        if time.time() > effective_deadline - 5.0: # Leave time for BH
            break
        E, P = _local_minimize(seed, rho)
        if E is not None:
            minimized_seeds.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P.copy()
                
    if best_P is None:
        best_P = _generate_random(n, rng)
        best_E = _morse_energy(best_P, rho)
        minimized_seeds.append((best_E, best_P))
        
    # Phase 2: Run Basin Hopping from the best few seeds
    # Sort seeds by energy
    minimized_seeds.sort(key=lambda x: x[0])
    
    # We will run BH on the top 3 seeds if time permits
    time_remaining = effective_deadline - time.time()
    
    if time_remaining > 2.0:
        # Split time among top seeds
        num_bh_runs = min(3, len(minimized_seeds))
        time_per_bh = (time_remaining - 2.0) / num_bh
        
        for i in range(num_bh_runs):
            if time.time() > effective_deadline - 2.0:
                break
                
            seed_E, seed_P = minimized_seeds[i]
            # Run BH from this seed
            E_bh, P_bh = _basin_hopping(seed_P, rho, rng, time_budget=time_per_bh)
            
            if E_bh is not None and E_bh < best_E:
                best_E = E_bh
                best_P = P_bh.copy()
                
    # Final Polish
    if time.time() < _HARD_DEADLINE - 0.5 and best_P is not None:
        E, P = _local_minimize(best_P, rho, max_iter=500)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    # Enforce minimum distance
    best_P = _enforce_min_dist(best_P, 0.5)
    
    return best_P
