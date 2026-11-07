"""New program: Simulated Annealing with diverse initialization and local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

def _get_kind_params(kind):
    """Map kind string to (n_exp, m_exp, c)."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        # Fallback to 12-6
        return 12, 6, 144.41

def _compute_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(positions)
    if n == 0:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero
    inv_r_n = np.power(dists, -n_exp)
    inv_r_m = np.power(dists, -m_exp)
    
    # Sum over j for each i
    sum_inv_r_n = np.sum(inv_r_n, axis=1)
    sum_inv_r_m = np.sum(inv_r_m, axis=1)
    
    # Energy term 1: 0.5 * sum_i sum_{j!=i} r^-n
    e_rep = 0.5 * np.sum(sum_inv_r_n)
    
    # Energy term 2: -c * sum_i sqrt(sum_{j!=i} r^-m)
    # Note: sum_inv_r_m already excludes self (diagonal was set to 1, but we need to subtract 1 for the self term if included in sum? 
    # Wait, dists diagonal is 1.0, so inv_r_m diagonal is 1.0.
    # The formula is sum_{j!=i}. So we must subtract the self term (1.0) from sum_inv_r_m.
    sum_inv_r_m_no_self = sum_inv_r_m - 1.0
    
    # Handle potential negative values due to numerical issues or very close atoms? 
    # sqrt of negative is NaN. Clip to 0.
    sum_inv_r_m_no_self = np.maximum(sum_inv_r_m_no_self, 0.0)
    
    e_att = -c * np.sum(np.sqrt(sum_inv_r_m_no_self))
    
    return e_rep + e_att

def _sa_optimize(positions, n_exp, m_exp, c, T0=1.0, Tend=0.01, steps=5000, alpha=0.999, seed=42, min_dist=0.35):
    """
    Vectorized Simulated Annealing.
    Returns (best_energy, best_positions).
    """
    n = len(positions)
    rng = np.random.default_rng(seed)
    
    # Initial energy
    E = _compute_energy(positions, n_exp, m_exp, c)
    if not np.isfinite(E):
        return np.inf, positions
    
    best_E = E
    best_P = positions.copy()
    
    # Pre-compute step size based on system size
    step_size = 0.1
    
    for i in range(steps):
        T = T0 * (alpha ** i)
        if T < Tend:
            break
            
        # Generate random move: pick one atom and displace it
        idx = rng.integers(0, n)
        displacement = rng.normal(0, step_size, 3)
        
        old_pos = positions[idx].copy()
        positions[idx] += displacement
        
        # Check min distance constraint quickly? 
        # Full check is O(N^2). For N~80, 6400 pairs. 5000 steps * 6400 = 32M ops. Might be slow in Python loop.
        # Let's do a quick check: if new position is too close to any other, reject immediately.
        dists_to_others = np.linalg.norm(positions - positions[idx], axis=1)
        dists_to_others[idx] = 1.0 # Ignore self
        if np.any(dists_to_others < min_dist):
            positions[idx] = old_pos
            continue
            
        new_E = _compute_energy(positions, n_exp, m_exp, c)
        
        if not np.isfinite(new_E):
            positions[idx] = old_pos
            continue
            
        delta_E = new_E - E
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            E = new_E
            if E < best_E:
                best_E = E
                best_P = positions.copy()
        else:
            positions[idx] = old_pos
            
        # Adaptive step size: reduce if too many rejections? 
        # Simple decay
        step_size *= 0.9999
        
    return best_E, best_P

def _polish(positions, potential, kind):
    """Local optimization using L-BFGS-B."""
    from autoresearch_suite.chem_clusters import relax
    try:
        e, p = relax(positions, potential, kind)
        if np.isfinite(e):
            return e, p
    except Exception:
        pass
    return np.inf, positions

def _fcc_fragment(n, spacing=1.0):
    """Generate n points from an FCC lattice."""
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    fcc_pts = []
    for n1 in range(-max_coord, max_coord + 1):
        for n2 in range(-max_coord, max_coord + 1):
            for n3 in range(-max_coord, max_coord + 1):
                x = (n1 + n2) * 0.5
                y = (n2 + n3) * 0.5
                z = (n3 + n1) * 0.5
                fcc_pts.append((x, y, z))
    fcc_pts = np.array(fcc_pts) * spacing
    fcc_pts -= np.mean(fcc_pts, axis=0)
    radii = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(radii)
    return fcc_pts[idx[:n]]

def _random_compact(n, spacing=0.8, seed=42):
    """Random points with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0)
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: relax min distance
        min_dist = 0.35
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 2000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _snowball_growth(n, spacing=0.8, seed=42):
    """Grow cluster by adding atoms one by one at lowest energy position."""
    rng = np.random.default_rng(seed)
    n_exp, m_exp, c = _get_kind_params("12-6") # Use 12-6 for growth heuristic? Or pass kind?
    # Let's just use random placement for simplicity in this function, 
    # but place at random surface positions.
    
    # Start with 1 atom
    positions = [np.zeros(3)]
    radius = spacing * 0.5
    
    for i in range(1, n):
        # Try to place new atom
        best_pos = None
        best_e = np.inf
        for _ in range(50): # 50 candidates
            # Random direction
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            # Radius increases slightly
            r = spacing * (i ** (1/3))
            pos = u * r + rng.normal(0, 0.1, 3)
            
            # Check min dist
            if all(np.linalg.norm(pos - q) >= 0.35 for q in positions):
                # Compute energy contribution roughly? 
                # Just take first valid one for speed, or compute full energy
                # Full energy is O(N^2). 50 * N^2. For N=80, 50*6400=320k ops. Acceptable.
                new_positions = np.array(positions + [pos])
                e = _compute_energy(new_positions, 12, 6, 144.41)
                if e < best_e:
                    best_e = e
                    best_pos = pos
                    
        if best_pos is None:
            # Fallback: random
            best_pos = rng.uniform(-spacing, spacing, 3)
            
        positions.append(best_pos)
        
    return np.array(positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    n_exp, m_exp, c = _get_kind_params(kind)
    
    # Time management
    start_time = time.time()
    time_limit = 250.0 # Leave buffer for overhead
    
    # Generate diverse starts
    starts = []
    
    # 1. Random compact (multiple seeds)
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        try:
            rnd = _random_compact(size, spacing=0.8, seed=seed)
            starts.append(rnd)
        except Exception:
            pass
            
    # 2. FCC fragment
    try:
        fcc = _fcc_fragment(size, spacing=0.8)
        starts.append(fcc)
    except Exception:
        pass
        
    # 3. Snowball growth
    try:
        snow = _snowball_growth(size, spacing=0.8, seed=42)
        starts.append(snow)
    except Exception:
        pass
        
    # 4. Icosahedral (simple)
    try:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        # Scale to fit size
        scale = 0.8 * size ** (1/3)
        ico_pts = verts * scale
        # Add more points by scaling and rotating? 
        # Just use the 12 vertices and pad with random if needed? 
        # Better: generate more icosahedral points
        all_ico = []
        for r_scale in [0.5, 0.8, 1.0]:
            all_ico.extend(verts * (scale * r_scale))
        all_ico = np.array(all_ico)
        # Center
        all_ico -= np.mean(all_ico, axis=0)
        radii = np.linalg.norm(all_ico, axis=1)
        idx = np.argsort(radii)
        ico_frag = all_ico[idx[:size]]
        starts.append(ico_frag)
    except Exception:
        pass
        
    best_E = np.inf
    best_P = None
    
    # Optimize each start with SA + Polish
    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        # SA
        try:
            sa_E, sa_P = _sa_optimize(start, n_exp, m_exp, c, 
                                      T0=1.0, Tend=0.01, 
                                      steps=2000, alpha=0.9995, 
                                      seed=100+i)
            
            # Polish
            pol_E, pol_P = _polish(sa_P, potential, kind)
            
            if np.isfinite(pol_E) and pol_E < best_E:
                best_E = pol_E
                best_P = pol_P
        except Exception:
            continue
            
    # If we still have time, do more SA runs on the best solution found
    if best_P is not None and time.time() - start_time < time_limit - 20:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            # Perturb best solution
            perturbed = best_P + np.random.normal(0, 0.1, best_P.shape)
            try:
                sa_E, sa_P = _sa_optimize(perturbed, n_exp, m_exp, c, 
                                          T0=0.5, Tend=0.01, 
                                          steps=1000, alpha=0.999, 
                                          seed=1000+_)
                pol_E, pol_P = _polish(sa_P, potential, kind)
                if np.isfinite(pol_E) and pol_E < best_E:
                    best_E = pol_E
                    best_P = pol_P
            except Exception:
                pass

    # Fallback if no valid solution
    if best_P is None:
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _polish(best_P, potential, kind)
        
    # Final validation and fix
    if not np.all(np.isfinite(best_P)):
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _polish(best_P, potential, kind)
        
    # Check min distance
    min_d = np.inf
    for i, j in combinations(range(len(best_P)), 2):
        d = np.linalg.norm(best_P[i] - best_P[j])
        if d < min_d:
            min_d = d
            
    if min_d < 0.35:
        # Attempt to fix by relaxing with a penalty? 
        # Or just perturb and re-polish
        for _ in range(10):
            perturbed = best_P + np.random.normal(0, 0.2, best_P.shape)
            try:
                e, p = _polish(perturbed, potential, kind)
                md = np.inf
                for i, j in combinations(range(len(p)), 2):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < md:
                        md = d
                if md >= 0.35 and np.isfinite(e):
                    best_E, best_P = e, p
                    break
            except Exception:
                continue
                
    return best_P
# EVOLVE-BLOCK-END
