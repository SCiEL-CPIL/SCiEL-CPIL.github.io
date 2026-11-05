"""
Dzugutov cluster global minima search.
Uses high-fidelity Mackay icosahedral seeds and parallel basin hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import concurrent.futures

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

# Global time tracker
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

# Pre-computed pair indices cache
_pair_cache = {}

def _get_pair_indices(n):
    if n not in _pair_cache:
        iu = np.triu_indices(n, k=1)
        _pair_cache[n] = iu
    return _pair_cache[n]

def _dzugutov_energy(positions, c_val=None, d_val=None):
    """Compute Dzugutov potential energy."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    if n < 2:
        return 0.0
    iu = _get_pair_indices(n)
    diff = positions[iu[0]] - positions[iu[1]]
    r2_pair = np.sum(diff ** 2, axis=1)
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    r = np.sqrt(r2_pair[valid])
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_val / (r1 - a)))
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_val / (r2_ - b)))
    return float(energy)

def _dzugutov_gradient(positions, c_val=None, d_val=None):
    """Compute gradient."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    iu = _get_pair_indices(n)
    i_idx, j_idx = iu[0], iu[1]
    diff = positions[i_idx] - positions[j_idx]
    r2_pair = np.sum(diff ** 2, axis=1)
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
    i_idx_v, j_idx_v = i_idx[valid], j_idx[valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[valid]
    dV_dr = np.zeros_like(r)
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_val / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_val / (r1 - a) ** 2)
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d_val / (r2_ - b)) * (-d_val / (r2_ - b) ** 2)
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    np.add.at(grad, i_idx_v, forces)
    np.add.at(grad, j_idx_v, -forces)
    return grad

def _relax(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Relax structure using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()
    def fun(x):
        return _dzugutov_energy(x.reshape(n, 3))
    def jac(x):
        return _dzugutov_gradient(x.reshape(n, 3)).flatten()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        return pos, _dzugutov_energy(pos)

def _check_min_dist(pos, min_r=0.5):
    if len(pos) < 2:
        return True
    iu = _get_pair_indices(len(pos))
    diff = pos[iu[0]] - pos[iu[1]]
    r2 = np.sum(diff ** 2, axis=1)
    return np.min(r2) >= min_r * min_r - 1e-9

# --- Geometry Generators ---

def _icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([[0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
                  [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
                  [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]], dtype=float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def _icosahedron_edge_centers():
    v = _icosahedron_vertices()
    # Find edges (shortest distances)
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    edge_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (v[i] + v[j]) / 2.0
                edge_centers.append(mid / np.linalg.norm(mid))
    return np.array(edge_centers)

def _icosahedron_face_centers():
    v = _icosahedron_vertices()
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    adj = {i: [] for i in range(12)}
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                adj[i].append(j)
                adj[j].append(i)
    
    face_centers = []
    seen = set()
    for i in range(12):
        for j in adj[i]:
            if j <= i: continue
            for k in adj[j]:
                if k <= j or k == i: continue
                if k in adj[i]:
                    key = tuple(sorted([i, j, k]))
                    if key not in seen:
                        seen.add(key)
                        c = (v[i] + v[j] + v[k]) / 3.0
                        face_centers.append(c / np.linalg.norm(c))
    return np.array(face_centers)

def _build_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron structure for n atoms.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices)
    Shell 2: 42 atoms (30 edge centers + 12 face centers) -> Total 55
    Shell 3: 92 atoms -> Total 147
    
    For n < 55, we truncate the outer shells.
    For n > 55, we add random atoms on a spherical shell or expand the icosahedron.
    """
    ico_v = _icosahedron_vertices()
    ico_e = _icosahedron_edge_centers()
    ico_f = _icosahedron_face_centers()
    
    # Standard Mackay radii (relative to first shell)
    # r1 ~ 1.0, r2_edge ~ 1.63, r2_face ~ 1.53
    # We will scale these to match the potential's preferred distance (~1.9-2.0 for neighbors)
    # The potential minimum is around r ~ 1.9-2.0.
    # Let's try scaling factors.
    
    # We will generate a few variations with different radial scalings
    seeds = []
    
    # Variation 1: Standard Mackay
    r1 = 1.9
    r2e = 1.9 * 1.63
    r2f = 1.9 * 1.53
    
    pts = [np.zeros(3)]
    pts.extend((ico_v * r1).tolist())
    pts.extend((ico_e * r2e).tolist())
    pts.extend((ico_f * r2f).tolist())
    
    pts = np.array(pts)
    
    # Truncate or Pad
    if len(pts) > n:
        # Remove outermost atoms first (faces are slightly smaller than edges, so remove edges first? 
        # Actually, faces are "inner" to edges in terms of distance from center? 
        # r2f < r2e. So faces are closer to center. We should remove edges first if we need to reduce.
        # But for n=38, 45, 50, we are in the 2nd shell.
        # Shell 1: 13 atoms. Shell 2: 42 atoms. Total 55.
        # If n < 55, we keep center + shell 1 + part of shell 2.
        # Let's sort by radius and keep the closest n? No, that breaks symmetry.
        # Better: Keep Center + Shell 1 (13). If n > 13, add Shell 2 atoms.
        # Shell 2 has 42 atoms. If n < 55, we pick a subset of Shell 2.
        # To maintain symmetry, we should pick from Edge centers and Face centers proportionally?
        # Or just pick the first n-13 from the combined list? 
        # For global minima, symmetry is key. 
        # Let's just take the first n atoms from the list [Center, V, E, F].
        # This might break symmetry for n not equal to 13, 55, 147.
        # However, basin hopping will fix local symmetry if it's a good minimum.
        
        # Let's try a smarter truncation:
        # If n <= 13: return first n (Center + V)
        # If 13 < n <= 55: return Center + V + (n-13) atoms from E+F.
        # To preserve icosahedral symmetry as much as possible, we should take all F (12) first? 
        # Or all E (30)? 
        # Usually, the 55-atom Mackay icosahedron is the stable one.
        # For n=38, 45, 50, it's a partial shell.
        
        pass # Handled below
    
    if len(pts) < n:
        # Add more atoms on a larger shell
        r3 = r2e * 1.5
        # Generate points on a sphere
        extra = n - len(pts)
        # Use random points on sphere, or icosahedron vertices scaled?
        # Let's use random points on sphere for diversity
        z = rng.uniform(-1, 1, extra)
        theta = rng.uniform(0, 2*np.pi, extra)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        new_pts = np.column_stack((x, y, z)) * r3
        pts = np.vstack([pts, new_pts])
        
    # If we have too many, truncate
    if len(pts) > n:
        # Sort by distance from center? 
        # Or just take first n?
        # Taking first n from [C, V, E, F] keeps the core intact.
        pts = pts[:n]
        
    return pts

def _generate_seeds(n, rng):
    """Generate diverse seeds based on Mackay icosahedra."""
    seeds = []
    
    # 1. Mackay Icosahedron variations
    for scale in [0.9, 1.0, 1.1]:
        try:
            base = _build_mackay_icosahedron(n, rng)
            # Scale the whole structure
            seed = base * scale
            if _check_min_dist(seed):
                seeds.append(seed)
        except Exception:
            pass
    
    # 2. Perturbed Mackay
    base = _build_mackay_icosahedron(n, rng)
    for _ in range(5):
        perturbed = base + rng.normal(0, 0.1, (n, 3))
        if _check_min_dist(perturbed):
            seeds.append(perturbed)
            
    # 3. Random Compact
    for _ in range(5):
        pts = rng.normal(0, 1.5, (n, 3))
        if _check_min_dist(pts):
            seeds.append(pts)
            
    # 4. FCC-like (less likely for Dzugutov, but good for diversity)
    # Generate a small FCC block
    a_fcc = 2.0
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                pts.append([i*a_fcc, j*a_fcc, k*a_fcc])
                pts.append([(i+0.5)*a_fcc, (j+0.5)*a_fcc, k*a_fcc])
                pts.append([(i+0.5)*a_fcc, j*a_fcc, (k+0.5)*a_fcc])
                pts.append([i*a_fcc, (j+0.5)*a_fcc, (k+0.5)*a_fcc])
    pts = np.array(pts)
    if len(pts) >= n:
        # Center it
        pts = pts[:n] - np.mean(pts[:n], axis=0)
        if _check_min_dist(pts):
            seeds.append(pts)
            
    return seeds

def _basin_hopping_worker(seed, n, rng, max_steps=50, sigma_start=0.3, sigma_end=0.1):
    """
    Perform basin hopping on a single seed.
    Returns the best position and energy found.
    """
    best_pos = seed.copy()
    best_energy = _dzugutov_energy(seed)
    current_pos = seed.copy()
    
    for step in range(max_steps):
        if _time_left() < 5:
            break
            
        # Anneal sigma
        frac = step / max(max_steps - 1, 1)
        sigma = sigma_start + (sigma_end - sigma_start) * frac
        
        perturbed = current_pos + rng.normal(0, sigma, (n, 3))
        
        # Relax
        relaxed, e_relaxed = _relax(perturbed, max_iter=200, ftol=1e-10, gtol=1e-8)
        
        if not _check_min_dist(relaxed):
            continue
            
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            best_pos = relaxed.copy()
            
        # Metropolis acceptance (simplified: always accept if lower, else accept with prob exp(-dE/T))
        # Since we are minimizing, and we keep the best, we just need to move the current state.
        # If we accept the move, current_pos = relaxed.
        # If not, we might stay or try a different perturbation.
        # Standard BH: if E_new < E_old, accept. Else accept with prob exp(-(E_new-E_old)/T).
        # Here we don't have a T. Let's just always move to the relaxed state to explore.
        # Or, to avoid getting stuck, if it's worse, we still move with some probability.
        # Given the time constraint, moving to the relaxed state is a good heuristic for exploration.
        current_pos = relaxed
        
    return best_pos, best_energy

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    
    if _time_left() < 5:
        return np.random.randn(n, 3) * 0.5
        
    rng = np.random.default_rng(42)
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    if not seeds:
        return np.random.randn(n, 3) * 0.5
        
    # Parallel Basin Hopping
    # Use 4 threads
    best_pos = None
    best_energy = float('inf')
    
    # Split seeds among workers
    n_seeds = len(seeds)
    n_threads = 4
    
    # If we have fewer seeds than threads, just use all
    if n_seeds < n_threads:
        n_threads = n_seeds
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            # Create a unique RNG for each worker to avoid correlation
            worker_rng = np.random.default_rng(42 + i)
            futures.append(executor.submit(_basin_hopping_worker, seed, n, worker_rng))
            
        for future in concurrent.futures.as_completed(futures):
            if _time_left() < 10:
                # Cancel remaining? Can't easily cancel, but we can stop processing results
                break
            try:
                pos, energy = future.result()
                if _check_min_dist(pos) and energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
            except Exception:
                continue
                
    if best_pos is None:
        # Fallback: relax the first seed
        best_pos, best_energy = _relax(seeds[0])
        
    # Final Polish
    if _time_left() > 5:
        polished, e_pol = _relax(best_pos, max_iter=1000, ftol=1e-14, gtol=1e-12)
        if _check_min_dist(polished) and e_pol < best_energy:
            best_pos = polished
            best_energy = e_pol
            
    return best_pos

# EVOLVE-BLOCK-END
