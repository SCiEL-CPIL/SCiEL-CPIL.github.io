"""Morse cluster global minima finder - Basin Hopping on Symmetric Templates.

Strategy:
1. Identify the specific instance (n, rho).
2. Generate initial configurations based on known structural motifs for Morse potentials:
   - High rho (short range): Icosahedral shells (Mackay icosahedron).
   - Low rho (long range): FCC/HCP spherical packings.
3. Use Basin Hopping with L-BFGS-B local relaxation to find the global minimum.
   Basin Hopping is robust for escaping local minima and is efficient for these cluster sizes.
4. Strict time management to ensure completion within 300s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration. Vectorized."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    # Only sum over upper triangle
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_energy_gradient(pos, rho):
    """Compute gradient of Morse energy. Vectorized."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # dE/dr = -2 * exp(2*rho*(1-r))
    # Force magnitude F = -dE/dr = 2 * exp(2*rho*(1-r))
    # But we need gradient of E.
    # dE/dpos_i = sum_j (dE/dr_ij) * (r_ij / |r_ij|) * (pos_i - pos_j) / |r_ij| ? No.
    # Let r_ij = pos_i - pos_j.
    # dE/dr = -2 * exp(2*rho*(1-r))
    # dE/dpos_i = sum_j (dE/dr_ij) * (pos_i - pos_j) / r_ij
    # Note: d(r_ij)/d(pos_i) = (pos_i - pos_j)/r_ij
    
    pair_force = -2.0 * np.exp(2.0 * rho * (1.0 - dists))
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    
    # grad[i] = sum_j W[i,j] * (pos[i] - pos[j])
    #         = (sum_j W[i,j]) * pos[i] - sum_j W[i,j] * pos[j]
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
    """Separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        # Find closest pair
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist >= min_d:
            break
        min_pair = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos

def _fcc_sphere(n, spacing=1.0):
    """Generate FCC lattice points within a sphere of radius R."""
    # Estimate radius
    R = spacing * (n ** (1/3)) * 0.8
    max_r = int(np.ceil(R / spacing)) + 2
    pts = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        # If not enough points, expand or use random fill
        # For simplicity, just take what we have and pad with random
        while len(pts) < n:
            pts = np.vstack([pts, np.random.normal(0, R, 3)])
    else:
        # Keep n closest to origin
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    return pts

def _hcp_sphere(n, spacing=1.0):
    """Generate HCP lattice points within a sphere of radius R."""
    R = spacing * (n ** (1/3)) * 0.8
    max_r = int(np.ceil(R / spacing)) + 2
    c_over_a = np.sqrt(8.0 / 3.0)
    pts = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * spacing
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * spacing
                if np.linalg.norm(p1) <= R:
                    pts.append(p1)
                if np.linalg.norm(p2) <= R:
                    pts.append(p2)
    pts = np.array(pts)
    if len(pts) < n:
        while len(pts) < n:
            pts = np.vstack([pts, np.random.normal(0, R, 3)])
    else:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    return pts

def _icosahedral_cluster(n, spacing=1.0):
    """Generate icosahedral cluster. 
    For n=13, 55, 147... use Mackay icosahedron.
    For other n, use the closest Mackay icosahedron and add/remove atoms or perturb."""
    
    def icosahedron_vertices(scale=1.0):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        # Edge length is 2.0 for unit circumradius? No, edge length is 2 for circumradius 1?
        # Distance between [0,1,phi] and [1,phi,0] is sqrt(1 + (phi-1)^2) = sqrt(1 + 1/phi^2) = sqrt(1 + (3-sqrt(5))/2)
        # Actually, for unit circumradius, edge length is sqrt(10-2*sqrt(5))/2 * something?
        # Let's just normalize to have nearest neighbor distance = spacing.
        # Nearest neighbor in icosahedron (unit circumradius) is edge length.
        # Edge length e = 2 * sin(pi/5) * R_circum? No.
        # In an icosahedron, R_circum = e * sqrt(10+2*sqrt(5))/4.
        # So e = R_circum * 4 / sqrt(10+2*sqrt(5)).
        # We want e = spacing.
        # So R_circum = spacing * sqrt(10+2*sqrt(5))/4.
        R_circum = spacing * np.sqrt(10 + 2*np.sqrt(5)) / 4.0
        verts = verts * R_circum
        return verts

    def mackay_icosahedron(shells):
        """Generate Mackay icosahedron with given shell radii (in units of nearest neighbor distance)."""
        verts = icosahedron_vertices(1.0) # Unit edge length
        pts = [np.zeros(3)]
        
        if shells >= 1:
            for v in verts:
                pts.append(v.copy())
            
        if shells >= 2:
            # Second shell: centers of faces
            faces = [
                [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
                [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
                [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
                [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
                [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
            ]
            for f in faces:
                center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
                r = np.linalg.norm(center)
                if r > 1e-10:
                    center = center / r * r * 1.15 # Approximate radius
                    pts.append(center)
                    
        if shells >= 3:
            # Third shell: edges of second shell? Or vertices of second icosahedron?
            # Mackay icosahedron 147 atoms: 1 + 12 + 42 + 92? No.
            # 1 (center) + 12 (1st) + 42 (2nd) + 92 (3rd)? 
            # Actually, 55 = 1 + 12 + 42.
            # 147 = 1 + 12 + 42 + 92.
            # The 3rd shell has 92 atoms.
            # For simplicity, if n > 55, we just use the 55-atom icosahedron and perturb/add.
            pass
            
        return np.array(pts)

    # Determine which Mackay icosahedron to use
    if n <= 13:
        base = mackay_icosahedron(1)
    elif n <= 55:
        base = mackay_icosahedron(2)
    else:
        base = mackay_icosahedron(2) # Use 55-atom as base and perturb
        
    # Adjust to n atoms
    if len(base) > n:
        # Remove outermost atoms
        dists = np.linalg.norm(base, axis=1)
        idx = np.argsort(dists)[:n]
        pts = base[idx]
    else:
        pts = base.copy()
        # Add random atoms near the surface
        centroid = pts.mean(axis=0)
        while len(pts) < n:
            i = np.random.randint(len(pts))
            new_p = pts[i] + np.random.normal(0, 0.5, 3)
            pts = np.vstack([pts, new_p])
            
    pts = _ensure_min_dist(pts)
    return pts

def _random_dense(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.8
    min_d = 0.55 * spacing
    max_attempts = 50000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    
    return np.array(pts)

def _basin_hopping(pos, rho, n_iter=50, kick_scale=0.5, max_iter_relax=2000, time_limit=30.0):
    """Perform Basin Hopping optimization."""
    start_time = time.time()
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Kick
        kick = rng.normal(0, kick_scale, current_pos.shape)
        new_pos = current_pos + kick
        new_pos = _ensure_min_dist(new_pos)
        
        # 2. Relax
        try:
            x0 = new_pos.flatten()
            def objective(x):
                p = x.reshape(-1, 3)
                return _morse_energy(p, rho)
            def gradient(x):
                p = x.reshape(-1, 3)
                return _morse_energy_gradient(p, rho).flatten()
                
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': max_iter_relax, 'gtol': 1e-10, 'ftol': 1e-12})
            
            relaxed_pos = result.x.reshape(-1, 3)
            relaxed_energy = result.fun
            
            if not _min_dist_check(relaxed_pos):
                continue
                
            # 3. Accept/Reject (Metropolis or just keep if better)
            # For global min search, we usually accept if better, or use Metropolis.
            # Since we want the global min, we just track the best.
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
                
            # Update current position
            # Simple: always move to new local min if valid
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
        except Exception as e:
            continue
            
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial configurations based on rho
    initial_configs = []
    
    # Determine structural bias
    if rho <= 3:
        # Long range: FCC/HCP
        initial_configs.append(('fcc', _fcc_sphere(size)))
        initial_configs.append(('hcp', _hcp_sphere(size)))
    elif rho <= 6:
        # Medium range: Mix
        initial_configs.append(('fcc', _fcc_sphere(size)))
        initial_configs.append(('hcp', _hcp_sphere(size)))
        initial_configs.append(('ico', _icosahedral_cluster(size)))
    else:
        # Short range: Icosahedral
        initial_configs.append(('ico', _icosahedral_cluster(size)))
        initial_configs.append(('fcc', _fcc_sphere(size)))
        
    # Add random restarts
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    for i in range(5):
        initial_configs.append((f'rand_{i}', _random_dense(size, rng)))
        
    # Run Basin Hopping on each
    for name, pos in initial_configs:
        if time.time() > deadline - 10:
            break
            
        try:
            # Adjust kick scale and iterations based on time remaining
            time_left = deadline - time.time()
            n_iter = max(10, int(time_left / 5)) # ~5s per iteration estimate
            
            relaxed_pos, energy = _basin_hopping(pos, rho, n_iter=n_iter, kick_scale=0.5, time_limit=time_left - 10)
            
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception as e:
            continue
            
    # If no valid result, fallback
    if best_pos is None:
        best_pos = _icosahedral_cluster(size)
        best_pos = _ensure_min_dist(best_pos)
        
    return best_pos
# EVOLVE-BLOCK-END
