"""
Strategy: Sequential Adaptive Basin Hopping with Metropolis Acceptance.
This approach uses a single-threaded, time-budgeted Basin Hopping loop.
It initializes from diverse structural templates (FCC, HCP, BCC, Icosahedral)
and uses an adaptive step size with a Metropolis criterion (dual temperature)
to accept uphill moves, allowing the search to escape local minima.
The step size is adjusted based on the acceptance rate to balance exploration
and exploitation.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None
_SAFETY_MARGIN = 5.0

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return max(0, _DEADLINE_SEC - (time.time() - _START_TIME) - _SAFETY_MARGIN)

def _generate_fcc(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_hcp(n, spacing=0.72):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_bcc(n, spacing=0.72):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_icosahedron(n, spacing=0.72):
    """Generate n atoms from an icosahedral cluster."""
    a = spacing
    # icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * (a / (2 * phi))
    
    points = list(verts)
    
    # Add more layers if needed
    if n > 12:
        # Add atoms in the centers of the faces or between vertices
        # Simple approach: add random points near the surface
        rng = np.random.default_rng(42)
        radius = np.linalg.norm(verts[0])
        while len(points) < n:
            p = rng.uniform(-1, 1, 3)
            p = p / np.linalg.norm(p) * radius
            # Check if too close to existing points
            if all(np.linalg.norm(p - q) > 0.5 * a for q in points):
                points.append(p)
            if len(points) > 1000: # Safety break
                break
    
    if len(points) < n:
        return None
        
    points = np.array(points[:n])
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points

def _generate_random(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Generate diverse starting points
    starts = []
    
    fcc = _generate_fcc(n)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _generate_hcp(n)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _generate_bcc(n)
    if bcc is not None:
        starts.append(bcc)
        
    ico = _generate_icosahedron(n)
    if ico is not None:
        starts.append(ico)
        
    # Add some random starts
    for seed in range(4):
        rand = _generate_random(n, seed=seed)
        starts.append(rand)
        
    # Initialize best
    best_energy = float('inf')
    best_pos = None
    current_energy = float('inf')
    current_pos = None
    
    # Select initial structure: try to relax all starts and pick the best one
    for start in starts:
        if _time_left() < 10:
            break
        try:
            e, pos = relax(start, potential, kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
                current_energy = e
                current_pos = pos.copy()
        except Exception:
            pass
            
    if best_pos is None:
        # Fallback
        fallback_start = _generate_random(n, seed=999)
        try:
            e, pos = relax(fallback_start, potential, kind)
            return pos
        except Exception:
            return fallback_start
            
    # Basin Hopping Loop
    rng = np.random.default_rng(12345)
    step_size = 0.2
    acceptance_rate = 0.5
    T = 0.1 # Initial temperature for Metropolis
    T_max = 1.0
    T_min = 0.01
    
    while _time_left() > 1:
        # Perturb
        noise = rng.normal(0, step_size, size=(n, 3))
        perturbed = current_pos + noise
        
        # Relax
        try:
            e_new, pos_new = relax(perturbed, potential, kind)
        except Exception:
            e_new = float('inf')
            pos_new = None
            
        if not np.isfinite(e_new):
            continue
            
        # Metropolis Acceptance
        delta_e = e_new - current_energy
        if delta_e < 0:
            accept = True
        else:
            # P(accept) = exp(-delta_e / T)
            if delta_e < T * 50: # Avoid underflow
                prob = np.exp(-delta_e / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            current_energy = e_new
            current_pos = pos_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
            acceptance_rate = 0.8 * acceptance_rate + 0.2 * 1.0
        else:
            acceptance_rate = 0.8 * acceptance_rate + 0.2 * 0.0
            
        # Adapt step size and temperature
        # Target acceptance rate ~0.5
        if acceptance_rate > 0.6:
            step_size *= 1.1
            T = min(T * 1.1, T_max)
        elif acceptance_rate < 0.4:
            step_size *= 0.9
            T = max(T * 0.9, T_min)
            
        step_size = np.clip(step_size, 0.05, 1.0)
        
        # Cooling schedule: slowly decrease T over time to focus on exploitation
        # But keep it high enough to escape deep local minima
        # Simple linear cooling
        elapsed = time.time() - _START_TIME
        progress = elapsed / _DEADLINE_SEC
        T_target = T_max * (1 - progress) + T_min * progress
        # Blend current T with target to avoid sudden drops
        T = 0.9 * T + 0.1 * T_target

    return best_pos
# EVOLVE-BLOCK-END
