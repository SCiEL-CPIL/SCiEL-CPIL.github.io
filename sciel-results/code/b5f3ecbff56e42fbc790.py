"""New program: Basin Hopping with diverse structural initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _get_kind_params(kind):
    """Map kind string to (n_exp, m_exp, c)."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _validate(positions, min_dist=0.35):
    """Check for finite coordinates and minimum distance."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                return False
    return True

def _fcc_lattice(n, spacing=1.0):
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

def _hcp_lattice(n, spacing=1.0):
    """Generate n points from an HCP lattice."""
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    hcp_pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if k % 2 == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                    z = k * c / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                    z = k * c / 2
                hcp_pts.append((x, y, z))
    hcp_pts = np.array(hcp_pts)
    hcp_pts -= np.mean(hcp_pts, axis=0)
    radii = np.linalg.norm(hcp_pts, axis=1)
    idx = np.argsort(radii)
    return hcp_pts[idx[:n]]

def _random_compact(n, spacing=0.8, seed=42):
    """Random points with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    if len(points) < n:
        min_dist = 0.35
        points = []
        attempts = 0
        max_attempts = n * 2000
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < min_dist:
                        valid = False
                        break
                if valid:
                    points.append(p)
            attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _icosahedral(n, spacing=1.0, seed=42):
    """Generate points on icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    shells = [1.0, 1.5, 2.0, 2.5, 3.0]
    rng = np.random.default_rng(seed)
    
    # Add center
    points.append(np.zeros(3))
    
    # Add shell points
    for shell in shells:
        if len(points) >= n:
            break
        # Rotate vertices slightly for diversity
        for v in verts:
            if len(points) >= n:
                break
            # Add some noise
            p = v * shell * spacing + rng.normal(0, 0.1, 3)
            # Check min dist
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.35:
                    valid = False
                    break
            if valid:
                points.append(p)
                
    # Pad if necessary
    while len(points) < n:
        p = rng.uniform(-spacing*3, spacing*3, 3)
        points.append(p)
        
    return np.array(points[:n])

def _basin_hopping(start_pos, potential, kind, n_exp, m_exp, c, time_limit, rng):
    """
    Perform Basin Hopping optimization.
    """
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
    
    current_pos = start_pos.copy()
    # Initial relaxation
    try:
        current_E, current_pos = relax(current_pos, potential, kind)
    except Exception:
        current_E = np.inf
        current_pos = start_pos.copy()
        
    if not np.isfinite(current_E):
        return np.inf, current_pos

    best_E = current_E
    best_pos = current_pos.copy()
    
    T = 1.0
    T_end = 0.01
    alpha = 0.9995
    step_size = 0.3
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate kick
        kick = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + kick
        
        # Relax
        try:
            new_E, new_pos = relax(new_pos, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(new_E):
            continue
            
        # Metropolis criterion
        delta_E = new_E - current_E
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_E = new_E
            current_pos = new_pos
            if current_E < best_E:
                best_E = current_E
                best_pos = current_pos.copy()
        
        # Anneal
        T *= alpha
        if T < T_end:
            break
            
        # Reduce step size slightly
        step_size *= 0.999
        
    return best_E, best_pos

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    n_exp, m_exp, c = _get_kind_params(kind)
    
    start_time = time.time()
    time_limit = 250.0 # Leave buffer
    
    # Generate diverse starts
    starts = []
    
    # 1. FCC
    try:
        starts.append(_fcc_lattice(size, spacing=0.8))
    except Exception:
        pass
        
    # 2. HCP
    try:
        starts.append(_hcp_lattice(size, spacing=0.8))
    except Exception:
        pass
        
    # 3. Icosahedral
    try:
        starts.append(_icosahedral(size, spacing=0.8, seed=42))
    except Exception:
        pass
        
    # 4. Random Compact (multiple seeds)
    for seed in [42, 123, 456, 789]:
        try:
            starts.append(_random_compact(size, spacing=0.8, seed=seed))
        except Exception:
            pass
            
    best_E = np.inf
    best_P = None
    
    # Allocate time for each start
    n_starts = len(starts)
    time_per_start = (time_limit - 10) / n_starts
    
    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        # Validate start
        if not _validate(start):
            continue
            
        # Run Basin Hopping
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        try:
            bh_E, bh_P = _basin_hopping(start, potential, kind, n_exp, m_exp, c, 
                                         min(time_per_start, remaining_time - 5), 
                                         np.random.default_rng(1000+i))
            if np.isfinite(bh_E) and bh_E < best_E:
                best_E = bh_E
                best_P = bh_P
        except Exception:
            continue
            
    # If we still have time, run more BH on the best solution found with smaller kicks
    if best_P is not None and time.time() - start_time < time_limit - 10:
        remaining_time = time_limit - (time.time() - start_time)
        try:
            bh_E, bh_P = _basin_hopping(best_P, potential, kind, n_exp, m_exp, c, 
                                         remaining_time - 5, 
                                         np.random.default_rng(9999))
            # Note: _basin_hopping uses standard kick size. 
            # For a "polishing" phase, we might want smaller kicks.
            # But the function signature is fixed. We can just run it.
            if np.isfinite(bh_E) and bh_E < best_E:
                best_E = bh_E
                best_P = bh_P
        except Exception:
            pass

    # Fallback
    if best_P is None:
        best_P = _random_compact(size, spacing=0.8, seed=42)
        try:
            from autoresearch_suite.chem_clusters import relax
            best_E, best_P = relax(best_P, potential, kind)
        except Exception:
            best_E = np.inf

    # Final validation
    if not _validate(best_P):
        # Attempt to fix
        for _ in range(5):
            perturbed = best_P + np.random.normal(0, 0.2, best_P.shape)
            try:
                from autoresearch_suite.chem_clusters import relax
                e, p = relax(perturbed, potential, kind)
                if _validate(p) and np.isfinite(e):
                    best_E, best_P = e, p
                    break
            except Exception:
                continue
    else:
        # Ensure it's relaxed one last time
        try:
            from autoresearch_suite.chem_clusters import relax
            e, p = relax(best_P, potential, kind)
            if _validate(p) and np.isfinite(e):
                best_E, best_P = e, p
        except Exception:
            pass

    return best_P
# EVOLVE-BLOCK-END
