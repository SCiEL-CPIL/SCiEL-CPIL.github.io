"""
Combined program for Sutton-Chen cluster optimization.
Strategy: Multi-start portfolio (Mackay, FCC, HCP, BCC, Hybrid, Random)
relaxed via L-BFGS-B, followed by adaptive basin hopping on the best candidates.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.spatial.distance import pdist

# Global deadline guard: 330s limit, aim for 280s to be safe
DEADLINE = time.time() + 280

# --- Structural Generators ---

def _generate_icosahedron_vertices(radius=1.0):
    """Generate coordinates for a regular icosahedron with 'radius'. Returns 12 vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norm = np.linalg.norm(verts, axis=1)
    scale = radius / norm
    verts = verts * scale[:, np.newaxis]
    return verts

def _generate_mackay_icosahedron(n_atoms, spacing=0.72):
    """
    Generate a Mackay icosahedral cluster.
    Shells: 1, 12, 42, 92, 156...
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    points = [np.zeros(3)]
    
    if n_atoms > 1:
        verts1 = _generate_icosahedron_vertices(radius=1.0)
        points.extend(verts1.tolist())
        
    if n_atoms > 13:
        # Shell 2: 42 atoms (12 vertices + 30 edge midpoints scaled)
        scale2 = 1.701
        verts2 = _generate_icosahedron_vertices(radius=scale2)
        points.extend(verts2.tolist())
        
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3:
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale2
                    edges.append(mid)
        points.extend(edges)
        
    if len(points) < n_atoms:
        # Shell 3: 92 atoms (approximated by denser icosahedral surface)
        # For simplicity, we add a 3rd shell of 12 verts + 30 edges + 20 face centers + 30 more?
        # Actually, standard Mackay shell 3 is complex. 
        # Let's use a simpler approach: generate a dense icosahedral shell and pick closest.
        # Or just scale the 55 points and add more.
        # For n=38, 59, 75, 80, we need more.
        # Let's generate a 4th shell of 156 atoms (approx) or just use FCC/HCP for larger n.
        # However, Mackay is good for 38, 59.
        # For 38: 1+12+42 = 55. So 38 is within first 2 shells (1+12+25).
        # For 59: 1+12+42 = 55. Need 4 more.
        # For 75: Need 20 more.
        # For 80: Need 25 more.
        
        # Let's add a 3rd shell of 92 points properly if possible, or approximate.
        # Approximation: Scale the 55 points by 2.571 and add them? No, that's shell 4.
        # Let's just add points from a larger icosahedron.
        scale3 = 2.571
        verts3 = _generate_icosahedron_vertices(radius=scale3)
        points.extend(verts3.tolist())
        
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3:
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale3
                    points.append(mid)
                    
        # If still not enough, add face centers of icosahedron scaled
        # Faces are triangles. We need to identify them.
        # For now, if still short, we will pad with random or FCC in construct().
        
    points = np.array(points)
    if len(points) > n_atoms:
        # Keep the n_atoms closest to the center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:n_atoms]]
    elif len(points) < n_atoms:
        # Pad with random points if necessary (should be rare for main cases)
        rng = np.random.default_rng(123)
        while len(points) < n_atoms:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 2.0
            points = np.vstack([points, p])
            
    points = points - points.mean(axis=0)
    points = points * spacing
    return points

def _generate_fcc_shells(n_atoms, spacing=0.72):
    """
    Generate a compact FCC-like cluster by taking the n_atoms closest points
    from an infinite FCC lattice to the origin.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n_atoms]]

def _generate_hcp_shells(n_atoms, spacing=0.72):
    """
    Generate a compact HCP-like cluster.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n_atoms]]

def _generate_bcc_shells(n_atoms, spacing=0.72):
    """
    Generate a compact BCC-like cluster.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n_atoms]]

def _generate_hybrid_icosa_fcc(n_atoms, spacing=0.72):
    """
    Hybrid: Icosahedral core (13 atoms) + FCC-like shell.
    """
    if n_atoms <= 13:
        return _generate_mackay_icosahedron(n_atoms, spacing)
        
    core = _generate_mackay_icosahedron(13, spacing)
    remaining = n_atoms - 13
    
    # Generate FCC points
    fcc_pts = _generate_fcc_shells(n_atoms + 10, spacing)
    shell_points = []
    for p in fcc_pts:
        min_d = np.min(np.linalg.norm(core - p, axis=1))
        if min_d > 0.8 * spacing:
            shell_points.append(p)
        if len(shell_points) >= remaining:
            break
            
    if len(shell_points) < remaining:
        rng = np.random.default_rng(123)
        while len(shell_points) < remaining:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 2.0 * spacing
            shell_points.append(p)
            
    cluster = np.vstack([core, np.array(shell_points[:remaining])])
    return cluster

def _random_cluster(n, seed, min_dist=0.6, spacing=0.72):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = spacing * n ** (1.0 / 3.0) * 1.5
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        points.append(p)
    return np.array(points)

# --- Optimization Helpers ---

def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    min_d = pdist(pos, metric='euclidean').min()
    return min_d >= 0.35

def _basin_hopping(positions, potential, kind, n_steps=10, step_size=0.1, seed=42):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    rng = np.random.default_rng(seed)
    current_energy, current_pos = relax(positions, potential, kind)
    if not _validate_positions(current_pos):
        return current_energy, current_pos
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(n_steps):
        if time.time() > DEADLINE - 20:
            break
            
        # Perturb
        perturb = rng.normal(0, step_size, positions.shape)
        perturbed = current_pos + perturb
        
        # Relax
        try:
            e_new, pos_new = relax(perturbed, potential, kind)
            if not _validate_positions(pos_new):
                continue
        except:
            continue
            
        # Accept if lower (or with small probability if higher, but simple BH usually accepts only lower)
        if e_new < current_energy:
            current_energy = e_new
            current_pos = pos_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
        else:
            # Simple Boltzmann acceptance
            # delta_E = e_new - current_energy
            # If delta_E < 0, accept.
            # If delta_E > 0, accept with prob exp(-delta_E / T)
            # T is tricky. Let's just stick to downhill for simplicity and speed, 
            # or use a small fixed T.
            pass
            
    return best_energy, best_pos

def construct(instance):
    """
    Best of multiple structured and random starting clusters, each relaxed locally,
    followed by basin hopping on the top candidates.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    t_start = time.time()
    
    # Spacing is typically around 0.7-0.8 for SC
    spacing = 0.72
    
    initial_configs = []
    
    # 1. Mackay Icosahedral
    try:
        cfg = _generate_mackay_icosahedron(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 2. FCC
    try:
        cfg = _generate_fcc_shells(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 3. HCP
    try:
        cfg = _generate_hcp_shells(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 4. BCC
    try:
        cfg = _generate_bcc_shells(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 5. Hybrid Icosa-FCC
    try:
        cfg = _generate_hybrid_icosa_fcc(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 6-10. Random clusters with different seeds
    for seed in [size, size + 100, size + 200, size + 300, size + 400]:
        try:
            cfg = _random_cluster(size, seed, min_dist=0.6, spacing=spacing)
            if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
                initial_configs.append(cfg)
        except Exception:
            pass
            
    # Remove duplicates
    unique_configs = []
    for cfg in initial_configs:
        is_dup = False
        for ucfg in unique_configs:
            if np.allclose(cfg, ucfg, atol=0.1):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
            
    initial_configs = unique_configs
    
    # Phase 1: Local Relaxation
    relaxed_results = []
    for i, cfg in enumerate(initial_configs):
        if time.time() > DEADLINE - 60:
            break
            
        if cfg.shape[0] > 1:
            min_d = pdist(cfg, metric='euclidean').min()
            if min_d < 0.35:
                continue
        
        try:
            energy, positions = relax(cfg, potential, kind)
            
            if not np.all(np.isfinite(positions)):
                continue
            if positions.shape[0] != size:
                continue
            if positions.shape[1] != 3:
                continue
            if positions.shape[0] > 1:
                min_d = pdist(positions, metric='euclidean').min()
                if min_d < 0.35:
                    continue
            
            relaxed_results.append((energy, positions))
        except Exception:
            continue
            
    if not relaxed_results:
        # Fallback
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        return positions

    # Sort by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    best_energy = relaxed_results[0][0]
    best_positions = relaxed_results[0][1].copy()
    
    # Phase 2: Basin Hopping on top candidates
    # We have ~60s left (DEADLINE - t_start was ~220s used for 10 relaxations? 
    # Relaxations are fast usually. Let's check time.)
    # Actually, let's just run BH on the best 3 if time permits.
    
    for i, (e, pos) in enumerate(relaxed_results[:3]):
        if time.time() > DEADLINE - 20:
            break
            
        try:
            bh_energy, bh_pos = _basin_hopping(pos, potential, kind, n_steps=15, step_size=0.15, seed=42+i)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
        except Exception:
            continue
            
    return best_positions
# EVOLVE-BLOCK-END
