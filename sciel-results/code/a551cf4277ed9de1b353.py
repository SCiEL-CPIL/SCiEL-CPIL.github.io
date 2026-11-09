"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    # Vectorized min-pair-distance check
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) > 0.35

def _fcc_cluster(n, lattice=1.0):
    """Generate a compact FCC-like cluster of n atoms."""
    fcc_offsets = np.array([
        [0, 0, 0],
        [lattice/2, lattice/2, 0],
        [lattice/2, 0, lattice/2],
        [0, lattice/2, lattice/2],
    ])
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    all_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i * lattice, j * lattice, k * lattice])
                for off in fcc_offsets:
                    all_points.append(base + off)
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    mask = dists <= max_r
    pts = all_points[mask]
    d2 = np.linalg.norm(pts, axis=1)
    order = np.argsort(d2)
    if len(pts) >= n:
        return pts[order[:n]].copy()
    else:
        # Fallback: random
        rng = np.random.default_rng(42)
        pts = rng.uniform(-max_r, max_r, size=(n, 3))
        return pts

def _hcp_cluster(n, lattice=1.0):
    """Generate a compact HCP-like cluster of n atoms."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    d2 = np.linalg.norm(pts, axis=1)
    order = np.argsort(d2)
    if len(pts) >= n:
        return pts[order[:n]].copy()
    else:
        rng = np.random.default_rng(43)
        pts = rng.uniform(-max_r, max_r, size=(n, 3))
        return pts

def _icosahedral_cluster(n):
    """Generate an icosahedral-like cluster (concentric shells)."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.zeros(3)]  # center
    for v in ico_verts:
        points.append(v.copy())
    
    # Shell 2: edge midpoints
    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i+1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 0.9 < d < 1.1:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)
    
    # Shell 3: face centers
    shell3 = []
    for i in range(n_verts):
        for j in range(i+1, n_verts):
            for k in range(j+1, n_verts):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    shell3.append(fc)
    for p in shell3:
        points.append(p)
    
    if len(points) >= n:
        return np.array(points[:n]).copy()
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * 2.0
            points.append(p)
    return np.array(points[:n]).copy()

def _perturb(pos, rng, step_size, frac=0.3):
    """Perturb a random fraction of atoms by a random displacement."""
    n = len(pos)
    n_perturb = max(1, int(n * frac))
    idx = rng.choice(n, size=n_perturb, replace=False)
    displacement = rng.normal(0, step_size, size=(n_perturb, 3))
    new_pos = pos.copy()
    new_pos[idx] += displacement
    return new_pos

def construct(instance):
    """
    Basin-Hopping global optimization with time-adaptive step size.
    Returns (n, 3) array of atom coordinates.
    """
    start_time = time.time()
    deadline = start_time + 45.0  # 45s per instance, 6 instances = 270s < 300s
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    rng = np.random.default_rng(12345 + n)
    
    # Generate initial candidates
    candidates = [
        _fcc_cluster(n, lattice=0.72),
        _hcp_cluster(n, lattice=0.72),
        _icosahedral_cluster(n),
    ]
    
    # Relax each initial candidate
    best_energy = np.inf
    best_pos = None
    for pos in candidates:
        if time.time() > deadline:
            break
        try:
            if not _validate_positions(pos):
                continue
            energy, relaxed = relax(pos, potential, kind)
            if np.isfinite(energy) and _validate_positions(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback
        pos = _fcc_cluster(n, lattice=0.72)
        try:
            best_energy, best_pos = relax(pos, potential, kind)
        except:
            return pos
    
    # Basin-Hopping loop
    current_pos = best_pos.copy()
    current_energy = best_energy
    step_size = 0.3  # initial perturbation step
    T = 0.5  # initial Metropolis temperature
    T_decay = 0.995
    
    while time.time() < deadline:
        # Perturb
        perturbed = _perturb(current_pos, rng, step_size, frac=0.25)
        
        if not _validate_positions(perturbed):
            # Reduce step size and retry
            step_size *= 0.9
            continue
        
        # Relax the perturbed configuration
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            step_size *= 0.9
            continue
        
        if not np.isfinite(new_energy) or not _validate_positions(new_pos):
            step_size *= 0.9
            continue
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Adaptive step size: increase if accepting, decrease if rejecting
        if delta_E < 0:
            step_size = min(step_size * 1.05, 0.8)
        else:
            step_size *= 0.95
        
        # Cool down
        T *= T_decay
        
        # Periodically re-perturb from best to avoid getting stuck
        if rng.random() < 0.1:
            current_pos = best_pos.copy()
            current_energy = best_energy
    
    # Final polish: one more full relaxation from best
    if time.time() < deadline - 5.0:
        try:
            final_energy, final_pos = relax(best_pos, potential, kind)
            if np.isfinite(final_energy) and _validate_positions(final_pos):
                best_pos = final_pos
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
