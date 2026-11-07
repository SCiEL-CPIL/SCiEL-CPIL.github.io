"""Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosa_shell(n):
    """Generate Mackay icosahedral cluster with n atoms."""
    phi = (1 + 5 ** 0.5) / 2
    # icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay shells: 1, 12, 42, 100, ...
    # shell radii proportional to sqrt(k*(k+1))/2 for shell k (0-indexed)
    # Actually for icosahedral: shell k has 10*k*(k+1)+2 atoms (k>=1), shell 0 has 1
    # Total: 1, 13, 55, 147, 301, ...
    
    # For arbitrary n, we fill shells and then trim
    points = [np.zeros(3)]  # center
    k = 1
    while len(points) < n:
        # shell k has 10*k*(k+1)+2 atoms
        shell_size = 10 * k * (k + 1) + 2
        # radius for shell k
        r = k * 1.0  # approximate, will be adjusted by relaxation
        # Generate points on the icosahedron at radius r
        for v in verts:
            points.append(r * v)
            if len(points) >= n:
                break
        # Add more points by interpolating between vertices
        if len(points) < n:
            # Add points along edges
            for i in range(12):
                v1 = verts[i]
                v2 = verts[(i + 1) % 12]
                for t in np.linspace(0.2, 0.8, 6):
                    p = r * (v1 * (1 - t) + v2 * t)
                    p /= np.linalg.norm(p) * np.linalg.norm(p)  # normalize
                    points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
        k += 1
    
    points = np.array(points[:n])
    return points


def _fcc_cluster(n):
    """Generate FCC-like cluster."""
    points = []
    # FCC lattice points
    a = 1.0  # lattice constant
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
                if len(points) >= n * 2:
                    break
            if len(points) >= n * 2:
                break
        if len(points) >= n * 2:
            break
    
    points = np.array(points)
    # Keep only points within a sphere of radius proportional to n^(1/3)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    # Sort by distance and keep closest n
    idx = np.argsort(dists)
    points = points[idx[:n]]
    # Center
    points -= points.mean(axis=0)
    return points


def _hcp_cluster(n):
    """Generate HCP-like cluster."""
    points = []
    a = 1.0
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: A layer at z=0, B layer at z=c/2
                # A layer: hexagonal
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3) / 2 * j
                z = 0
                points.append(np.array([x, y, z]))
                # B layer
                x2 = a * (i + 0.5 * ((j + 1) % 2))
                y2 = a * np.sqrt(3) / 2 * (j + 1)
                z2 = c / 2
                points.append(np.array([x2, y2, z2]))
                if len(points) >= n * 2:
                    break
            if len(points) >= n * 2:
                break
        if len(points) >= n * 2:
            break
    
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points


def _random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    return np.array(points)


def _perturb(positions, rng, magnitude=0.1):
    """Randomly perturb positions."""
    return positions + rng.normal(0, magnitude, positions.shape)


def _basin_hopping(positions, kind, rng, n_iter=5, temp_start=0.1, temp_end=0.01):
    """Simple basin hopping."""
    current_energy, current_pos = relax(positions, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(n_iter):
        temp = temp_start * (temp_end / temp_start) ** (i / max(n_iter - 1, 1))
        perturbed = _perturb(current_pos, rng, temp)
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        # Metropolis acceptance
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(temp, 1e-6)):
            current_energy = new_energy
            current_pos = new_pos
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    rng = np.random.default_rng(n)
    
    # Generate diverse starting structures
    starts = []
    
    # 1. Icosahedral
    try:
        starts.append(_icosa_shell(n))
    except:
        pass
    
    # 2. FCC-like
    try:
        starts.append(_fcc_cluster(n))
    except:
        pass
    
    # 3. HCP-like
    try:
        starts.append(_hcp_cluster(n))
    except:
        pass
    
    # 4-13. Random compact (10 more)
    for _ in range(10):
        try:
            starts.append(_random_compact(n, rng))
        except:
            pass
    
    # Relax each starting structure
    best_energy = np.inf
    best_pos = None
    
    for i, pos in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except:
            continue
    
    if best_pos is None:
        # Fallback: return a simple linear arrangement
        return np.array([[i, 0, 0] for i in range(n)])
    
    # Basin hopping from the best structure
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Do basin hopping
        n_iter = int(remaining_time / 5)  # ~5s per iteration
        n_iter = max(3, min(n_iter, 20))
        
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, kind, rng, n_iter=n_iter)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    # Final check: ensure no atoms are too close
    if best_pos is not None:
        # Check minimum distance
        from scipy.spatial.distance import pdist
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
            # Spread out atoms
            for _ in range(100):
                best_pos = _perturb(best_pos, rng, 0.05)
                min_dist = pdist(best_pos).min()
                if min_dist >= 0.35:
                    break
    
    return best_pos
# EVOLVE-BLOCK-END
