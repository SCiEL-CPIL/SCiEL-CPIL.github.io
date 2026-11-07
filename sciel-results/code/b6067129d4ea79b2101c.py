"""Structure-motif + adaptive basin hopping for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 320.0  # safety margin before 330s kill


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


_START_TIME = time.time()


def _fcc_cluster(n, scale=1.0):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    # FCC lattice vectors
    a = scale
    # Generate FCC points in a cube
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for dx, dy in [(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + (dx + dy) * 0]) * a
                    # Actually let's do proper FCC
                    break
                # Redo: proper FCC points
                for (fx, fy, fz) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + fx, j + fy, k + fz]) * a
                    pts.append(p)
    
    pts = np.array(pts)
    # Center
    pts -= pts.mean(axis=0)
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, scale=1.0):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    a = scale
    c = a * np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    # HCP layers
    for layer in range(-6, 7):
        z = layer * c / 2.0
        for i in range(-6, 7):
            for j in range(-6, 7):
                # AB stacking
                if layer % 2 == 0:
                    x = i * a + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a + (j % 2) * a / 2.0 + a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                pts.append(np.array([x, y, z]))
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosa_cluster(n, scale=1.0):
    """Generate an icosahedral (Mackay) cluster with n atoms."""
    phi = (1 + 5 ** 0.5) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    # Shell 1: 12 vertices at radius 1
    r1 = scale
    for v in verts:
        pts.append(v * r1)
    
    # Shell 2: 42 atoms at radius sqrt(2) * r1 (approximate)
    r2 = scale * np.sqrt(2.0)
    # Generate points on icosahedral symmetry
    for v1 in verts:
        for v2 in verts:
            # Midpoints of edges
            p = (v1 + v2) / np.linalg.norm(v1 + v2) * r2
            # Check if already added
            if not any(np.allclose(p, q, atol=0.01) for q in pts):
                pts.append(p)
    
    # Shell 3: 100 atoms at radius sqrt(3) * r1
    r3 = scale * np.sqrt(3.0)
    for i in range(12):
        for j in range(12):
            if i != j:
                p = (verts[i] + 2 * verts[j]) / np.linalg.norm(verts[i] + 2 * verts[j]) * r3
                if not any(np.allclose(p, q, atol=0.01) for q in pts):
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster with n atoms."""
    pts = []
    a = scale
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                pts.append(np.array([i, j, k]) * a)
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, rng, scale=0.72):
    """Generate a random compact cluster."""
    radius = scale * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.35
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we didn't get enough, fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _perturb(positions, rng, scale=0.5):
    """Perturb positions by adding random noise."""
    return positions + rng.normal(0, scale, positions.shape)


def _polish(positions, potential, kind):
    """Relax positions using L-BFGS-B."""
    try:
        energy, new_positions = relax(positions, potential, kind)
        if np.all(np.isfinite(new_positions)):
            return energy, new_positions
    except Exception:
        pass
    # Fallback: use gradient directly
    try:
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        return energy, positions
    except Exception:
        return np.inf, positions


def _basin_hopping(positions, potential, kind, rng, n_iterations=50, initial_scale=0.5, min_scale=0.1):
    """Adaptive basin hopping."""
    best_energy, best_positions = _polish(positions, potential, kind)
    current_energy = best_energy
    current_positions = best_positions.copy()
    scale = initial_scale
    accepted = 0
    
    for it in range(n_iterations):
        if _time_left() < 5.0:
            break
        
        new_positions = _perturb(current_positions, rng, scale)
        new_energy, new_positions = _polish(new_positions, potential, kind)
        
        if not np.isfinite(new_energy):
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_positions
            accepted += 1
        else:
            # Accept with probability exp(-(new_energy - current_energy) / T)
            # Use a simple temperature schedule
            T = 0.01 * (1.0 + 0.1 * it)
            if np.log(rng.random()) < -(new_energy - current_energy) / T:
                current_energy = new_energy
                current_positions = new_positions
                accepted += 1
        
        # Update best
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = new_positions
        
        # Adapt scale
        if accepted > 0 and it % 10 == 0:
            scale = max(min_scale, scale * 0.9)
        elif it % 10 == 0:
            scale = min(initial_scale, scale * 1.2)
    
    return best_energy, best_positions


def construct(instance):
    """Best of multiple structure-motif starts with adaptive basin hopping."""
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(size * 7919 + 12345)
    
    # Generate candidate starting points
    candidates = []
    
    # FCC
    candidates.append(_fcc_cluster(size, scale=0.72))
    # HCP
    candidates.append(_hcp_cluster(size, scale=0.72))
    # Icosahedral
    candidates.append(_icosa_cluster(size, scale=0.72))
    # BCC
    candidates.append(_bcc_cluster(size, scale=0.72))
    
    # Random compact starts
    for _ in range(4):
        candidates.append(_random_compact(size, rng, scale=0.72))
    
    best_energy = np.inf
    best_positions = candidates[0]
    
    # Try each candidate with basin hopping
    for i, cand in enumerate(candidates):
        if _time_left() < 10.0:
            break
        
        # First polish the candidate
        energy, polished = _polish(cand, potential, kind)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_positions = polished
        
        # Basin hopping from the polished position
        if _time_left() < 5.0:
            break
        
        bh_energy, bh_positions = _basin_hopping(
            polished, potential, kind, rng,
            n_iterations=30, initial_scale=0.4, min_scale=0.1
        )
        
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions
    
    # Final polish
    if _time_left() > 2.0:
        final_energy, final_positions = _polish(best_positions, potential, kind)
        if np.isfinite(final_energy):
            best_positions = final_positions
    
    return best_positions
# EVOLVE-BLOCK-END
