"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _generate_macky_icosahedron_55(scale=1.0):
    """
    Generate a 55-atom Mackay icosahedron.
    Shell 1: 1 center + 12 vertices (13 atoms)
    Shell 2: 30 edge midpoints + 12 face centers (42 atoms)
    Total: 55 atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    # Standard coordinates
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 vertices
    # Distance from center to vertex in a regular icosahedron with edge length a is R = a * sqrt(10 + 2*sqrt(5)) / 4
    # If we want nearest neighbor distance (center to vertex) to be ~1.8, we scale accordingly.
    # However, the 42 atoms in shell 2 are closer to the vertices than the center.
    # Let's determine the scale such that the nearest neighbor distance is optimal.
    # In a Mackay icosahedron, the nearest neighbors are:
    # 1. Center to Shell 1 vertices (12 neighbors)
    # 2. Shell 1 vertex to Shell 2 atoms (6 neighbors per vertex? No, each vertex is connected to 6 shell 2 atoms)
    # 3. Shell 2 atoms to each other.
    
    # Let's construct the geometry on a unit sphere first, then scale.
    # Shell 1 points are on unit sphere.
    shell1 = v.copy()
    
    # Find edges of the icosahedron
    # Edge length of unit icosahedron: 
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + phi^2 - 2phi + 1 + phi^2) = sqrt(2 + 2phi^2 - 2phi)
    # phi^2 = phi + 1
    # 2 + 2(phi+1) - 2phi = 2 + 2phi + 2 - 2phi = 4. So edge length is 2.
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell1[i] - shell1[j])
            if abs(d - 2.0) < 1e-2:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (shell1[i] + shell1[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
        
    # 12 face centers
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(shell1[i] - shell1[j])
                d2 = np.linalg.norm(shell1[j] - shell1[k])
                d3 = np.linalg.norm(shell1[k] - shell1[i])
                if abs(d1 - 2.0) < 1e-2 and abs(d2 - 2.0) < 1e-2 and abs(d3 - 2.0) < 1e-2:
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (shell1[i] + shell1[j] + shell1[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
        
    shell2 = np.array(edge_mids + face_centers)
    
    # Now we have points on unit sphere.
    # We need to scale them to appropriate radii.
    # In a Mackay icosahedron, the ratio of radii R1/R0 and R2/R0 is fixed by the geometry.
    # However, for the Dzugutov potential, the optimal spacing might differ slightly from the ideal Mackay ratio if the potential is not purely geometric.
    # But usually, the ideal Mackay geometry is a very good starting point.
    
    # Let's estimate the optimal scale.
    # The nearest neighbor distance in the unit Mackay icosahedron (R=1 for shell 1) is:
    # Center to Shell 1: 1.0
    # Shell 1 to Shell 2: 
    # Distance between vertex (1,0,0)-like and edge midpoint.
    # Let's calculate the distance between shell1[0] and edge_mids[0].
    # shell1[0] is (0, 1/sqrt(1+phi^2), phi/sqrt(1+phi^2)).
    # This is getting complicated. Let's just use a single scale factor for the whole cluster and let relaxation adjust it.
    # Or, better, use the fact that the nearest neighbor distance in the ideal Mackay icosahedron is constant.
    # If we scale the whole cluster by S, all distances scale by S.
    # We want the nearest neighbor distance to be ~1.8.
    
    # Calculate the minimum distance in the unit cluster (shell 1 at R=1, shell 2 at R=1).
    # Wait, in a true Mackay icosahedron, shell 2 is NOT at the same radius as shell 1.
    # The points I generated (edge mids, face centers) are on the unit sphere.
    # To form a Mackay icosahedron, we must project them to the correct radii.
    # The vertices of the icosahedron are at distance R1 from center.
    # The edge midpoints are at distance R2 = R1 * sqrt(1 - (L/2)^2 / R1^2) ? No.
    # In a regular icosahedron, the distance from center to vertex is R_v.
    # The distance from center to edge midpoint is R_e = R_v * cos(theta/2) where theta is the angle subtended by an edge.
    # For a regular icosahedron, R_e / R_v = sqrt(5 - 2*sqrt(5)) / 2 ? No.
    # Let's calculate R_e/R_v.
    # Vertex v = (0, 1, phi). Norm = sqrt(1 + phi^2) = sqrt(1 + phi + 1) = sqrt(2 + phi).
    # Edge midpoint m = (v1 + v2) / 2. Norm = |v1 + v2| / 2.
    # |v1 + v2|^2 = |v1|^2 + |v2|^2 + 2 v1.v2 = 2|v|^2 + 2|v|^2 cos(theta).
    # So R_e = R_v * sqrt(1 + cos(theta)) / sqrt(2) * sqrt(2)? No.
    # R_e = R_v * sqrt(1 + cos(theta)) / sqrt(2) * sqrt(2) ... 
    # Actually, R_e = R_v * cos(alpha) where alpha is half the central angle of an edge.
    
    # Let's just calculate the distances numerically for the unit sphere points.
    # If we place shell 1 at radius r1 and shell 2 at radius r2, we can optimize r1, r2.
    # However, a simpler heuristic: The ideal Mackay icosahedron has a specific geometry.
    # Let's assume the nearest neighbor distance is d_nn.
    # In the ideal Mackay icosahedron, d_nn is the same for all pairs.
    # Let's find the scale factor S such that the nearest neighbor distance is 1.8.
    
    # First, let's construct the points with shell 1 at radius 1.0 and shell 2 at radius 1.0 (on the sphere).
    # Then calculate the minimum distance.
    # Then scale the whole cluster so that min_dist becomes 1.8.
    
    all_points_unit = np.vstack([center, shell1, shell2])
    
    # Calculate pairwise distances
    # This is O(N^2), N=55. 55*55 = 3025. Very fast.
    diffs = all_points_unit[:, np.newaxis, :] - all_points_unit[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist_unit = np.min(dists)
    
    # Scale factor
    scale_factor = 1.8 / min_dist_unit
    
    # Apply scale
    all_points_scaled = all_points_unit * scale_factor
    
    return all_points_scaled


def _generate_macky_icosahedron_13(scale=1.0):
    """Generate a 13-atom icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    center = np.array([[0.0, 0.0, 0.0]])
    all_points_unit = np.vstack([center, v])
    
    diffs = all_points_unit[:, np.newaxis, :] - all_points_unit[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist_unit = np.min(dists)
    
    scale_factor = 1.8 / min_dist_unit
    return all_points_unit * scale_factor


def _random_compact(n, seed, spacing=1.0, radius_factor=0.85):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.7 * spacing
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
        
    return np.array(points[:n])


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(points, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # Pre-generate master structures
    # 55-atom Mackay
    macky_55 = _generate_macky_icosahedron_55()
    # 13-atom Icosahedron
    ico_13 = _generate_macky_icosahedron_13()
    
    # Safeguard: if generation failed, use random
    if len(macky_55) != 55:
        macky_55 = _random_compact(55, seed=999, spacing=1.0, radius_factor=0.85)
    if len(ico_13) != 13:
        ico_13 = _random_compact(13, seed=998, spacing=1.0, radius_factor=0.85)
    
    # Strategy based on size
    if size == 55:
        # Directly use Mackay 55
        _try_relax(macky_55, "mackay55")
        
        # Perturbed Mackay 55
        rng = np.random.default_rng(42)
        for _ in range(5):
            perturbed = macky_55 + rng.normal(0, 0.05, macky_55.shape)
            _try_relax(perturbed, "mackay55_pert")
            
    elif size == 57 or size == 60:
        # Start with Mackay 55 and add atoms
        # Add atoms in symmetric positions on the surface
        # For now, just perturb and add random atoms near the surface
        base = macky_55
        n_add = size - 55
        for attempt in range(10):
            rng = np.random.default_rng(attempt)
            # Add atoms near the surface
            added = []
            for _ in range(n_add):
                # Pick a random surface atom
                idx = rng.integers(12, 55)  # Shell 2
                p = base[idx]
                # Move outwards
                dir_vec = p / np.linalg.norm(p)
                new_p = p + dir_vec * 0.5 + rng.normal(0, 0.1, 3)
                added.append(new_p)
            
            full = np.vstack([base, np.array(added)])
            _try_relax(full, "super_macky")
            
        # Also try random compact
        rc = _random_compact(size, seed=size, spacing=1.0, radius_factor=0.9)
        _try_relax(rc, "random")
        
    elif size < 55:
        # Use Mackay subset only if it is valid
        if len(macky_55) == 55:
            n_remove = 55 - size
            rng = np.random.default_rng(size)
            shell2_indices = np.arange(13, 55)
            rng.shuffle(shell2_indices)
            remove_indices = shell2_indices[:n_remove]
            
            mask = np.ones(55, dtype=bool)
            mask[remove_indices] = False
            sub_macky = macky_55[mask]
            
            _try_relax(sub_macky, "sub_macky")
            
            # Perturbed sub-macky
            for _ in range(5):
                perturbed = sub_macky + rng.normal(0, 0.05, sub_macky.shape)
                _try_relax(perturbed, "sub_macky_pert")
        else:
            # Fallback to random if Mackay is invalid
            rc = _random_compact(size, seed=size, spacing=1.0, radius_factor=0.9)
            _try_relax(rc, "random")
            
        # Also try 13-atom icosahedron based structures if size is small
        if size <= 13 and len(ico_13) == 13:
            _try_relax(ico_13[:size], "ico13")
        elif size <= 25:
            rc = _random_compact(size, seed=size, spacing=1.0, radius_factor=0.9)
            _try_relax(rc, "random")
        else:
            # Random compact
            rc = _random_compact(size, seed=size, spacing=1.0, radius_factor=0.9)
            _try_relax(rc, "random")
            rc2 = _random_compact(size, seed=size+1, spacing=1.1, radius_factor=0.8)
            _try_relax(rc2, "random2")

    # If time remains, do more random starts
    if _time_remaining() > 30:
        for i in range(5):
            if _time_remaining() < 10: break
            rc = _random_compact(size, seed=size + 100 + i, spacing=1.0 + 0.1*i, radius_factor=0.85)
            _try_relax(rc, f"random_{i}")

    return best_positions

# EVOLVE-BLOCK-END
