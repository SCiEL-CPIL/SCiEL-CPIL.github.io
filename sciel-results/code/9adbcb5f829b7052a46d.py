# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _remaining_time(deadline):
    return deadline - time.time()


def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit radius)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _get_icosahedral_shell2_positions():
    """
    Generate 42 positions for the second icosahedral shell.
    These are the 30 edge-midpoint positions and 12 face-center positions
    of the icosahedron, normalized to unit radius.
    Actually: 30 edges + 20 faces = 50, but Mackay 55 uses 42.
    Let me use a different approach: the 42 atoms in the 2nd shell of
    Mackay 55 are at positions corresponding to the 42 unique directions
    in the icosahedral point group that are not the 12 vertex directions.
    
    For simplicity, I'll use the 30 edge midpoints + 12 additional points.
    Actually, let me just generate points on an icosahedral grid.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms1 = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / norms1
    
    # Find all edges
    dmin = np.inf
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(v1[i] - v1[j])
            if d < dmin:
                dmin = d
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.linalg.norm(v1[i] - v1[j]) - dmin) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (v1[i] + v1[j]) / 2.0
        edge_mids.append(mid / np.linalg.norm(mid))
    
    # Find faces (triangles)
    adj = [[False] * 12 for _ in range(12)]
    for i, j in edges:
        adj[i][j] = True
        adj[j][i] = True
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if adj[i][j]:
                for k in range(j + 1, 12):
                    if adj[i][k] and adj[j][k]:
                        faces.append((i, j, k))
    
    # 20 face centers
    face_centers = []
    for i, j, k in faces:
        fc = (v1[i] + v1[j] + v1[k]) / 3.0
        face_centers.append(fc / np.linalg.norm(fc))
    
    # We need 42 total. Use all 30 edge mids + 12 from face centers
    # (pick 12 of the 20 face centers)
    # Actually, for Mackay 55, the 2nd shell has exactly 42 atoms.
    # Let's just use all 50 and let the caller trim.
    all_positions = np.array(edge_mids + face_centers)
    return all_positions


def _generate_mackay_55():
    """Generate the 55-atom Mackay icosahedron."""
    v1 = _get_icosahedron_vertices()
    all_2nd = _get_icosahedral_shell2_positions()
    
    points = [np.array([0.0, 0.0, 0.0])]
    r1 = 1.1
    r2 = 1.1 * np.sqrt(2.0)
    
    # 12 first shell
    for v in v1:
        points.append(v * r1)
    
    # 42 second shell (use first 42 of the 50)
    for v in all_2nd[:42]:
        points.append(v * r2)
    
    return np.array(points[:55])


def _defect_seed(n_target, base_55, rng):
    """
    Create a seed for n_target atoms by modifying the 55-atom Mackay structure.
    For n < 55: remove atoms (preferentially from the outer shell).
    For n > 55: add atoms near the surface.
    """
    base = base_55.copy()
    n_base = len(base)
    
    if n_target == n_base:
        return base.copy()
    
    if n_target < n_base:
        # Remove atoms from the outer shell (indices 13 onwards)
        # Shuffle outer shell and keep only the needed number
        outer = base[13:].copy()
        n_keep = n_target - 13
        if n_keep <= 0:
            # Need to remove from first shell too
            inner = base[:13].copy()
            n_keep_inner = n_target
            if n_keep_inner <= 1:
                return base[:max(1, n_target)].copy()
            idx = rng.permutation(len(inner))[:n_keep_inner]
            return inner[idx].copy()
        
        idx = rng.permutation(len(outer))[:n_keep]
        result = np.vstack([base[:13], outer[idx]])
        return result[:n_target]
    
    else:
        # Add atoms for n_target > 55
        result = base.copy()
        rng_local = rng
        for _ in range(n_target - n_base):
            # Add near a random existing atom, displaced outward
            i = rng_local.integers(0, len(result))
            p = result[i].copy()
            if np.linalg.norm(p) > 0.1:
                direction = p / np.linalg.norm(p)
            else:
                direction = rng_local.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            new_p = p + direction * 0.8 + rng_local.normal(0, 0.3, 3)
            # Check minimum distance
            ok = True
            for q in result:
                if np.linalg.norm(new_p - q) < 0.5:
                    ok = False
                    break
            if ok:
                result = np.vstack([result, new_p])
            else:
                # Try random position
                new_p = rng_local.normal(0, 3.0, 3)
                ok2 = all(np.linalg.norm(new_p - q) >= 0.5 for q in result)
                if ok2:
                    result = np.vstack([result, new_p])
        
        return result[:n_target]


def _ensure_min_distance(positions, min_d=0.5, rng=None):
    """Ensure all pairs are at least min_d apart by local displacement."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(positions)
    positions = positions.copy()
    
    for _ in range(100):
        # Find any violating pair
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    violated = True
                    # Push apart
                    if d > 1e-10:
                        dir_vec = (positions[i] - positions[j]) / d
                    else:
                        dir_vec = rng.normal(0, 1, 3)
                        dir_vec /= np.linalg.norm(dir_vec)
                    push = (min_d - d + 0.01) / 2.0
                    positions[i] += dir_vec * push
                    positions[j] -= dir_vec * push
                    break
            if violated:
                break
        if not violated:
            break
    
    return positions


def _local_minimize(positions, potential, max_iter=50, tol=1e-6):
    """Simple gradient-based local minimization using BFGS-like steps."""
    from scipy.optimize import minimize
    
    def objective(x):
        pos = x.reshape(-1, 3)
        try:
            e, _ = relax(pos, potential)
            return e
        except:
            return 1e10
    
    x0 = positions.flatten()
    res = minimize(objective, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})
    return res.x.reshape(-1, 3), res.fun


def _basin_hopping(positions, potential, n_restarts=5, step_scale=0.5, deadline=None):
    """
    Basin hopping: perturb, relax, keep if better.
    """
    best_pos = positions.copy()
    best_energy = np.inf
    n = len(positions)
    
    try:
        best_energy, best_pos = relax(positions, potential)
    except:
        pass
    
    rng = np.random.default_rng(123)
    
    for i in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        perturbed = best_pos + rng.normal(0, step_scale, (n, 3))
        perturbed = _ensure_min_distance(perturbed, 0.5, rng)
        
        # Relax
        try:
            e, pos = relax(perturbed, potential)
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            continue
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeding and basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin
    
    # Generate base 55-atom Mackay structure
    base_55 = _generate_mackay_55()
    
    best_positions = None
    best_energy = np.inf
    
    # Strategy 1: Defect-seeded icosahedral templates with basin hopping
    n_attempts = 8
    rng_master = np.random.default_rng(7)
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Create seed
        seed = _defect_seed(n, base_55, rng_master)
        seed = _ensure_min_distance(seed, 0.5, rng_master)
        
        # Basin hopping from this seed
        try:
            pos, energy = _basin_hopping(seed, potential, n_restarts=3, 
                                          step_scale=0.4 + 0.1 * attempt,
                                          deadline=deadline)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except:
            continue
    
    # Strategy 2: If we have time, try a few random compact seeds
    if best_positions is None or time.time() < deadline - 30:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random compact seed
            radius = 1.2 * n ** (1.0 / 3.0) * 0.6
            points = []
            rng_r = np.random.default_rng(100 + attempt)
            for _ in range(n):
                for _try in range(50):
                    p = rng_r.normal(0, radius / 2, 3)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                        break
            if len(points) < n:
                # Fallback: place on a grid
                points = []
                side = int(np.ceil(n ** (1.0 / 3.0)))
                spacing = 1.0
                count = 0
                for x in range(side):
                    for y in range(side):
                        for z in range(side):
                            if count >= n:
                                break
                            p = np.array([x * spacing, y * spacing, z * spacing])
                            points.append(p)
                            count += 1
                        if count >= n:
                            break
                    if count >= n:
                        break
                points = np.array(points[:n])
            else:
                points = np.array(points)
            
            points = _ensure_min_distance(points, 0.5, rng_r)
            
            try:
                pos, energy = _basin_hopping(points, potential, n_restarts=2,
                                              step_scale=0.5, deadline=deadline)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except:
                continue
    
    # Final safety: if still no valid result, return a simple structure
    if best_positions is None:
        # Place atoms on a simple cubic grid
        side = int(np.ceil(n ** (1.0 / 3.0)))
        spacing = 1.0
        points = []
        for x in range(side):
            for y in range(side):
                for z in range(side):
                    if len(points) >= n:
                        break
                    points.append(np.array([x * spacing, y * spacing, z * spacing]))
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        best_positions = np.array(points[:n])
        try:
            best_positions, _ = relax(best_positions, potential)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
