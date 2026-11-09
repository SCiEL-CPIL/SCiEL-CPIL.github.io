"""Dzugutov cluster global minima (polytetrahedral model metal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_icosahedron_grown(n, seed=42):
    """
    Generate an icosahedral cluster by growing from a central icosahedron.
    Uses face-center addition to maintain icosahedral symmetry.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12 atoms)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    # Start with center + icosahedron vertices
    points = [np.zeros(3)]
    if n > 1:
        points.extend(verts.tolist())
    
    if n <= 13:
        return np.array(points[:n])
    
    # Find all triangular faces of the current polyhedron
    def _find_faces(pts):
        """Find all triangular faces of a convex polyhedron."""
        pts = np.array(pts)
        n_pts = len(pts)
        faces = []
        
        # For each triple of points, check if it forms a face
        # A face is a triangle where all other points are on the same side
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    # Check if triangle is non-degenerate
                    v1 = p2 - p1
                    v2 = p3 - p1
                    normal = np.cross(v1, v2)
                    if np.linalg.norm(normal) < 1e-10:
                        continue
                    normal = normal / np.linalg.norm(normal)
                    
                    # Check if all other points are on the same side
                    all_same_side = True
                    for m in range(n_pts):
                        if m in (i, j, k):
                            continue
                        v = pts[m] - p1
                        dot = np.dot(v, normal)
                        if dot > 1e-8:  # On the "wrong" side
                            all_same_side = False
                            break
                    
                    if all_same_side:
                        faces.append([i, j, k])
        
        return faces
    
    # Add atoms at face centers until we reach n
    rng = np.random.default_rng(seed)
    current_points = np.array(points)
    
    while len(current_points) < n:
        faces = _find_faces(current_points)
        if not faces:
            break
        
        # Compute face centers
        face_centers = []
        for face in faces:
            center = np.mean(current_points[face], axis=0)
            # Check if this center is already close to an existing point
            min_dist = np.min(np.linalg.norm(current_points - center, axis=1))
            if min_dist > 0.3:  # Not too close to existing
                face_centers.append((min_dist, center))
        
        if not face_centers:
            break
        
        # Sort by distance from existing points (add farthest first)
        face_centers.sort(key=lambda x: -x[0])
        
        # Add the best face center
        best_center = face_centers[0][1]
        current_points = np.vstack([current_points, best_center])
    
    return current_points[:n]

def _generate_fcc(n):
    """Generate a truncated FCC cluster."""
    points = []
    # FCC lattice basis vectors
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]
    
    # Generate points in a grid
    N = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + np.array(b)
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    # Select closest n points
    points = candidates[:n]
    
    # Center it
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points

def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.8
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    # If we didn't get enough, just fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _relax_with_timeout(pos, potential, parameter, timeout_sec=30.0):
    """Relax with a timeout guard."""
    start = time.time()
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        return energy, relaxed_pos
    except Exception as e:
        return np.inf, pos

def _basin_hopping(pos, potential, parameter, max_steps=15, timeout=None, seed=42):
    """
    Basin hopping with adaptive step size.
    """
    rng = np.random.default_rng(seed)
    current_energy, current_pos = relax(pos, potential, parameter)
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(max_steps):
        if timeout and time.time() > timeout:
            break
        
        # Adaptive step size: start larger, decrease
        base_sigma = 0.2
        sigma = base_sigma * np.exp(-step * 0.08)
        
        # Try multiple perturbations per step
        for attempt in range(3):
            if timeout and time.time() > timeout:
                break
            
            noise = rng.normal(0, sigma, size=current_pos.shape)
            perturbed_pos = current_pos + noise
            
            try:
                energy, new_pos = relax(perturbed_pos, potential, parameter)
                if energy < current_energy:
                    current_energy = energy
                    current_pos = new_pos
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = new_pos.copy()
            except Exception:
                continue
    
    return best_energy, best_pos

def _parallel_relax(candidates, potential, parameter, timeout=290.0):
    """Relax multiple candidates in parallel using threads."""
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {}
        for i, pos in enumerate(candidates):
            if time.time() - start_time > timeout - 10:
                break
            future = executor.submit(_relax_with_timeout, pos, potential, parameter)
            future_to_idx[future] = i
        
        for future in concurrent.futures.as_completed(future_to_idx):
            if time.time() - start_time > timeout:
                break
            try:
                energy, relaxed_pos = future.result(timeout=60)
                results.append((energy, relaxed_pos))
            except Exception:
                continue
    
    return results

def construct(instance):
    """
    Find the lowest-energy structure for a Dzugutov cluster.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    timeout = 295.0  # Safety margin before 330s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedral grown cluster (primary template for Dzugutov)
    try:
        p_ico = _generate_icosahedron_grown(size, seed=42)
        candidates.append(p_ico)
    except Exception:
        pass
    
    # 2. Second icosahedral variant with different seed
    try:
        p_ico2 = _generate_icosahedron_grown(size, seed=123)
        candidates.append(p_ico2)
    except Exception:
        pass
    
    # 3. FCC template
    try:
        p_fcc = _generate_fcc(size)
        candidates.append(p_fcc)
    except Exception:
        pass
    
    # 4. Random packings (multiple seeds for diversity)
    for seed in [1, 7, 13, 21, 33]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append(p_rand)
        except Exception:
            pass
    
    # Relax all candidates in parallel
    best_energy = np.inf
    best_pos = None
    
    if candidates:
        results = _parallel_relax(candidates, potential, parameter, timeout=timeout - 20)
        
        for energy, relaxed_pos in results:
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
    
    # If no valid result yet, try sequential relaxation
    if best_pos is None:
        for pos in candidates:
            if time.time() - start_time > timeout - 10:
                break
            try:
                energy, relaxed_pos = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
            except Exception:
                continue
    
    # Refinement: Basin hopping on the best structure
    if best_pos is not None:
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 30:
            max_refine_steps = min(25, int(remaining_time / 3))
            try:
                energy, pos = _basin_hopping(best_pos, potential, parameter,
                                              max_steps=max_refine_steps,
                                              timeout=start_time + timeout - 5,
                                              seed=42)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass
        
        # Additional basin hopping from second-best if available
        if remaining_time > 60:
            # Try basin hopping from a different starting point
            second_best_candidates = []
            for pos in candidates:
                if time.time() - start_time > timeout - 20:
                    break
                try:
                    energy, relaxed_pos = relax(pos, potential, parameter)
                    if energy < best_energy * 1.1:  # Within 10% of best
                        second_best_candidates.append((energy, relaxed_pos))
                except Exception:
                    continue
            
            if second_best_candidates:
                second_best_candidates.sort(key=lambda x: x[0])
                if second_best_candidates[0][0] < best_energy:
                    second_energy, second_pos = second_best_candidates[0]
                    try:
                        energy, pos = _basin_hopping(second_pos, potential, parameter,
                                                      max_steps=10,
                                                      timeout=start_time + timeout - 5,
                                                      seed=99)
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = pos
                    except Exception:
                        pass
    
    if best_pos is None:
        # Fallback: Return one of the generated candidates if relax failed entirely
        if candidates:
            return candidates[0]
        else:
            # Last resort: linear chain
            return np.array([np.array([i, 0, 0]) for i in range(size)])
    
    return best_pos
# EVOLVE-BLOCK-END
