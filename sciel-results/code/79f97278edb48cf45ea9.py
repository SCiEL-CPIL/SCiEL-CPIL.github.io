"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_mackay_icosahedron(n, scale=1.0):
    """Generate a Mackay icosahedral cluster (13 + 30 + 55 = 98 atoms max)."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (13 atoms: center + 12)
    verts12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts12 /= np.linalg.norm(verts12[0])
    
    points = [np.zeros(3)]
    points.extend((verts12 * scale).tolist())
    
    if n <= 13:
        return np.array(points[:n])
    
    # Second shell: 30 atoms at edge midpoints of the icosahedron
    # Find edges (pairs of vertices at minimum distance)
    d_min = np.inf
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts12[i] - verts12[j])
            if d < d_min - 1e-5:
                d_min = d
                edges = [(i, j)]
            elif abs(d - d_min) < 1e-5:
                edges.append((i, j))
    
    # Add edge midpoints (second shell)
    for i, j in edges:
        mid = (verts12[i] + verts12[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * scale * 1.5  # push outward
        points.append(mid)
    
    if n <= len(points):
        return np.array(points[:n])
    
    # Third shell: 55 atoms (Mackay icosahedron completion)
    # Face centers of the icosahedron (20 faces)
    faces = [
        [0, 4, 8], [0, 8, 1], [0, 1, 5], [0, 5, 4],
        [1, 8, 9], [1, 9, 2], [1, 2, 6], [1, 6, 5],
        [2, 9, 10], [2, 10, 3], [2, 3, 7], [2, 7, 6],
        [3, 10, 11], [3, 11, 0], [3, 0, 4], [3, 4, 7],
        [4, 5, 6], [4, 6, 7], [4, 7, 8], [4, 8, 9],
        [5, 6, 7], [5, 7, 8], [5, 8, 9], [5, 9, 10],
        [6, 7, 8], [6, 8, 9], [6, 9, 10], [6, 10, 11],
        [7, 8, 9], [7, 9, 10], [7, 10, 11], [7, 11, 0],
        [8, 9, 10], [8, 10, 11], [8, 11, 0], [8, 0, 1],
        [9, 10, 11], [9, 11, 0], [9, 0, 1], [9, 1, 2],
        [10, 11, 0], [10, 0, 1], [10, 1, 2], [10, 2, 3],
        [11, 0, 1], [11, 1, 2], [11, 2, 3], [11, 3, 4]
    ]
    # Only use unique faces (20)
    seen = set()
    for face in faces:
        key = tuple(sorted(face))
        if key not in seen:
            seen.add(key)
            centroid = (verts12[face[0]] + verts12[face[1]] + verts12[face[2]]) / 3.0
            centroid = centroid / np.linalg.norm(centroid) * scale * 2.0
            points.append(centroid)
    
    if n <= len(points):
        return np.array(points[:n])
    
    # If still need more, add random atoms in outer shell
    rng = np.random.default_rng(42 + n)
    current = np.array(points)
    min_dist = 0.7
    while len(current) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = rng.uniform(scale * 2.2, scale * 2.8)
        p = v * r
        dists = np.linalg.norm(current - p, axis=1)
        if np.min(dists) >= min_dist:
            current = np.vstack([current, [p]])
        elif len(current) < n and rng.random() < 0.3:
            current = np.vstack([current, [p]])
    
    return current[:n]

def _generate_fcc(n, scale=1.0):
    """Generate a truncated FCC cluster."""
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * scale
    
    N = int(np.ceil(n ** (1/3.0))) + 2
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    points = candidates[:n].copy()
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points

def _generate_random_packing(n, seed, min_dist=0.8):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.2 * n ** (1.0 / 3.0)
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_dist:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _generate_deformed_icosahedron(n, seed, deformation=0.3):
    """Generate a deformed icosahedral cluster."""
    base = _generate_mackay_icosahedron(n, scale=1.0)
    rng = np.random.default_rng(seed)
    
    # Apply small random deformations
    noise = rng.normal(0, deformation, size=base.shape)
    deformed = base + noise
    
    # Ensure minimum distance
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(deformed[i] - deformed[j])
            if d < 0.55:
                # Push apart
                mid = (deformed[i] + deformed[j]) / 2.0
                diff = deformed[i] - deformed[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = rng.normal(size=3)
                diff /= np.linalg.norm(diff)
                push = (0.55 - d) / 2.0 + 0.05
                deformed[i] += diff * push
                deformed[j] -= diff * push
    
    return deformed

def construct(instance):
    """
    Parallel exploration of diverse structural templates with basin hopping refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    timeout = 290.0  # Safety margin before 330s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Mackay icosahedron (primary template for polytetrahedral order)
    try:
        candidates.append(_generate_mackay_icosahedron(size, scale=1.0))
    except Exception:
        pass
    
    # 2. Deformed icosahedra (multiple seeds)
    for seed in [10, 20, 30]:
        try:
            candidates.append(_generate_deformed_icosahedron(size, seed, deformation=0.2))
        except Exception:
            pass
    
    # 3. FCC template
    try:
        candidates.append(_generate_fcc(size, scale=1.0))
    except Exception:
        pass
    
    # 4. Random packings (multiple seeds)
    for seed in [1, 2, 3, 4]:
        try:
            candidates.append(_generate_random_packing(size, seed))
        except Exception:
            pass
    
    # Deduplicate (remove very similar structures)
    unique_candidates = []
    for c in candidates:
        if len(unique_candidates) == 0:
            unique_candidates.append(c)
        else:
            # Check if this candidate is too similar to any existing
            is_unique = True
            for uc in unique_candidates:
                if len(c) != len(uc):
                    is_unique = False
                    break
                # Compare using sorted distances (crude but fast)
                c_dists = np.sort(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2).flatten())
                uc_dists = np.sort(np.linalg.norm(uc[:, None, :] - uc[None, :, :], axis=2).flatten())
                if np.mean(np.abs(c_dists - uc_dists)) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                unique_candidates.append(c)
    
    candidates = unique_candidates
    
    best_energy = np.inf
    best_pos = None
    lock = None  # Simple approach: collect results and pick best
    
    # Phase 1: Parallel relaxation of all candidates
    def _relax_candidate(pos):
        try:
            energy, relaxed = relax(pos, potential, parameter)
            return (energy, relaxed)
        except Exception:
            return (np.inf, None)
    
    if len(candidates) > 0:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_candidate, c): i for i, c in enumerate(candidates)}
            for future in as_completed(futures):
                if time.time() - start_time > timeout - 60:
                    break
                try:
                    energy, relaxed = future.result()
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed
                except Exception:
                    continue
    
    # Phase 2: Basin hopping refinement from best structure
    if best_pos is not None:
        rng = np.random.default_rng(size + 100)
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        # Adaptive basin hopping with multiple restarts
        max_refine_steps = 30
        sigma_start = 0.3
        sigma_end = 0.1
        
        for step in range(max_refine_steps):
            if time.time() - start_time > timeout - 10:
                break
            
            # Linearly decrease perturbation size
            sigma = sigma_start + (sigma_end - sigma_start) * (step / max_refine_steps)
            noise = rng.normal(0, sigma, size=current_pos.shape)
            perturbed_pos = current_pos + noise
            
            try:
                energy, new_pos = relax(perturbed_pos, potential, parameter)
                # Metropolis-like acceptance: always accept if better, sometimes accept if worse
                if energy < current_energy:
                    current_energy = energy
                    current_pos = new_pos
                    best_energy = current_energy
                    best_pos = current_pos
                elif rng.random() < 0.1:  # 10% chance to accept worse for exploration
                    current_energy = energy
                    current_pos = new_pos
            except Exception:
                continue
    
    # Phase 3: Final polish - try a few more perturbations from the global best
    if best_pos is not None:
        rng2 = np.random.default_rng(size + 200)
        for _ in range(5):
            if time.time() - start_time > timeout - 3:
                break
            noise = rng2.normal(0, 0.05, size=best_pos.shape)
            perturbed = best_pos + noise
            try:
                energy, new_pos = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = new_pos
            except Exception:
                continue
    
    if best_pos is None:
        # Fallback
        if candidates:
            return candidates[0]
        else:
            return np.array([np.array([i * 0.6, 0, 0]) for i in range(size)])

    return best_pos
# EVOLVE-BLOCK-END
