"""
Rewritten program for Dzugutov cluster global minima.
Focuses on icosahedral/polytetrahedral structures with diverse initial conditions,
relaxed in parallel with time management and a final polish pass.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit to stop

def _get_elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0

def _remaining_time():
    return max(0.0, 300.0 - _DEADLINE_MARGIN - _get_elapsed())

def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster with shells."""
    # Mackay icosahedron shells: 1, 12, 42, 92, 140, ...
    # Shell radii: r1, r1*sqrt(5), r1*sqrt(15), ...
    # Actually for Mackay: shells at distances r, r*sqrt(5), r*sqrt(15), etc.
    # But let's use a simpler approach: place atoms in icosahedral shells
    
    phi = (1 + np.sqrt(5)) / 2
    base_r = 1.05  # Equilibrium-ish distance
    
    # Icosahedron vertices (12 vertices)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Dodecahedron vertices (20 vertices) for second shell
    # The dodecahedron vertices are at the face centers of the icosahedron
    # Or we can use the dual: dodecahedron has 20 vertices
    # Simpler: use golden ratio to generate second shell points
    # Second shell: 30 points at edges of icosahedron
    # Third shell: 60 points
    
    points = []
    
    # Center atom
    if n >= 1:
        points.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: 12 atoms (icosahedron vertices)
    if n >= 13:
        shell1 = ico_verts * base_r
        for v in shell1:
            if len(points) < n:
                points.append(v)
    
    # Second shell: 30 atoms (midpoints of icosahedron edges, projected to sphere)
    if n >= 43:
        # Icosahedron has 30 edges
        # Find all pairs of vertices that are connected (distance = 2)
        edge_midpoints = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.1:  # Connected vertices
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints.append(mid)
        
        # Deduplicate
        unique_mids = []
        for m in edge_midpoints:
            if all(np.linalg.norm(m - u) > 0.01 for u in unique_mids):
                unique_mids.append(m)
        
        shell2_r = base_r * np.sqrt(5)  # Approximate second shell radius
        for m in unique_mids:
            if len(points) < n:
                points.append(m * shell2_r)
    
    # Third shell and beyond: add atoms on a sphere
    if len(points) < n:
        shell3_r = base_r * np.sqrt(15)
        # Generate points on sphere using Fibonacci sphere or random
        remaining = n - len(points)
        # Use a grid on sphere
        num_lat = int(np.sqrt(remaining * 4 / np.pi)) + 1
        num_lon = num_lat * 2
        for lat in range(-num_lat, num_lat + 1):
            for lon in range(num_lon):
                if len(points) >= n:
                    break
                theta = (lat + 0.5) * np.pi / (num_lat + 1)
                phi_angle = lon * 2 * np.pi / num_lon
                p = np.array([
                    np.sin(theta) * np.cos(phi_angle),
                    np.sin(theta) * np.sin(phi_angle),
                    np.cos(theta)
                ]) * shell3_r
                # Check minimum distance
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    points.append(p)
    
    # If still not enough, add random points
    while len(points) < n:
        u = rng.uniform(-1, 1)
        v = rng.uniform(0, 2 * np.pi)
        r = base_r * (1.0 + rng.uniform(0, 0.5))
        p = np.array([
            np.sqrt(1 - u**2) * np.cos(v) * r,
            np.sqrt(1 - u**2) * np.sin(v) * r,
            u * r
        ])
        points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _generate_dense_random(n, rng):
    """Generate a dense random cluster using rejection sampling."""
    # Estimate radius based on packing density
    # For dense packing, volume per atom ~ 1.5-2.0
    R = (3.0 * n / (4.0 * np.pi) * 1.8) ** (1.0 / 3.0) * 1.05
    
    points = []
    max_attempts = 5000 * n
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            # Check minimum distance using numpy for speed
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                d = np.linalg.norm(arr - p, axis=1)
                if np.min(d) >= 0.55:
                    points.append(p)
        attempts += 1
    
    # Fallback if not enough points
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _generate_bcc_chunk(n, rng):
    """Generate a BCC lattice chunk (alternative to FCC)."""
    R = (3.0 * n / (4.0 * np.pi) * 14.0) ** (1.0 / 3.0) * 1.1
    a = 2.3  # Lattice constant
    
    points = []
    max_idx = int(R / a) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: corner + center
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                if len(points) > 0:
                    d = np.linalg.norm(points - p, axis=1)
                    if np.min(d) >= 0.5:
                        points = np.vstack([points, [p]])
                else:
                    points = np.array([[p]])
    
    points -= np.mean(points, axis=0)
    return points

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential = args
    try:
        energy, relaxed_positions = relax(positions, potential)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def _polish_worker(args):
    """Worker for final polish with tighter tolerance."""
    positions, potential = args
    try:
        # Call relax again for additional refinement
        energy, relaxed_positions = relax(positions, potential)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def construct(instance):
    """Find low-energy Dzugutov cluster configurations."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    # Check deadline
    if _remaining_time() < 5.0:
        # Return a simple random cluster if we're out of time
        rng = np.random.default_rng(size)
        R = (3.0 * size / (4.0 * np.pi) * 1.8) ** (1.0 / 3.0)
        points = rng.uniform(-R, R, (size, 3))
        points -= np.mean(points, axis=0)
        return points
    
    rng = np.random.default_rng(size)
    
    initial_clusters = []
    
    # 1. Mackay icosahedral (most important for Dzugutov)
    ico = _generate_mackay_icosahedron(size, rng)
    initial_clusters.append(ico)
    
    # 2. Dense random clusters (3 variants)
    for _ in range(3):
        random_cluster = _generate_dense_random(size, rng)
        initial_clusters.append(random_cluster)
    
    # 3. BCC chunk
    bcc = _generate_bcc_chunk(size, rng)
    if bcc is not None:
        initial_clusters.append(bcc)
    
    # 4. Another icosahedral with different seed
    ico2 = _generate_mackay_icosahedron(size, rng)
    initial_clusters.append(ico2)
    
    # Relax all clusters in parallel
    args_list = [(cluster, potential) for cluster in initial_clusters]
    
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    # Use a thread pool for parallel relaxation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, args) for args in args_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            # Check deadline
            if _remaining_time() < 15.0:
                break
    
    # Final polish pass if time allows
    if _remaining_time() > 10.0 and best_energy < float('inf'):
        try:
            energy, polished = relax(best_positions, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = polished
        except Exception:
            pass
    
    return best_positions
