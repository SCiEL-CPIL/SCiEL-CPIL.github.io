import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_icosahedron(n, rng):
    """
    Generate a valid icosahedral cluster of n atoms using divide-and-conquer.
    We build a 13-atom icosahedron (1 center + 12 vertices), relax it,
    and then use it as a building block to assemble larger clusters.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # 1. Create a prototype 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # The Dzugutov potential has a minimum around r=1.0 (actually slightly less, ~0.95-1.0).
    # Let's place vertices at r=1.0.
    proto = np.vstack([np.zeros(3), verts * 1.0])
    
    # Relax the prototype to find the optimal icosahedral geometry
    try:
        _, proto_relaxed = relax(proto, 'dzugutov', None)
    except Exception:
        proto_relaxed = proto
        
    # If relaxation failed or produced something weird, use the ideal one
    if np.linalg.norm(proto_relaxed[0] - proto_relaxed[1]) < 0.5:
        proto_relaxed = proto

    # 2. Assemble larger clusters
    # For n > 13, we can attach more icosahedra or atoms.
    # A simple strategy: Start with one icosahedron, and add atoms one by one
    # at positions that maximize distance from existing atoms but keep them close.
    # Or, better: Use a "growth" approach.
    
    if n <= 13:
        # Just take the first n atoms of the relaxed prototype
        # (Ideally we should pick the most symmetric subset, but for now just slice)
        # Actually, for n < 13, it's better to generate a smaller icosahedron.
        # But let's stick to the prototype for simplicity in this "divide and conquer" seed.
        # If n < 13, we can just return a random compact cluster or a subset.
        # Let's return a subset of the prototype.
        return proto_relaxed[:n]
    
    # For n > 13, we need to add atoms.
    # We will add atoms in a way that mimics the 2nd shell of a Mackay icosahedron.
    # The 2nd shell has 42 atoms.
    # We can approximate this by placing atoms at the midpoints of the edges of the icosahedron
    # and at the centers of the faces.
    
    points = list(proto_relaxed)
    
    # Calculate edge midpoints and face centers of the relaxed prototype
    # Vertices are points[1:]
    v = points[1:]
    
    # Find edges (shortest distances)
    dists = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Get 30 shortest pairs
    idx = np.argsort(dists.flatten())
    edges = []
    seen = set()
    for i in idx:
        i, j = divmod(i, 12)
        if i == j: continue
        pair = (min(i,j), max(i,j))
        if pair not in seen:
            seen.add(pair)
            edges.append(pair)
            if len(edges) == 30:
                break
    
    # Add edge midpoints (scaled slightly outwards to avoid clashes)
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        # Normalize and scale to be slightly further out than the midpoint
        # The midpoint is at r ~ 0.85. We want it at r ~ 1.8?
        # Let's project it to a sphere of radius 1.8
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            mid = mid / norm * 1.8
        else:
            mid = mid * 1.8
        points.append(mid)
        
    # Add face centers
    # We need the 20 faces.
    # A face is defined by 3 vertices.
    # We can find faces by looking for triangles in the graph of edges.
    # This is a bit complex. Let's approximate by adding 20 points on a sphere of radius 1.8
    # that are far from the edge midpoints.
    # Or, simpler: The 20 face centers of an icosahedron are the vertices of a dodecahedron.
    # We can generate a dodecahedron and place its vertices at radius 1.8.
    # Dodecahedron vertices (radius 1):
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    dodeca_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Wait, these are the icosahedron vertices again?
    # No, the dodecahedron vertices are (±1, ±1, ±1) and (0, ±1/phi, ±phi) etc.
    # Let's use the correct dodecahedron vertices.
    dodeca_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ])
    dodeca_verts /= np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    
    for dv in dodeca_verts:
        points.append(dv * 1.8)
        
    # Now we have 1 + 12 + 30 + 20 = 63 atoms.
    # We need exactly n atoms.
    # If n > 63, we add more atoms in a 3rd shell (radius ~ 2.4).
    # If n < 63, we remove atoms from the outermost shell first.
    
    # Sort points by distance from center
    dists_to_center = np.linalg.norm(np.array(points), axis=1)
    order = np.argsort(dists_to_center)
    
    # We want the first n atoms.
    # But we should keep the core (icosahedron) intact.
    # So we take the first n atoms in the sorted order.
    # This will remove outer atoms first.
    
    result = np.array(points)[order[:n]]
    
    # Relax the whole assembly to fix any clashes
    try:
        _, result_relaxed = relax(result, 'dzugutov', None)
    except Exception:
        result_relaxed = result
        
    return result_relaxed

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points)

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    # 4 Icosahedral (divide and conquer)
    for _ in range(4):
        seeds.append(_generate_icosahedron(n, rng))
    # 4 Random
    for _ in range(4):
        seeds.append(_generate_random(n, rng))
        
    best_energy = float('inf')
    best_positions = seeds[0]
    
    # Phase 1: Relax all seeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # Phase 2: Multi-trajectory Basin Hopping
    # We will run 4 parallel basin hopping trajectories from the best 4 structures found in Phase 1.
    
    # Re-sort results to get the best 4
    results = []
    for f in futures:
        energy, positions = f.result()
        results.append((energy, positions))
    results.sort(key=lambda x: x[0])
    
    # Take the best 4
    initial_structures = [res[1] for res in results[:4]]
    
    def _basin_hopping_worker(args):
        n, init_pos, potential, parameter, rng_seed = args
        rng_bh = np.random.default_rng(rng_seed)
        
        current_energy, current_positions = relax(init_pos, potential, parameter)
        best_energy, best_positions = current_energy, current_positions
        
        step_size = 0.5
        max_iters = 100
        
        for i in range(max_iters):
            if _get_time_remaining() < 5.0:
                break
                
            new_positions = current_positions + rng_bh.normal(0, step_size, current_positions.shape)
            energy, relaxed = relax(new_positions, potential, parameter)
            
            if energy < current_energy:
                current_energy = energy
                current_positions = relaxed
                step_size *= 0.95
            else:
                step_size *= 1.05
                if step_size > 2.0:
                    step_size = 2.0
                    
            if current_energy < best_energy:
                best_energy = energy
                best_positions = relaxed
                
        return best_energy, best_positions

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_basin_hopping_worker, (n, s, potential, parameter, n+i)) for i, s in enumerate(initial_structures)]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions

    return best_positions
