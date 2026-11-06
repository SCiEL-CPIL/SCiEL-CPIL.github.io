# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-8:
                continue
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster.
    For n=13: 1 center + 12 vertices.
    For larger n: Add layers of icosahedral symmetry.
    The Dzugutov potential favors icosahedral/polytetrahedral order.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (normalized)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius r1
    # Optimal radius for Dzugutov is around 1.0-1.2 for nearest neighbors
    # Let's use 1.1 as a good starting guess
    if n > 1:
        r1 = 1.1
        points.extend([v * r1 for v in verts])
        
    # If we need more atoms, add a second shell
    # A common extension is to place atoms at the centers of the triangular faces of the icosahedron
    # or simply scale up the icosahedron and add more points on the surface.
    # For simplicity and robustness, we will generate a "spherical shell" of points
    # using Fibonacci sphere distribution if n > 13, centered on the icosahedral directions.
    
    if n > len(points):
        remaining = n - len(points)
        # Use a second shell at radius ~2.2
        r2 = 2.2
        # Generate points on a sphere with icosahedral symmetry bias
        # We can just use random points on the sphere for the outer shell, 
        # but bias them towards the 12 vertices and 20 face centers.
        
        # Face centers of icosahedron
        face_centers = []
        # Indices of vertices forming triangles
        # This is complex to hardcode. Simpler: just use Fibonacci sphere for outer points
        # and let relaxation fix it.
        
        # Actually, a better "polytetrahedral" extension is to use the 13-atom icosahedron
        # and add more 13-atom icosahedra sharing faces? That's too complex for a quick generator.
        
        # Let's just add points on a sphere of radius 2.2 using a Fibonacci spiral
        # This provides good coverage.
        for i in range(remaining):
            # Fibonacci sphere
            i_fib = i + 0.5
            phi_fib = np.acos(1 - 2 * i_fib / remaining)
            theta_fib = np.pi * (1 + 5**0.5) * i_fib
            x = np.sin(phi_fib) * np.cos(theta_fib)
            y = np.sin(phi_fib) * np.sin(theta_fib)
            z = np.cos(phi_fib)
            points.append(np.array([x, y, z]) * r2)
            
    points = np.array(points[:n])
    # Center the cluster
    points = points - np.mean(points, axis=0)
    return points

def _generate_random_polytetrahedral(n, rng):
    """
    Generate a random cluster by placing atoms near icosahedral sites.
    """
    # Start with a small icosahedron
    base = _generate_mackay_icosahedron(min(n, 13))
    points = list(base)
    
    if n > 13:
        # Add remaining atoms randomly, but constrained to be close to existing atoms
        # to maintain a compact, polytetrahedral-like structure
        for _ in range(n - 13):
            # Pick a random existing atom
            idx = rng.integers(0, len(points))
            anchor = points[idx]
            # Place new atom at distance ~1.1 from anchor in random direction
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            new_point = anchor + direction * 1.1
            points.append(new_point)
            
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _basin_hopping(positions, potential, n_iter=50, rng=None):
    """
    Basin Hopping algorithm.
    1. Relax current structure.
    2. Perturb the relaxed structure.
    3. Relax the perturbed structure.
    4. Accept if lower energy (or with Metropolis criterion).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    best_pos = positions.copy()
    best_energy = _dzugutov_energy(best_pos)
    
    # Initial relaxation
    try:
        relaxed = relax(best_pos, potential)
        best_energy = _dzugutov_energy(relaxed)
        best_pos = relaxed
    except:
        pass
        
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    step_size = 0.5
    
    for i in range(n_iter):
        # Perturb
        # Random displacement
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed_pos = current_pos + noise
        
        # Relax perturbed
        try:
            new_pos = relax(perturbed_pos, potential)
            new_energy = _dzugutov_energy(new_pos)
        except:
            new_energy = np.inf
            new_pos = perturbed_pos
            
        # Metropolis criterion
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            # Reduce step size slightly on success? Or keep it?
            # Standard BH often keeps step size constant or anneals it.
            # Let's anneal step size slowly
            step_size = max(0.1, step_size * 0.99)
        else:
            # Accept with probability exp(-dE/T)
            # T should be decreasing
            T = 0.5 * (0.95 ** i)
            if T > 1e-6:
                dE = new_energy - current_energy
                if dE < 0:
                    current_pos = new_pos
                    current_energy = new_energy
                else:
                    if rng.random() < np.exp(-dE / T):
                        current_pos = new_pos
                        current_energy = new_energy
            else:
                # Greedy mode
                if new_energy < current_energy:
                    current_pos = new_pos
                    current_energy = new_energy
                    step_size = max(0.1, step_size * 0.95)

        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Mackay Icosahedron (Physically informed)
    try:
        mackay = _generate_mackay_icosahedron(n)
        initial_structures.append(mackay)
    except:
        pass
    
    # 2. Random Polytetrahedral (Physically informed)
    rng = np.random.default_rng(n)
    for i in range(3):
        try:
            rt = _generate_random_polytetrahedral(n, rng)
            initial_structures.append(rt)
        except:
            pass
            
    # 3. Fallback: Random Compact
    for i in range(2):
        try:
            radius = 1.1 * n**(1.0/3.0) * 0.8
            points = []
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            initial_structures.append(np.array(points))
        except:
            pass

    best_energy = np.inf
    best_positions = None
    
    # Run Basin Hopping from each initial structure
    # We have ~300s total. 6 instances. ~50s per instance.
    # Let's allocate time based on number of instances processed so far.
    # Since we don't know how many calls, we assume ~50s per call is safe.
    
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline:
            break
            
        # Allocate remaining time to this BH run
        time_left = deadline - time.time()
        # Estimate cost of one BH iteration: 1 relax call.
        # Relax call is fast (~10-50ms).
        # Let's just run a fixed number of iterations that fits in time_left.
        # If time_left is large, do more iterations.
        max_iters = int(time_left / 0.1) # 100ms per iter estimate
        max_iters = max(10, min(max_iters, 200)) # Cap at 200 to avoid infinite loops
        
        try:
            bh_energy, bh_pos = _basin_hopping(init_pos, potential, n_iter=max_iters, rng=np.random.default_rng(n+i))
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
        except:
            pass
            
    # If no positions found (shouldn't happen), fallback
    if best_positions is None:
        if initial_structures:
            try:
                best_positions = relax(initial_structures[0], potential)
            except:
                best_positions = initial_structures[0]
        else:
            # Last resort: random
            best_positions = np.random.rand(n, 3) * 2 - 1
            
    return best_positions
# EVOLVE-BLOCK-END
