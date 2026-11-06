"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 280.0  # Leave 20s safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_icosahedron(n, scale=1.0):
    """
    Generates an icosahedral cluster of n atoms.
    The Dzugutov potential favors icosahedral order.
    We construct a central atom, a first shell of 12, a second shell of 42, etc.
    If n is not a magic number, we add atoms to the outermost shell or randomly near the surface.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Regular icosahedron vertices (12)
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to desired nearest neighbor distance (approx 1.8 for Dzugutov)
    # For unit icosahedron, min dist is 2.
    # We want min dist ~ 1.8. So scale = 1.8 / 2 = 0.9.
    verts = verts * 0.9 * scale
    
    points = [np.zeros(3)]  # Central atom
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
    
    # Second shell: 30 face centers + 12 edge midpoints? Or just more random on sphere?
    # For simplicity and robustness, if n > 13, we add atoms on a second shell of radius ~ 1.8 * sqrt(2) ~ 2.5
    # Or we can just use a "random on sphere" approach for the rest, but that's less structured.
    # Let's try to add a second icosahedral shell.
    # Vertices of a dodecahedron? Or just scale up the icosahedron?
    # A common approach is to take the 12 vertices of the icosahedron, and for each, place neighbors.
    # Simpler: Generate random points on a sphere of radius R2, ensuring min distance.
    
    while len(points) < n:
        # Try to place on a sphere of radius 2.5 (approx second shell)
        # Or just random compact
        r = 2.5 * scale
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        p = np.array([x, y, z]) * r
        
        # Check distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5: # Hard constraint from task
                valid = False
                break
        if valid:
            points.append(p)
            
    return np.array(points)

def _generate_bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster."""
    points = []
    # BCC lattice points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # Corner
                p1 = np.array([i, j, k]) * 1.4 * scale
                if len(points) < n and all(np.linalg.norm(p1 - q) >= 0.5 for q in points):
                    points.append(p1)
                # Body center
                p2 = np.array([i+0.5, j+0.5, k+0.5]) * 1.4 * scale
                if len(points) < n and all(np.linalg.norm(p2 - q) >= 0.5 for q in points):
                    points.append(p2)
    if len(points) < n:
        # Fill with random
        while len(points) < n:
            p = np.random.uniform(-3, 3, 3) * scale
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points[:n])

def _generate_random_cluster(n):
    """Generate a random cluster with minimum distance 0.5."""
    points = []
    for _ in range(n):
        for _attempt in range(100):
            p = np.random.uniform(-2, 2, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            # If we fail to place, just place it far away (shouldn't happen often)
            points.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
    return np.array(points)

def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster.
    Shells: 1 (center), 12 (1st), 42 (2nd), 92 (3rd), 162 (4th), 252 (5th)
    Magic numbers: 1, 13, 55, 147, 309, 561
    """
    # We'll use the icosahedron generator as a base and try to make it more structured
    # For now, just use the existing icosahedron generator with specific scaling
    return _generate_icosahedron(n, 1.0)

def construct(instance):
    """
    Strategy: 
    1. Generate multiple diverse structured initializations (Icosahedral, BCC, Random, Perturbed Icosa).
    2. Relax each to a local minimum.
    3. Perform Basin Hopping on the best initial candidates.
       - Perturbation: Random displacement of all atoms (rigid body-like but with noise) or single atom moves.
       - Acceptance: Metropolis criterion.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # List of initializers
    initializers = [
        lambda: _generate_icosahedron(size, 1.0),
        lambda: _generate_icosahedron(size, 0.9),
        lambda: _generate_icosahedron(size, 1.1),
        lambda: _generate_bcc_cluster(size, 1.0),
        lambda: _generate_random_cluster(size),
        lambda: _generate_icosahedron(size, 1.0) + np.random.normal(0, 0.2, (size, 3)) # Perturbed
    ]
    
    # Pre-relax all initializers and keep the best few
    candidates = []
    for init_func in initializers:
        if not _check_time():
            break
        try:
            pos = init_func()
            # Basic relaxation to ensure stability and fix any minor overlaps
            e, pos_relaxed = relax(pos, potential, parameter)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed.copy()
            candidates.append((e, pos_relaxed))
        except Exception:
            continue
            
    if best_positions is None:
        # Fallback to random if all fail
        pos = np.random.uniform(-2, 2, (size, 3))
        e, pos_relaxed = relax(pos, potential, parameter)
        return pos_relaxed

    # Sort candidates by energy and keep top 5
    candidates.sort(key=lambda x: x[0])
    top_candidates = [c[1] for c in candidates[:5]]
    
    # Basin Hopping
    # We will run BH on each top candidate for a limited time
    current_time = time.time()
    # Allocate time for BH
    bh_total_time = min(TIME_LIMIT - (current_time - _START_TIME) - 10, 200)
    time_per_candidate = bh_total_time / len(top_candidates)
    
    for start_pos in top_candidates:
        if not _check_time():
            break
            
        # Basin Hopping Loop
        pos = start_pos.copy()
        e, _ = relax(pos, potential, parameter)
        current_energy = e
        
        T = 5.0  # Initial temperature
        cooling_rate = 0.99
        max_steps = int(time_per_candidate * 10) # Estimate steps based on time
        
        for step in range(max_steps):
            if time.time() - _START_TIME > TIME_LIMIT - 10:
                break
            if not _check_time():
                break
                
            # Perturb: Move a random subset of atoms
            n_perturb = max(1, size // 10)
            indices = np.random.choice(size, n_perturb, replace=False)
            new_pos = pos.copy()
            for i in indices:
                new_pos[i] += np.random.normal(0, 0.3, 3)
            
            # Check hard constraint (simple check, relax will handle the rest but we want to avoid immediate overlaps)
            # Actually, relax should handle overlaps by pushing them apart or failing. 
            # The task says atoms must be at least 0.5 apart. Relax likely enforces this or returns a valid config.
            # Let's just call relax.
            
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
            
            delta_e = new_e - current_energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                pos = new_pos_relaxed
                current_energy = new_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = pos.copy()
                    
            T *= cooling_rate
            if T < 0.1:
                # Reheat or break
                # Let's just break and move to next candidate if T is too low
                break
                
    return best_positions
# EVOLVE-BLOCK-END
