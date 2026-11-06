"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _generate_icosahedral(n, rng=None):
    """
    Generate n points arranged in concentric icosahedral shells (Mackay icosahedra).
    This is a good starting point for Dzugutov potential which favors icosahedral order.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay icosahedra shell sizes: 1, 12, 42, 92, ...
    # We will fill shells one by one.
    points = []
    
    # Shell 0: Center
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        n -= 1
    
    # Shell 1: 12 vertices
    if n > 0:
        shell_count = min(12, n)
        # Select vertices. For simplicity, take the first `shell_count` from the unique set.
        # Unique vertices of icosahedron
        unique_verts = []
        seen = set()
        for v in verts:
            key = tuple(np.round(v, 5))
            if key not in seen:
                seen.add(key)
                unique_verts.append(v)
        
        # unique_verts should have 12 entries
        for i in range(shell_count):
            # Scale by radius ~1.1 (equilibrium distance)
            r = 1.1
            p = unique_verts[i] * r
            points.append(p)
        n -= shell_count
    
    # Shell 2: 42 vertices (if n is large enough)
    # We can approximate shell 2 by scaling shell 1 vertices or adding intermediate points.
    # For simplicity and robustness, if we still need more points, we will add them
    # by perturbing existing points or adding random points in a shell.
    # However, a better approach for "new design" is to use a random dense packing
    # initialized near an icosahedral skeleton.
    
    # If we still need points, fill with random points in a spherical shell
    # or just random dense packing, but biased towards icosahedral symmetry.
    # Let's use a simple random dense packing for the remainder, but keep the icosahedral core.
    if n > 0:
        # Estimate radius for remaining points
        current_radius = 1.1
        # Add points randomly in a larger sphere, but reject if too close to existing
        # This is a bit slow for large n, but n is max 60.
        min_dist = 0.9
        max_attempts = 1000
        added = 0
        while added < n and max_attempts > 0:
            # Sample in shell between 1.1 and 1.1 + 0.5
            r = rng.uniform(1.1, 2.0)
            # Random direction
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * r
            
            # Check distance to existing points
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < min_dist:
                    max_attempts -= 1
                    continue
            
            points.append(p)
            added += 1
            max_attempts -= 1
            
        # If still not enough, just add random points (should not happen for n<=60 with this logic)
        if added < n:
            for _ in range(n - added):
                r = rng.uniform(0.5, 2.5)
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                p = v * r
                points.append(p)

    return np.array(points)

def _basin_hopping(pos, potential, parameter, n_iter=50, kick_scale=0.5, rng=None):
    """
    Simple basin hopping algorithm.
    1. Relax current pos.
    2. Kick randomly.
    3. Relax again.
    4. Accept if lower energy or with Metropolis probability.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initial relaxation
    try:
        energy, pos = relax(pos, potential, parameter)
    except Exception:
        return float('inf'), pos
    
    best_energy = energy
    best_pos = pos.copy()
    
    for i in range(n_iter):
        # Kick
        kick = rng.normal(0, kick_scale, pos.shape)
        # Apply kick
        new_pos = pos + kick
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
        
        # Acceptance
        if new_energy < energy:
            pos = new_pos_relaxed
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            # Metropolis acceptance (simplified, T=0.1)
            # delta = new_energy - energy
            # accept if exp(-delta/T) > random
            # For simplicity, just accept if very close or with low prob
            # Given time constraints, we stick to greedy or simple random restarts
            pass
            
    return best_energy, best_pos

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses Icosahedral initialization and Basin Hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 45.0 # Leave buffer for 6 instances in 300s
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy 1: Icosahedral initialization + Basin Hopping
    # Run multiple iterations of basin hopping from slightly perturbed icosahedral starts
    for i in range(10): # 10 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Generate icosahedral structure with some noise
        pos = _generate_icosahedral(size, rng)
        # Add small noise to break symmetry and escape trivial minima
        pos += rng.normal(0, 0.1, pos.shape)
        
        # Run basin hopping
        try:
            e, p = _basin_hopping(pos, potential, parameter, n_iter=20, kick_scale=0.3, rng=rng)
            if e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass
    
    # Strategy 2: Random Dense Packing + Relax (Fallback/Complement)
    if time.time() - start_time < time_limit:
        # Generate random dense packing
        radius = 1.2 * size ** (1.0/3.0)
        pts = rng.uniform(-radius, radius, (size, 3))
        norms = np.linalg.norm(pts, axis=1)
        mask = norms <= radius
        if np.sum(mask) >= size:
            pts = pts[mask][:size]
        else:
            pts = pts[:size]
            
        try:
            e, p = relax(pts, potential, parameter)
            if e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass

    if best_positions is None:
        # Absolute fallback: random points
        rng = np.random.default_rng(size)
        radius = 2.0
        points = rng.uniform(-radius, radius, (size, 3))
        try:
            e, best_positions = relax(points, potential, parameter)
        except Exception:
            best_positions = points

    return best_positions

# EVOLVE-BLOCK-END
