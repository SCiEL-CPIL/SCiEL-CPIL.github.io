"""Sutton-Chen cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_initial_structures(n, kind):
    """Generate diverse initial structures for basin hopping."""
    inits = []
    
    # 1. Random compact cluster
    def _random_compact(n, spacing=0.72, seed=42):
        rng = np.random.default_rng(seed)
        points = [np.zeros(3)]
        radius = spacing * n ** (1.0/3.0) * 0.9
        
        attempts = 0
        while len(points) < n and attempts < n * 200:
            attempts += 1
            p = rng.normal(0, radius/2, 3)
            if np.linalg.norm(p) > radius:
                continue
            # Check distance to existing points
            min_dist = np.inf
            for q in points:
                d = np.linalg.norm(p - q)
                if d < min_dist:
                    min_dist = d
            if min_dist >= 0.6 * spacing:
                points.append(p)
        
        # Fill remaining if necessary
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        
        return np.array(points)

    inits.append(_random_compact(n, seed=1))
    inits.append(_random_compact(n, seed=2))

    # 2. FCC fragment
    def _fcc_positions(n, a=0.72):
        points = []
        basis = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ]
        side = int(np.ceil((n / 4) ** (1/3))) + 2
        candidates = []
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                for k in range(-side, side + 1):
                    for b in basis:
                        p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                        candidates.append(p)
        candidates = np.array(candidates)
        center = candidates.mean(axis=0)
        candidates -= center
        dists = np.linalg.norm(candidates, axis=1)
        idx = np.argsort(dists)[:n]
        return candidates[idx]
        
    try:
        inits.append(_fcc_positions(n))
    except:
        pass

    # 3. Icosahedron-like
    def _icosahedron_positions(n, r=1.0):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r
        
        pts = [np.zeros(3)]
        for v in verts:
            pts.append(v)
        
        if n <= 13:
            return np.array(pts[:n])
            
        # Add outer shell
        def fib_sphere(n_pts, radius):
            pts = []
            golden = (1 + np.sqrt(5)) / 2
            for i in range(n_pts):
                z = 1 - 2 * (i + 0.5) / n_pts
                r = np.sqrt(1 - z*z)
                theta = 2 * np.pi * i / golden
                pts.append([r * np.cos(theta), r * np.sin(theta), z])
            return np.array(pts) * radius
            
        shell1 = fib_sphere(12, 1.8)
        pts.extend([p for p in shell1])
        if n <= 25:
            return np.array(pts[:n])
            
        shell2 = fib_sphere(42, 2.6)
        pts.extend([p for p in shell2])
        if n <= 67:
            return np.array(pts[:n])
            
        shell3 = fib_sphere(92, 3.4)
        pts.extend([p for p in shell3])
        
        return np.array(pts[:n])
        
    try:
        inits.append(_icosahedron_positions(n, r=0.72 * n**(1/3)))
    except:
        pass

    return inits


def _basin_hopping(positions, potential, kind, max_iters=1000, rng=None, time_limit=None):
    """Basin Hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(positions)
    
    # Initial relaxation
    try:
        current_energy, current_positions = relax(positions, potential, kind)
    except:
        return None, None
        
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    start_time = time.time()
    
    for i in range(max_iters):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Generate perturbation
        # Scale step size based on n and temperature-like factor
        # Standard BH uses a fixed step size or adaptive. 
        # For clusters, a step of ~0.1-0.5 is typical.
        step_size = 0.2 
        delta = rng.normal(0, step_size, (n, 3))
        perturbed_positions = current_positions + delta
        
        # Relax the perturbed structure
        try:
            new_energy, new_positions = relax(perturbed_positions, potential, kind)
        except:
            continue
            
        if not np.isfinite(new_energy):
            continue
            
        # Metropolis acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1): # T=0.1 is typical for BH
            current_energy = new_energy
            current_positions = new_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                
    return best_energy, best_positions


def construct(instance):
    """Construct Sutton-Chen cluster using Basin Hopping."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    start_time = time.time()
    time_limit = 250.0  # Leave buffer for 300s total
    
    inits = _generate_initial_structures(size, kind)
    
    best_energy = np.inf
    best_positions = None
    
    # Run Basin Hopping on each initialization
    # Distribute time among initializations
    num_inits = len(inits)
    time_per_init = time_limit / num_inits
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Estimate max_iters based on time remaining and complexity
        # Relaxation cost scales roughly with n^2 * iterations_of_l_bfgs
        # Let's assume ~10-50ms per relaxation step for n<100.
        # So ~20-100 steps per second.
        estimated_steps_per_sec = 50.0
        time_remaining = time_limit - (time.time() - start_time)
        max_iters = int(estimated_steps_per_sec * time_remaining / (num_inits - i))
        
        if max_iters < 10:
            max_iters = 10
            
        try:
            energy, positions = _basin_hopping(init, potential, kind, max_iters=max_iters, time_limit=time_remaining)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
            
    if best_positions is None:
        # Fallback: just relax a random structure
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1/3)
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            best_positions = relax(best_positions, potential, kind)[1]
        except:
            pass

    return best_positions
# EVOLVE-BLOCK-END
