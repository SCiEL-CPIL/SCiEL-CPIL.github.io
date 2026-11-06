"""Morse cluster global minima search with structured seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            u = rho * (1.0 - r)
            eu = np.exp(u)
            e += eu * (eu - 2.0)
    return e


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient for a configuration."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        # Fallback: numerical gradient
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-7
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad


def _min_distance(pos):
    """Compute minimum pairwise distance in a configuration."""
    n = len(pos)
    if n < 2:
        return float('inf')
    tree = cKDTree(pos)
    pairs = tree.query_pairs(0.5 + 1e-6)
    min_d = float('inf')
    for i, j in pairs:
        d = np.linalg.norm(pos[i] - pos[j])
        min_d = min(min_d, d)
    if min_d == float('inf'):
        # No pairs within 0.5, check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d)
    return min_d


def _fix_min_distance(pos, min_dist=0.5, max_iter=100):
    """Iteratively separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(min_dist + 1e-6)
        moved = False
        for i, j in pairs:
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                # Push apart
                direction = pos[i] - pos[j]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction /= np.linalg.norm(direction)
                shift = (min_dist - d + 1e-3) / 2.0
                pos[i] += shift * direction
                pos[j] -= shift * direction
                moved = True
        if not moved:
            break
    return pos


def _generate_fcc_cluster(n, rng, noise=0.05):
    """
    Generate an FCC cluster of approximately n atoms, centered.
    Uses a cubic block of FCC lattice points, taking the n closest to the center.
    """
    # FCC basis vectors
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Estimate side length to get enough points
    # Volume per atom in FCC (a=1) is 0.25. So V = 0.25 * n.
    # Side of cube approx (0.25 * n)^(1/3)
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    
    pts = np.array(pts)
    # Center the cloud
    pts -= np.mean(pts, axis=0)
    
    # Sort by distance to center and take n closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    
    # Add small noise to break symmetry
    cluster += rng.normal(0, noise, size=cluster.shape)
    
    return cluster


def _generate_hcp_cluster(n, rng, noise=0.05):
    """
    Generate an HCP cluster of approximately n atoms, centered.
    """
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    
    pts = []
    # Estimate number of layers
    radius = np.sqrt(n / (np.pi * 0.866))
    num_layers = int(np.ceil(2 * radius / (c/2))) + 2
    
    for layer_idx in range(-num_layers//2, num_layers//2 + 1):
        z = layer_idx * (c / 2.0)
        # Shift x,y for alternating layers
        if layer_idx % 2 == 0:
            offset = np.array([0.0, 0.0])
        else:
            offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
            
        # Generate hexagonal points in this layer within radius
        for i in range(-int(radius), int(radius) + 1):
            for j in range(-int(radius), int(radius) + 1):
                # Hexagonal to Cartesian
                x = a * (i + j/2.0) + offset[0]
                y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                p = np.array([x, y, z])
                if np.linalg.norm(p[:2]) <= radius:
                    pts.append(p)
                    
    if len(pts) < n:
        # Fallback if generation was too sparse
        return _generate_fcc_cluster(n, rng, noise)
        
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, noise, size=cluster.shape)
    
    return cluster


def _generate_icosahedron_cluster(n, rng, noise=0.05):
    """
    Generate a Mackay icosahedron cluster.
    For n <= 13, use the 13-atom icosahedron.
    For n > 13, generate a larger structure by adding shells or falling back to FCC/HCP if too complex.
    Here we implement a simple version: 
    If n is close to a magic number, use it. Otherwise, generate a spherical shell structure.
    """
    if n <= 13:
        # 13-atom icosahedron
        phi = (1 + np.sqrt(5)) / 2.0
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
        pts = np.vstack([np.zeros(3), verts])
        pts = pts[:n]
        pts += rng.normal(0, noise, size=pts.shape)
        return pts
    
    # For larger n, a simple spherical shell approach is often competitive and robust
    # Let's use a Fibonacci sphere for the outer shell and a core
    # Or just use FCC/HCP as fallback for simplicity and robustness in this context
    # Actually, let's try a simple spherical shell: 
    # Core: 1 atom
    # Shell 1: 12 atoms
    # Shell 2: 42 atoms
    # Shell 3: 92 atoms
    # If n is not a magic number, we take the largest magic number <= n and add random points?
    # Or just generate points on a sphere of radius ~ (n)^(1/3)
    
    # For robustness and to avoid complex geometry bugs, let's use a "spherical" init
    # which is often good for medium/long range.
    return _generate_spherical_cluster(n, rng, noise)


def _generate_spherical_cluster(n, rng, noise=0.05):
    """Generate points on a sphere using Fibonacci lattice."""
    # Estimate radius
    # Volume ~ n * 0.25 (approximate packing)
    # R ~ (3n/(4pi*0.25))^(1/3) ?
    # Simpler: R ~ n^(1/3)
    R = 1.0 * (n ** (1.0/3.0))
    
    # Fibonacci sphere
    N = n
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        z = 1.0 - (i / float(N - 1)) * 2.0  # z goes from 1 to -1
        radius_at_z = np.sqrt(1.0 - z * z) * R  # radius at z
        theta = golden_angle * i
        
        x = np.cos(theta) * radius_at_z
        y = np.sin(theta) * radius_at_z
        z_coord = z * R
        
        points.append([x, y, z_coord])
        
    points = np.array(points)
    points += rng.normal(0, noise, size=points.shape)
    return points


def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration (Simple Cubic)."""
    shell = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return _fix_min_distance(points)


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.55 * spacing
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return _fix_min_distance(points)


def _optimize_local(pos, rho, deadline, max_iter=300):
    """Run L-BFGS-B optimization on a single configuration."""
    try:
        def fun_and_grad(x):
            if time.time() > deadline - 5:
                raise ValueError("deadline")
            p = x.reshape(-1, 3)
            e, g = _morse_energy_grad(p, rho)
            return e, g.reshape(-1)
        
        x0 = pos.flatten()
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
        )
        
        if result.fun < float('inf'):
            pos_opt = result.x.reshape(-1, 3)
            pos_opt = _fix_min_distance(pos_opt)
            e_opt = _morse_energy(pos_opt, rho)
            if _min_distance(pos_opt) >= 0.5:
                return e_opt, pos_opt
    except Exception:
        pass
    
    # Fallback: use relaxed position
    try:
        e, p = relax(pos.copy(), 'morse', rho)
        p = _fix_min_distance(p)
        if _min_distance(p) >= 0.5:
            return _morse_energy(p, rho), p
    except Exception:
        pass
    
    return float('inf'), None


def _basin_hop(pos, rho, deadline, n_iter=10, step_size=0.1):
    """Simple basin hopping algorithm."""
    best_e, best_pos = _optimize_local(pos, rho, deadline, max_iter=200)
    
    if best_pos is None:
        return float('inf'), None
    
    for i in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        # Perturb
        perturb = np.random.normal(0, step_size, best_pos.shape)
        new_pos = best_pos + perturb
        new_pos = _fix_min_distance(new_pos)
        
        # Optimize
        e_new, pos_new = _optimize_local(new_pos, rho, deadline, max_iter=150)
        
        if pos_new is not None and e_new < best_e:
            best_e, best_pos = e_new, pos_new
    
    return best_e, best_pos


def construct(instance):
    """Best of multiple initial structures with basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Structured seeds (High quality starting points)
    inits.append(_generate_fcc_cluster(size, rng))
    inits.append(_generate_hcp_cluster(size, rng))
    inits.append(_generate_icosahedron_cluster(size, rng))
    inits.append(_generate_spherical_cluster(size, rng))
    
    # 2. Random/Lattice seeds (Diversity)
    inits.append(_lattice_init(size, rng, 1.0))
    inits.append(_random_packed_init(size, rng, 1.0))
    
    best_energy = float('inf')
    best_positions = None
    best_init_idx = -1
    
    # Phase 1: Quick local optimization on all inits
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        e, pos = _optimize_local(init_pos, rho, deadline, max_iter=200)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos
            best_init_idx = idx
    
    # Phase 2: Basin hopping from best configurations
    if best_positions is not None and time.time() < deadline - 30:
        # Basin hop from the best
        e_bh, pos_bh = _basin_hop(best_positions.copy(), rho, deadline, 
                                   n_iter=8, step_size=0.15)
        if pos_bh is not None and e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh
        
        # Also try basin hopping from second-best init (if different)
        # Find the second best init index
        second_idx = -1
        min_second = float('inf')
        for idx, init_pos in enumerate(inits):
            if idx == best_init_idx:
                continue
            # We didn't store all energies, so we just pick a different one
            # Ideally we'd re-optimize, but for time, let's just pick a different structured one
            if idx != best_init_idx:
                second_idx = idx
                break
        
        if second_idx != -1 and time.time() < deadline - 15:
            # Re-optimize the second best to get a proper start for BH
            e2, pos2 = _optimize_local(inits[second_idx], rho, deadline, max_iter=100)
            if pos2 is not None:
                e_bh2, pos_bh2 = _basin_hop(pos2.copy(), rho, deadline,
                                             n_iter=5, step_size=0.12)
                if pos_bh2 is not None and e_bh2 < best_energy:
                    best_energy = e_bh2
                    best_positions = pos_bh2
    
    # Final polish
    if best_positions is not None and time.time() < deadline - 5:
        e_final, pos_final = _optimize_local(best_positions.copy(), rho, deadline, max_iter=300)
        if pos_final is not None and e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    
    if best_positions is None:
        # Fallback
        try:
            _, positions = relax(inits[0], potential, parameter)
            positions = _fix_min_distance(positions)
            return positions
        except Exception:
            return inits[0]
    
    return best_positions
# EVOLVE-BLOCK-END
