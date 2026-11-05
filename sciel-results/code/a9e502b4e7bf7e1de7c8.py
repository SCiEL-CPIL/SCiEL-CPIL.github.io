"""Dzugutov cluster global minima search using topological polytetrahedral seeds,
barrier-augmented L-BFGS refinement, and basin hopping."""
import numpy as np
from scipy.optimize import minimize
import time

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 5000)
    energies = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri < a:
            energies[i] = A * (ri ** (-m) - B) * np.exp(c / (ri - a))
        elif ri < b:
            energies[i] = B * np.exp(d / (ri - b))
        else:
            energies[i] = 0.0
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for n atoms. Vectorized pair computation."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    
    r_safe = np.maximum(r, 1e-10)
    
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    
    energy = np.zeros_like(r)
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        energy[mask_a] = (A * (r_a**(-m) - B)) * np.exp(c / (r_a - a))
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        energy[mask_b] = B * np.exp(d / (r_b - b))
    
    return np.sum(energy)

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy. Vectorized."""
    n = positions.shape[0]
    if n < 2:
        return np.zeros_like(positions)
    
    grad = np.zeros_like(positions)
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    r_safe = np.maximum(r, 1e-10)
    
    dVdr = np.zeros_like(r)
    
    mask_a = r < a
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        exp_term = np.exp(c / (r_a - a))
        dVdr[mask_a] = A * (-m * r_a**(-m-1) * exp_term + (r_a**(-m) - B) * exp_term * (-c / (r_a - a)**2))
    
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        dVdr[mask_b] = B * np.exp(d / (r_b - b)) * (-d / (r_b - b)**2)
    
    unit_vec = diff / r_safe[:, np.newaxis]
    
    np.add.at(grad, i_idx, dVdr[:, np.newaxis] * unit_vec)
    np.add.at(grad, j_idx, -dVdr[:, np.newaxis] * unit_vec)
    
    return grad

def _barrier_energy(positions, beta=50.0):
    """Energy with a barrier penalizing distances below 0.5."""
    n = positions.shape[0]
    if n < 2:
        return _dzugutov_energy(positions)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    
    e = _dzugutov_energy(positions)
    
    # Add barrier for r < 0.5
    mask = r < 0.5
    if np.any(mask):
        barrier = np.sum((0.5 - r[mask])**2) * beta
        e += barrier
    
    return e

def _barrier_gradient(positions, beta=50.0):
    """Gradient of barrier-augmented energy."""
    n = positions.shape[0]
    if n < 2:
        return _dzugutov_gradient(positions)
    
    grad = _dzugutov_gradient(positions)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    r_safe = np.maximum(r, 1e-10)
    
    mask = r < 0.5
    if np.any(mask):
        # d/dr [(0.5-r)^2 * beta] = -2*beta*(0.5-r)
        force = -2.0 * beta * (0.5 - r[mask])
        unit_vec = diff[mask] / r_safe[mask, np.newaxis]
        np.add.at(grad, i_idx[mask], force[:, np.newaxis] * unit_vec)
        np.add.at(grad, j_idx[mask], -force[:, np.newaxis] * unit_vec)
    
    return grad

def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedron-based initial guess.
    Layer 1: 13 atoms (1 center + 12 icosahedron vertices at r_min)
    Layer 2: 55 atoms total (add 42 atoms on second shell)
    For n between 13 and 55, use partial second shell.
    For n > 55, add third shell.
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        idx = rng.choice(13, n, replace=False)
        return np.array([points[i] for i in idx])
    
    # Shell 2: Mackay icosahedron has 55 atoms total
    # Second shell radius ~ 1.72 * r_min (sqrt(3) * r_min for face centers, but Mackay uses specific positions)
    # For simplicity, use a slightly larger radius
    r2 = 1.72 * r_min
    needed = min(n - 13, 42)  # Max 42 atoms in second shell for Mackay
    
    # Generate points on sphere of radius r2, rejecting those too close to existing
    added = 0
    attempts = 0
    min_dist = 0.5
    
    while added < needed and attempts < 20000:
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = r2 * np.sin(phi) * np.cos(theta)
        y = r2 * np.sin(phi) * np.sin(theta)
        z = r2 * np.cos(phi)
        p = np.array([x, y, z])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
            added += 1
        attempts += 1
    
    if n > 55:
        # Third shell
        r3 = 2.4 * r_min
        needed3 = n - len(points)
        added3 = 0
        attempts3 = 0
        while added3 < needed3 and attempts3 < 20000:
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = r3 * np.sin(phi) * np.cos(theta)
            y = r3 * np.sin(phi) * np.sin(theta)
            z = r3 * np.cos(phi)
            p = np.array([x, y, z])
            
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            
            if valid:
                points.append(p)
                added3 += 1
            attempts3 += 1
    
    # If we still don't have enough, fill randomly
    while len(points) < n:
        r_max = r_min * (n / 13) ** (1/3) * 1.5
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) > 1e-6:
            p = p / np.linalg.norm(p) * r_min * (n / 13) ** (1/3)
        points.append(p)
    
    return np.array(points[:n])

def _build_polytetrahedral_random(n, rng, r_min=R_MIN):
    """
    Build a random polytetrahedral-like cluster.
    Start with a tetrahedron, then add atoms in tetrahedral positions.
    """
    # Start with a tetrahedron
    phi = (1 + np.sqrt(5)) / 2
    tetra = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) * (r_min / 2)
    
    points = [tetra[i].copy() for i in range(4)]
    
    # Add more atoms in tetrahedral configurations
    min_dist = 0.5
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        # Pick a random existing atom and try to add a new one in a tetrahedral position
        base_idx = rng.randint(0, len(points))
        base = points[base_idx]
        
        # Generate a random direction
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi_angle = np.arccos(2 * v - 1)
        
        direction = np.array([
            np.sin(phi_angle) * np.cos(theta),
            np.sin(phi_angle) * np.sin(theta),
            np.cos(phi_angle)
        ])
        
        # Place at r_min distance
        p = base + direction * r_min
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
        attempts += 1
    
    # If we still don't have enough, fill randomly
    while len(points) < n:
        r_max = r_min * (n / 4) ** (1/3) * 1.5
        p = rng.uniform(-r_max, r_max, 3)
        points.append(p)
    
    return np.array(points[:n])

def _build_compact_random(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    radius = r_min * (n / 13) ** (1/3) * 1.3
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
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
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.3 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def _refine(positions, time_limit, deadline, rng):
    """
    Refine positions using barrier-augmented L-BFGS, then basin hopping.
    Returns (energy, positions).
    """
    n = positions.shape[0]
    best_energy = _dzugutov_energy(positions)
    best_pos = positions.copy()
    
    # Phase 1: Barrier-augmented L-BFGS
    elapsed = time.time() - time_limit[0]
    if elapsed < deadline - 30:
        try:
            result = minimize(
                _barrier_energy,
                best_pos,
                jac=_barrier_gradient,
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Standard L-BFGS without barrier
    elapsed = time.time() - time_limit[0]
    if elapsed < deadline - 20:
        try:
            result = minimize(
                _dzugutov_energy,
                best_pos,
                jac=_dzugutov_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Basin hopping
    elapsed = time.time() - time_limit[0]
    if elapsed < deadline - 10:
        n_bh_steps = 20
        sigma_start = 0.15
        sigma_end = 0.05
        
        for step in range(n_bh_steps):
            if time.time() > deadline:
                break
            
            t = step / max(1, n_bh_steps - 1)
            sigma = sigma_start + t * (sigma_end - sigma_start)
            
            # Perturb
            perturbed = best_pos + rng.normal(0, sigma, size=best_pos.shape)
            
            # Refine perturbed
            try:
                result = minimize(
                    _dzugutov_energy,
                    perturbed,
                    jac=_dzugutov_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
            except Exception:
                pass
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov cluster structures using topological seeds and basin hopping."""
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations from different geometric families
    generators = [
        ('mackay', lambda: _build_mackay_icosahedron(n, rng, R_MIN)),
        ('polytetra', lambda: _build_polytetrahedral_random(n, rng, R_MIN)),
        ('compact', lambda: _build_compact_random(n, rng, R_MIN)),
    ]
    
    # Try multiple restarts from different geometric families
    n_restarts_per_family = 2
    total_restarts = n_restarts_per_family * len(generators)
    
    for restart in range(total_restarts):
        if time.time() > deadline - 30:
            break
        
        family_idx = restart // n_restarts_per_family
        gen_name, gen_func = generators[family_idx]
        
        # Generate initial positions
        try:
            positions = gen_func()
        except Exception:
            positions = _build_compact_random(n, rng, R_MIN)
        
        # Refine
        try:
            energy, refined_pos = _refine(positions, start_time, deadline, rng)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos.copy()
        except Exception:
            continue
    
    # Final polish: high-precision optimization from best
    if best_positions is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                _dzugutov_energy,
                best_positions,
                jac=_dzugutov_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: return random positions
        best_positions = _build_compact_random(n, rng, R_MIN)
    
    return best_positions
