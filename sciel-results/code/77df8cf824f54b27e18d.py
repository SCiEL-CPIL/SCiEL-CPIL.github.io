"""Dzugutov cluster global minima search using continuous annealing of potential
parameters c and d to smooth the energy landscape, with L-BFGS refinement."""
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

def _dzugutov_energy_annealed(positions, c_val, d_val):
    """Compute Dzugutov potential energy with annealed c and d parameters."""
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
        energy[mask_a] = (A * (r_a**(-m) - B)) * np.exp(c_val / (r_a - a))
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        energy[mask_b] = B * np.exp(d_val / (r_b - b))
    
    return np.sum(energy)

def _dzugutov_gradient_annealed(positions, c_val, d_val):
    """Compute gradient of Dzugutov potential with annealed c and d parameters."""
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
        exp_term = np.exp(c_val / (r_a - a))
        dVdr[mask_a] = A * (-m * r_a**(-m-1) * exp_term + (r_a**(-m) - B) * exp_term * (-c_val / (r_a - a)**2))
    
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        dVdr[mask_b] = B * np.exp(d_val / (r_b - b)) * (-d_val / (r_b - b)**2)
    
    unit_vec = diff / r_safe[:, np.newaxis]
    
    np.add.at(grad, i_idx, dVdr[:, np.newaxis] * unit_vec)
    np.add.at(grad, j_idx, -dVdr[:, np.newaxis] * unit_vec)
    
    return grad

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for n atoms. Vectorized pair computation."""
    return _dzugutov_energy_annealed(positions, c, d)

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy. Vectorized."""
    return _dzugutov_gradient_annealed(positions, c, d)

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
    Build a Mackay icosahedron-based initial guess with proper shell structure.
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        idx = rng.choice(13, n, replace=False)
        return np.array([points[i] for i in idx])
    
    # Shell 2: 42 atoms at radius ~1.72*r_min
    r2 = 1.72 * r_min
    needed = min(n - 13, 42)
    
    # Use FWHM-inspired placement: distribute on sphere with minimum distance
    added = 0
    attempts = 0
    min_dist = 0.5
    
    # Pre-compute existing points for fast distance check
    existing = np.array(points)
    
    while added < needed and attempts < 30000:
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = r2 * np.sin(phi) * np.cos(theta)
        y = r2 * np.sin(phi) * np.sin(theta)
        z = r2 * np.cos(phi)
        p = np.array([x, y, z])
        
        # Vectorized distance check
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
            existing = np.vstack([existing, p])
            added += 1
        attempts += 1
    
    if n > 55:
        # Third shell
        r3 = 2.4 * r_min
        needed3 = n - len(points)
        added3 = 0
        attempts3 = 0
        existing = np.array(points)
        
        while added3 < needed3 and attempts3 < 30000:
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = r3 * np.sin(phi) * np.cos(theta)
            y = r3 * np.sin(phi) * np.sin(theta)
            z = r3 * np.cos(phi)
            p = np.array([x, y, z])
            
            dists = np.linalg.norm(existing - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
                existing = np.vstack([existing, p])
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

def _continuous_annealing(positions, start_time, deadline, rng, n_anneal_steps=50):
    """
    Perform continuous annealing of c and d parameters from high to low,
    allowing gradient descent to flow into the global basin.
    """
    pos = positions.copy()
    
    # Anneal c and d from full values down to near-zero, then back up
    # This creates a smooth landscape that guides to global minimum
    c_schedule = np.linspace(c, 0.01, n_anneal_steps)
    d_schedule = np.linspace(d, 0.01, n_anneal_steps)
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 30:
            break
        
        c_val = c_schedule[step]
        d_val = d_schedule[step]
        
        # Optimize with current annealed parameters
        try:
            result = minimize(
                lambda p: _dzugutov_energy_annealed(p, c_val, d_val),
                pos,
                jac=lambda p: _dzugutov_gradient_annealed(p, c_val, d_val),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
            )
            pos = result.x
        except Exception:
            pass
    
    # Now ramp back up to full parameters with fine-tuning
    c_ramp = np.linspace(0.01, c, 20)
    d_ramp = np.linspace(0.01, d, 20)
    
    for step in range(20):
        if time.time() > deadline - 15:
            break
        
        c_val = c_ramp[step]
        d_val = d_ramp[step]
        
        try:
            result = minimize(
                lambda p: _dzugutov_energy_annealed(p, c_val, d_val),
                pos,
                jac=lambda p: _dzugutov_gradient_annealed(p, c_val, d_val),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            pos = result.x
        except Exception:
            pass
    
    # Final optimization with full parameters
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _dzugutov_energy,
                pos,
                jac=_dzugutov_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            pos = result.x
        except Exception:
            pass
    
    final_energy = _dzugutov_energy(pos)
    return final_energy, pos

def construct(instance):
    """Find low-energy Dzugutov cluster structures using continuous annealing."""
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
    
    # Try each geometric family with continuous annealing
    for gen_name, gen_func in generators:
        if time.time() > deadline - 60:
            break
        
        # Generate initial positions
        try:
            positions = gen_func()
        except Exception:
            positions = _build_compact_random(n, rng, R_MIN)
        
        # Continuous annealing refinement
        try:
            energy, refined_pos = _continuous_annealing(positions, start_time, deadline, rng)
            
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
