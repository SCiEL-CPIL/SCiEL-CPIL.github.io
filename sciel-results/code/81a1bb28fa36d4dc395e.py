"""Dzugutov cluster global minima via polytetrahedral geometric initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_HARD_LIMIT = 320.0  # safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient(positions):
    """
    Compute energy and gradient for the Dzugutov potential (piecewise).
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a]
          + B exp(d/(r - b)) [a <= r < b]
          + 0 [r >= b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            r_vec = diff / r
            force_mag = 0.0
            
            if r < a:
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
                energy += B * np.exp(d / (r - b))
            
            gradient[i] += force_mag * r_vec
            gradient[j] -= force_mag * r_vec
    
    return energy, gradient

def _icosahedron_13():
    """Return the 13-atom icosahedral structure (known global minimum for n=13)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # Optimal bond length for Dzugutov is around 1.9
    r1 = 1.9
    positions = np.vstack([np.zeros(3), verts * r1])
    return positions

def _biicosahedron_55():
    """
    Return the 55-atom bi-icosahedral structure (known global minimum for n=55).
    This is a Mackay icosahedron with 55 atoms (1 + 12 + 42 = 55).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    r1 = 1.9
    positions = [np.zeros(3)]
    # Shell 1: 12 atoms
    for v in verts:
        positions.append(v * r1)
    
    # Shell 2: 42 atoms - these are at the vertices of a larger icosahedron
    # and face centers. For Mackay icosahedron, shell 2 has 42 atoms.
    # The 42 atoms are: 12 vertices at r2, 30 face centers at r2*sqrt(1+1/phi^2)
    # Actually, for the 55-atom Mackay icosahedron:
    # - 1 center
    # - 12 at r1 (first shell)
    # - 30 at r2 (face centers of the icosahedron)
    # - 12 at r3 (vertices of a larger icosahedron)
    # Total: 1 + 12 + 30 + 12 = 55
    
    # Face centers: midpoints of edges of the icosahedron, normalized
    # The icosahedron has 30 edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.5:  # Close enough to be an edge
                edges.append((i, j))
    
    r2 = r1 * np.sqrt(1 + 1/phi**2)  # Distance to face centers
    for i, j in edges[:30]:
        face_center = (verts[i] + verts[j]) / 2
        face_center = face_center / np.linalg.norm(face_center) * r2
        positions.append(face_center)
    
    # Outer vertices: scaled icosahedron
    r3 = r1 * 2.0
    for v in verts:
        positions.append(v * r3)
    
    return np.array(positions)

def _generate_pt_core(n_target, rng):
    """
    Generate a PT-ordered core structure for n_target atoms.
    Uses known global minima as cores and adds atoms geometrically.
    """
    if n_target <= 13:
        return _icosahedron_13()[:n_target]
    
    if n_target <= 55:
        # Start with 13-atom core, add atoms
        core = _icosahedron_13()
        positions = [core]
        remaining = n_target - 13
        
        # Add atoms on a second shell
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        r2 = 1.9 * 1.7
        
        for _ in range(remaining):
            # Pick a random direction biased toward interstitial sites
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            positions.append(v)
        
        return np.array(positions)
    
    # For n > 55, use 55-atom core
    core = _biicosahedron_55()
    positions = [core]
    remaining = n_target - 55
    
    # Add atoms on an outer shell
    r_outer = 1.9 * 2.5
    for _ in range(remaining):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * r_outer
        positions.append(v)
    
    return np.array(positions)

def _random_compact(n_atoms, rng, spacing=1.1):
    """Generate a random compact structure."""
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)

def _local_minimize(positions, rng, maxiter=200):
    """Perform local minimization using L-BFGS-B."""
    try:
        res = minimize(
            lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
            positions.flatten(),
            jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxls': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res.fun < np.inf:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return _dzugutov_energy_gradient(positions)[0], positions

def _basin_hopping_chain(initial_positions, rng, time_limit):
    """Run a basin hopping chain."""
    current_energy, current_positions = _local_minimize(initial_positions, rng)
    best_energy, best_positions = current_energy, current_positions.copy()
    
    start_time = time.time()
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        try:
            new_energy, new_positions = _local_minimize(perturbed, rng, maxiter=100)
        except Exception:
            continue
        
        if new_energy < best_energy:
            best_energy, best_positions = new_energy, new_positions.copy()
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy, current_positions = new_energy, new_positions
        else:
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / 0.1):
                current_energy, current_positions = new_energy, new_positions
    
    return best_energy, best_positions

def _solve_instance_with_deadline(instance, deadline):
    """
    Solve a single instance with a specific deadline.
    """
    n = instance['n']
    potential = instance['potential']
    
    rng = np.random.default_rng(n)
    
    # Generate multiple diverse seeds
    seeds = []
    
    # Seed 1: PT-ordered core
    try:
        seeds.append(_generate_pt_core(n, rng))
    except Exception:
        pass
    
    # Seed 2: Random compact
    try:
        seeds.append(_random_compact(n, rng))
    except Exception:
        pass
    
    # Seed 3: Perturbed icosahedron
    try:
        base = _icosahedron_13()
        extra = n - 13
        if extra > 0:
            extra_pos = rng.normal(0, 1.0, (extra, 3)) + 2.5
            seeds.append(np.vstack([base, extra_pos]))
    except Exception:
        pass
    
    # Seed 4: Bi-icosahedron based
    try:
        base = _biicosahedron_55()
        if n >= 55:
            extra = n - 55
            extra_pos = rng.normal(0, 1.0, (extra, 3)) + 4.0
            seeds.append(np.vstack([base, extra_pos]))
        else:
            # Trim
            seeds.append(base[:n])
    except Exception:
        pass
    
    if not seeds:
        seeds.append(_random_compact(n, rng))
    
    best_energy = np.inf
    best_pos = None
    
    # Time budget per seed
    num_seeds = len(seeds)
    time_per_seed = max(1.0, (deadline - time.time()) / num_seeds * 0.8)
    
    for seed in seeds:
        if time.time() > deadline - 2:
            break
        
        try:
            # First try the provided relax function
            try:
                e, opt_pos = relax(seed, potential)
            except Exception:
                try:
                    parameter = instance.get("rho", instance.get("kind"))
                    e, opt_pos = relax(seed, potential, parameter)
                except Exception:
                    # Fallback to our own local minimization
                    e, opt_pos = _local_minimize(seed, rng)
            
            if e < best_energy:
                best_energy = e
                best_pos = opt_pos
                
        except Exception:
            pass
        
        # If we have time, do basin hopping from this seed
        if time.time() < deadline - 5:
            try:
                bh_energy, bh_pos = _basin_hopping_chain(seed, rng, time_per_seed * 0.5)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos
            except Exception:
                pass
    
    if best_pos is None:
        best_pos = seeds[0]
        
    return best_pos, best_energy

def construct(instance):
    """
    Construct a low-energy Dzugutov cluster.
    """
    n = instance['n']
    potential = instance['potential']
    
    # Local deadline for this call
    # 300s / 6 instances = 50s. Let's use 45s.
    local_deadline = time.time() + 45.0
    
    pos, energy = _solve_instance_with_deadline(instance, local_deadline)
    
    return pos

# EVOLVE-BLOCK-END
