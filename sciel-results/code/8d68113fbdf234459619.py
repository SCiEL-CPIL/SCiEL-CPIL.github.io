import numpy as np
import time
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# We aim to finish by 290s to be safe, leaving 40s buffer for overhead/evaluation
_DEADLINE = _START_TIME + 290.0

def _time_remaining():
    return _DEADLINE - time.time()

def _get_morse_energy_and_grad(rho, coords):
    """
    Compute Morse energy and gradient using the provided library function.
    The library function `morse_energy_gradient` likely returns (energy, gradient).
    """
    try:
        e, g = morse_energy_gradient(coords, rho)
        return e, g
    except Exception:
        # Fallback if library function fails or behaves unexpectedly
        n = len(coords)
        if n < 2:
            return 0.0, np.zeros_like(coords)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Check for overlaps
        min_dist = dist.min()
        if min_dist < 0.5:
            return 1e10, np.zeros_like(coords)
            
        x = rho * (1.0 - dist)
        ex = np.exp(x)
        e_pair = ex * (ex - 2.0)
        energy = 0.5 * np.sum(e_pair)
        
        # Gradient
        # dE/dr = -2 * rho * ex * (ex - 1)
        # grad_i = sum_j dE/dr_ij * (r_i - r_j) / r_ij
        dE_dr = -2.0 * rho * ex * (ex - 1.0)
        dE_dr[np.diag_indices(n)] = 0.0
        
        # grad[i] = sum_j dE_dr[i,j] * (coords[i] - coords[j]) / dist[i,j]
        # dist_safe to avoid division by zero
        dist_safe = np.where(dist > 0, dist, 1.0)
        coeff = dE_dr / dist_safe
        coeff[np.diag_indices(n)] = 0.0
        
        grad = np.einsum('ij,ijc->ic', coeff, diff)
        return energy, grad

def _local_minimize(coords, rho, max_iter=100):
    """
    Local minimization using L-BFGS-B.
    """
    n = len(coords)
    x0 = coords.flatten()
    
    def objective(x):
        c = x.reshape(-1, 3)
        e, _ = _get_morse_energy_and_grad(rho, c)
        return e
    
    def objective_grad(x):
        c = x.reshape(-1, 3)
        e, g = _get_morse_energy_and_grad(rho, c)
        return e, g.flatten()
        
    try:
        res = minimize(objective_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e9: # Valid solution
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Fallback to simple relax if minimize fails
    try:
        e, c = relax(coords, 'morse', rho)
        return e, c
    except Exception:
        e, _ = _get_morse_energy_and_grad(rho, coords)
        return e, coords

def _generate_icosahedral_shell(n, rng):
    """
    Generate points on an icosahedral shell.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Add center
    points.append(np.zeros(3))
    
    # Add vertices
    for v in verts:
        points.append(v * 1.0)
        
    # If we need more, add points on edges or faces
    # For simplicity, if n > 13, we can jitter or add more shells
    while len(points) < n:
        # Pick a random vertex and add a point near it
        v = verts[rng.integers(0, len(verts))]
        p = v * (1.0 + rng.normal(0, 0.1))
        points.append(p)
        
    return np.array(points[:n])

def _generate_fcc_like(n, rng):
    """
    Generate points in an FCC-like lattice cut to a sphere.
    """
    a = 1.0 # lattice parameter
    max_radius = (n / 2) ** (1/3) * 1.5 # rough estimate
    candidates = []
    
    # Generate grid points
    range_val = int(np.ceil(max_radius / a)) + 1
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= max_radius:
                    candidates.append(p)
                    # FCC offsets
                    candidates.append(p + np.array([a/2, a/2, 0]))
                    candidates.append(p + np.array([a/2, 0, a/2]))
                    candidates.append(p + np.array([0, a/2, a/2]))
                    
    candidates = np.array(candidates)
    # Shuffle
    idx = rng.permutation(len(candidates))
    candidates = candidates[idx]
    
    # Select points ensuring minimum distance
    selected = []
    min_dist = 0.9
    for p in candidates:
        if all(np.linalg.norm(p - s) >= min_dist for s in selected):
            selected.append(p)
            if len(selected) == n:
                break
                
    # If not enough, fill randomly
    while len(selected) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        selected.append(p)
        
    return np.array(selected[:n])

def _generate_random(n, rng):
    """
    Generate random points in a sphere.
    """
    radius = (n / (4/3 * np.pi * 0.5**3)) ** (1/3) * 1.2 # rough volume estimate
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing
            if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
    return np.array(points)

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance.get("potential", 'morse')
    
    # Check time
    if _time_remaining() < 5:
        # Not enough time, return a simple random guess
        rng = np.random.default_rng(0)
        return rng.uniform(-5, 5, (n, 3))
        
    rng = np.random.default_rng(hash((n, rho)) % (2**32))
    
    best_energy = float('inf')
    best_coords = None
    
    # 1. Generate diverse initial configurations
    inits = []
    
    # Icosahedral
    try:
        inits.append(_generate_icosahedral_shell(n, rng))
    except:
        pass
        
    # FCC-like
    try:
        inits.append(_generate_fcc_like(n, rng))
    except:
        pass
        
    # Random
    try:
        inits.append(_generate_random(n, rng))
    except:
        pass
        
    # Random with larger radius
    try:
        radius = (n / (4/3 * np.pi * 0.5**3)) ** (1/3) * 2.0
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
        inits.append(np.array(points))
    except:
        pass
        
    if not inits:
        # Fallback
        inits.append(rng.uniform(-5, 5, (n, 3)))
        
    # 2. Relax each initial configuration
    for coords in inits:
        if _time_remaining() < 10:
            break
        e, c = _local_minimize(coords, rho)
        if e < best_energy:
            best_energy = e
            best_coords = c.copy()
            
    if best_coords is None:
        # Should not happen
        best_coords = inits[0]
        best_energy, _ = _get_morse_energy_and_grad(rho, best_coords)
        
    # 3. Basin Hopping
    # Estimate steps based on time
    time_per_step = 0.5 # rough estimate
    max_steps = int(_time_remaining() / time_per_step)
    max_steps = min(max_steps, 50) # cap steps
    
    current_energy = best_energy
    current_coords = best_coords.copy()
    
    for step in range(max_steps):
        if _time_remaining() < 5:
            break
            
        # Adaptive displacement
        disp = 0.3 + 0.1 * np.sin(step * 0.5)
        
        # Displace
        displaced = current_coords + rng.normal(0, disp, current_coords.shape)
        
        # Local minimize
        e_new, c_new = _local_minimize(displaced, rho, max_iter=50)
        
        # Metropolis accept
        delta = e_new - current_energy
        if delta < 0:
            current_energy = e_new
            current_coords = c_new
            if e_new < best_energy:
                best_energy = e_new
                best_coords = c_new.copy()
        else:
            # Accept with probability
            if delta < 1.0 and rng.random() < np.exp(-delta / max(0.1, abs(current_energy)*0.05)):
                current_coords = c_new
                current_energy = e_new
                
    return best_coords
