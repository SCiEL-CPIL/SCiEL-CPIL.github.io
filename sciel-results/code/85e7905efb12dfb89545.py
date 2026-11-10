"""
Strategy: Deterministic Multi-Start Annealing (DMSA) with Structured Crystalline Initialization.
Instead of Basin Hopping (random kick + local relax), we use Simulated Annealing (SA) 
with a cooling schedule to explore the energy landscape. 
Initializations are diverse and structured: FCC, BCC, and Shell models.
SA is robust for escaping local minima and can be run to convergence.
We use multiple independent SA runs from different structured starts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf) # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    
    # Sum over i<j
    # The matrix r is symmetric. Sum over all i!=j is 2 * sum_{i<j}.
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    a = np.sqrt(2.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_icosahedron(n, rng):
    """Generate an icosahedral shell structure. Good for short range (rho=3)."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [1, 0, -phi], [-1, 0, phi], [-1, 0, -phi],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [phi, 1, 0], [phi, -1, 0], [-phi, 1, 0], [-phi, -1, 0]
    ])
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Scale to match packing density roughly
    # For n=13, radius is 1.
    # For larger n, we add shells.
    # Simple approach: generate multiple shells of icosahedra.
    
    points = []
    # Shell 1 (12 atoms + 1 center)
    points.append(np.zeros(3))
    for v in verts:
        points.append(v)
        
    # If we need more, add a second shell scaled by ~2.0
    if n > 13:
        # Scale factor for second shell
        # Distance between shells in icosahedral clusters is typically ~2.0
        scale2 = 2.0
        # Add more vertices by rotating or just scaling the same set (approximation)
        # Better: generate a larger set of points on the sphere
        # For simplicity, we'll just add more points on a sphere of radius 2.0
        # using a Fibonacci sphere distribution for uniformity
        num_to_add = min(n - len(points), 50) # Cap additions per shell
        if num_to_add > 0:
            # Fibonacci sphere
            N = num_to_add
            i = np.arange(N)
            phi_i = np.arccos(1 - 2 * (i + 0.5) / N)
            theta_i = np.pi * (1 + 5**0.5) * i
            x = np.cos(theta_i) * np.sin(phi_i)
            y = np.sin(theta_i) * np.sin(phi_i)
            z = np.cos(phi_i)
            shell2 = np.column_stack((x, y, z)) * scale2
            points.extend(shell2.tolist())
            
            # If still not enough, add a third shell
            if n > len(points):
                scale3 = 3.0
                num_to_add3 = n - len(points)
                if num_to_add3 > 0:
                    N3 = num_to_add3
                    i3 = np.arange(N3)
                    phi_i3 = np.arccos(1 - 2 * (i3 + 0.5) / N3)
                    theta_i3 = np.pi * (1 + 5**0.5) * i3
                    x3 = np.cos(theta_i3) * np.sin(phi_i3)
                    y3 = np.sin(theta_i3) * np.sin(phi_i3)
                    z3 = np.cos(phi_i3)
                    shell3 = np.column_stack((x3, y3, z3)) * scale3
                    points.extend(shell3.tolist())
                    
    points = np.array(points[:n])
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_shell(n, rng):
    """Generate a generic spherical shell structure. Good for medium range."""
    # Use Fibonacci sphere for uniform distribution
    points = []
    # Estimate radius
    # Volume ~ 4/3 pi R^3 ~ n * volume_per_atom
    # volume_per_atom ~ 1 (roughly)
    R = (3 * n / (4 * np.pi)) ** (1/3) * 1.5
    
    # Distribute n points on sphere of radius R
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.column_stack((x, y, z)) * R
    points += rng.normal(0, 0.01, points.shape)
    return points

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _simulated_annealing(initial_pos, rho, n_steps=2000, T_start=1.0, T_end=0.01):
    """
    Simulated Annealing for global optimization.
    Moves are random Gaussian steps.
    Acceptance follows Metropolis criterion.
    """
    current_pos = initial_pos.copy()
    current_energy = _morse_energy_and_grad(current_pos, rho)[0]
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    # Linear cooling schedule
    temperatures = np.linspace(T_start, T_end, n_steps)
    
    for step in range(n_steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        T = temperatures[step]
        
        # Random move
        # Step size decreases with T
        step_size = 0.2 * (T / T_start) + 0.01
        move = rng.normal(0, step_size, current_pos.shape)
        proposed_pos = current_pos + move
        
        # Check validity (soft constraint: if invalid, reject or penalize)
        # For simplicity, we reject if invalid, or use a penalty
        # Let's just reject if invalid to ensure final result is valid
        if not _is_valid(proposed_pos):
            continue
            
        proposed_energy = _morse_energy_and_grad(proposed_pos, rho)[0]
        
        delta_E = proposed_energy - current_energy
        
        # Metropolis criterion
        if delta_E < 0:
            accept = True
        else:
            if T > 1e-8:
                accept = np.random.random() < np.exp(-delta_E / T)
            else:
                accept = False
                
        if accept:
            current_pos = proposed_pos
            current_energy = proposed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
    # Final local relaxation
    final_pos, final_energy = _local_minimize(best_pos, rho)
    if _is_valid(final_pos):
        return final_pos, final_energy
        
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. FCC
    fcc = _generate_fcc(size, rng)
    if fcc is not None:
        configs.append(fcc)
        
    # 2. BCC
    bcc = _generate_bcc(size, rng)
    if bcc is not None:
        configs.append(bcc)
        
    # 3. Icosahedral (especially good for rho=3)
    ico = _generate_icosahedron(size, rng)
    if ico is not None:
        configs.append(ico)
        
    # 4. Shell
    shell = _generate_shell(size, rng)
    if shell is not None:
        configs.append(shell)
        
    best_energy = float('inf')
    best_positions = None
    
    # Run Simulated Annealing for each configuration
    # We allocate time based on number of configs
    num_configs = len(configs)
    steps_per_run = 1500 # Tuned for speed
    
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
            
        try:
            final_pos, final_energy = _simulated_annealing(pos, parameter, n_steps=steps_per_run)
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception:
            # Fallback to local relax
            try:
                energy, positions = relax(pos, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If no valid position found, return a simple valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
