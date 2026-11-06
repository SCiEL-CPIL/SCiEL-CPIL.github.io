"""Morse cluster global minima: basin hopping with L-BFGS, lattice seeds, and time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # Safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = [0]


def _morse_energy_gradient(positions, rho):
    """Compute Morse energy and gradient for all pairs.
    E = sum_{i<j} exp(rho*(1-r_ij)) * (exp(rho*(1-r_ij)) - 2)
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Mask out self-pairs
    mask = ~np.eye(n, dtype=bool)
    dist_masked = dist[mask]
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    # Let u = rho*(1-r), then V = exp(u)*(exp(u)-2)
    # dV/dr = exp(u)*(exp(u)-2)*(-rho) + exp(u)*exp(u)*(-rho) = -rho*exp(u)*(2*exp(u)-2) = -2*rho*exp(u)*(exp(u)-1)
    # Actually let me recompute:
    # V = exp(u)*(exp(u) - 2) where u = rho*(1-r)
    # dV/du = exp(u)*(exp(u)-2) + exp(u)*exp(u) = exp(u)*(2*exp(u) - 2) = 2*exp(u)*(exp(u)-1)
    # du/dr = -rho
    # dV/dr = 2*exp(u)*(exp(u)-1)*(-rho) = -2*rho*exp(u)*(exp(u)-1)
    
    u = rho * (1.0 - dist_masked)
    exp_u = np.exp(u)
    energy = np.sum(exp_u * (exp_u - 2.0))
    
    # dV/dr for each pair
    dV_dr = -2.0 * rho * exp_u * (exp_u - 1.0)
    
    # Gradient: dE/dx_i = sum_{j!=i} dV/dr_ij * (r_i - r_j)/r_ij
    # dV/dr_ij * (r_i - r_j)/r_ij = dV_dr * (r_i - r_j) / r_ij
    # But we need to handle r_ij = 0 case (shouldn't happen with valid positions)
    
    # Build full gradient
    grad = np.zeros_like(positions)
    dist_full = dist.copy()
    np.fill_diagonal(dist_full, 1.0)  # avoid division by zero
    
    # For each pair (i,j), contribution to grad[i] is dV_dr_ij * (r_i - r_j)/r_ij
    # and contribution to grad[j] is dV_dr_ij * (r_j - r_i)/r_ij = -dV_dr_ij * (r_i - r_j)/r_ij
    
    # Vectorized: for each pair (i,j), weight = dV_dr / r_ij
    # grad[i] += weight * (r_i - r_j)
    
    # Create weight matrix
    weight = np.zeros((n, n))
    weight[mask] = dV_dr / dist_masked
    
    # grad[i] = sum_j weight[i,j] * (r_i - r_j)
    # = (sum_j weight[i,j]) * r_i - sum_j weight[i,j] * r_j
    row_sum = np.sum(weight, axis=1, keepdims=True)
    weighted_sum = weight @ positions
    
    grad = row_sum * positions - weighted_sum
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute just the Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dist_masked = dist[mask]
    u = rho * (1.0 - dist_masked)
    exp_u = np.exp(u)
    return np.sum(exp_u * (exp_u - 2.0))


def _generate_fcc(n, spacing=1.0):
    """Generate n points from FCC lattice structure."""
    points = []
    a = spacing
    # FCC basis vectors
    basis = np.array([
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0],
    ])
    # Generate grid
    r_max = spacing * (n ** (1.0/3.0) + 3)
    n_grid = int(r_max / (a * 0.4)) + 2
    for i in range(-n_grid, n_grid + 1):
        for j in range(-n_grid, n_grid + 1):
            for k in range(-n_grid, n_grid + 1):
                for b in basis:
                    p = np.array([i*a, j*a, k*a]) + b
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    points = np.array(points)
    # Center
    points -= np.mean(points, axis=0)
    
    # Select n closest to origin
    if len(points) >= n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    else:
        # Pad with random points if needed
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p.reshape(1, 3)])
    
    return points[:n]


def _generate_shell(n, spacing=1.0):
    """Generate n points on concentric shells (icosahedral-like)."""
    points = []
    # First shell: 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * spacing
    points.extend(ico.tolist())
    
    # Second shell: 30 points (cuboctahedron-like)
    if n > 12:
        # Use points on a sphere
        r2 = spacing * 2.0
        n2 = min(n - len(points), 30)
        for i in range(n2):
            theta = 2 * np.pi * i / n2
            z = r2 * (1 - 2*(i+0.5)/n2)
            xy = np.sqrt(max(0, r2**2 - z**2))
            points.append([xy * np.cos(theta), xy * np.sin(theta), z])
    
    # Add more shells if needed
    shell_r = spacing * 2.0
    while len(points) < n:
        shell_r += spacing * 0.8
        n_shell = int(4 * np.pi * shell_r**2 / (spacing**2 * 1.5))
        n_add = min(n - len(points), n_shell)
        for i in range(n_add):
            theta = 2 * np.pi * i / max(1, n_shell)
            z = shell_r * (1 - 2*(i+0.5)/max(1, n_shell))
            xy = np.sqrt(max(0, shell_r**2 - z**2))
            points.append([xy * np.cos(theta), xy * np.sin(theta), z])
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_random(n, spacing=1.0, seed=None):
    """Generate n random points with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 100000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough points, just add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _lbfgs_optimize(positions, rho, max_iter=200, tol=1e-8):
    """Run L-BFGS optimization on Morse energy."""
    from scipy.optimize import minimize
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, _ = _morse_energy_gradient(x, rho)
        return e
    
    def jac(x):
        x = x.reshape(-1, 3)
        _, g = _morse_energy_gradient(x, rho)
        return g.flatten()
    
    try:
        result = minimize(fun, positions.flatten(), jac=jac, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy_only(positions, rho)


def _enforce_min_distance(positions, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    n = len(positions)
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    # Push apart
                    if d < 1e-10:
                        direction = np.random.randn(3)
                    else:
                        direction = (positions[i] - positions[j]) / d
                    push = (min_dist - d) / 2.0 + 0.01
                    positions[i] += push * direction
                    positions[j] -= push * direction
                    moved = True
        if not moved:
            break
    return positions


def _basin_hopping(positions, rho, n_kicks, kick_scale=0.5, max_iter_per_kick=100):
    """Basin hopping: kick, optimize, accept if lower energy."""
    best_pos = positions.copy()
    best_energy = _morse_energy_only(positions, rho)
    
    for kick in range(n_kicks):
        if time.time() > _HARD_DEADLINE:
            break
        
        # Kick
        kicked = best_pos + np.random.randn(*best_pos.shape) * kick_scale
        kicked = _enforce_min_distance(kicked, 0.5)
        
        # Optimize
        opt_pos, opt_energy = _lbfgs_optimize(kicked, rho, max_iter=max_iter_per_kick)
        
        # Accept if better
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pos = opt_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Construct low-energy Morse cluster configuration."""
    _INSTANCE_INDEX[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Generate diverse initial structures
    seeds = []
    
    # FCC lattice
    try:
        seeds.append(_generate_fcc(size, spacing=1.0))
    except Exception:
        pass
    
    # Shell structure
    try:
        seeds.append(_generate_shell(size, spacing=1.0))
    except Exception:
        pass
    
    # Random with different seeds
    for s in [size, size*2+1, size*3+7]:
        try:
            seeds.append(_generate_random(size, spacing=1.0, seed=s))
        except Exception:
            pass
    
    # If we have no seeds, use random
    if not seeds:
        seeds.append(_generate_random(size, spacing=1.0, seed=42))
    
    # Optimize each seed with basin hopping
    best_pos = None
    best_energy = np.inf
    
    # Time budget per instance
    time_per_instance = 270.0 / _INSTANCE_COUNT  # ~45 seconds per instance
    instance_start = time.time()
    
    for seed_idx, seed_pos in enumerate(seeds):
        if time.time() > _HARD_DEADLINE:
            break
        if time.time() - instance_start > time_per_instance:
            break
        
        # First optimize the seed itself
        opt_pos, opt_energy = _lbfgs_optimize(seed_pos, rho, max_iter=150)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pos = opt_pos.copy()
        
        # Basin hopping from this optimized structure
        remaining_time = min(time_per_instance - (time.time() - instance_start), 
                            _HARD_DEADLINE - time.time())
        if remaining_time < 5:
            continue
        
        # Estimate kicks based on remaining time
        n_kicks = max(3, int(remaining_time / 3.0))  # ~3 seconds per kick
        bh_pos, bh_energy = _basin_hopping(opt_pos, rho, n_kicks=n_kicks, 
                                            kick_scale=0.3, max_iter_per_kick=80)
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Final optimization of the best result
    if best_pos is not None and time.time() < _HARD_DEADLINE - 10:
        final_pos, final_energy = _lbfgs_optimize(best_pos, rho, max_iter=200)
        if final_energy < best_energy:
            best_pos = final_pos
            best_energy = final_energy
    
    if best_pos is None:
        # Fallback: return a simple structure
        best_pos = _generate_random(size, spacing=1.0, seed=123)
    
    # Ensure minimum distance constraint
    best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos
# EVOLVE-BLOCK-END
