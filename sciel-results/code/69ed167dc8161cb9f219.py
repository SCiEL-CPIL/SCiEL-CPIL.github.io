"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary: min over all coordinates of (0.5 - |coord|)
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    return min(min_pair / 2.0, min_boundary)

def _negative_radius(pos_flat):
    """Objective function: negative of the packing radius."""
    pos = pos_flat.reshape(-1, 2)
    r = _compute_radius(pos)
    return -r

def _initial_hex_grid(n):
    """Create initial hexagonal lattice packing."""
    # Hexagonal lattice
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col
            y = row * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n])
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points *= scale
    
    return points

def _initial_square_grid(n):
    """Create initial square grid packing."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col
            y = row
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n])
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points *= scale
    
    return points

def _initial_random(n, seed=42):
    """Create initial random positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))

def _softmin_logsumexp(values, beta=50.0):
    """Compute soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    return v_min - math.log(np.sum(np.exp(beta * (values - v_min)))) / beta

def _grad_softmin_neg(pos_flat, beta=50.0):
    """Compute gradient of negative soft-min radius."""
    pos = pos_flat.reshape(-1, 2)
    n = len(pos)
    
    # Boundary distances
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    
    # Pairwise distances squared
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 0)
    
    # Pairwise distances
    dist = np.sqrt(dist_sq + 1e-15)
    
    # Soft min of boundary distances and half pairwise distances
    # r = min(min_boundary, min_pair/2)
    # Use soft min: r_soft = -1/beta * log(sum(exp(-beta * values)))
    
    # Values: boundary distances and half pairwise distances
    # For gradient, we need d(r_soft)/d(pos)
    
    # Simplified: use the actual min and its gradient
    min_boundary = float(np.min(dist_boundary))
    min_pair = float(np.min(dist[np.triu_indices(n, k=1)]))
    
    r = min(min_pair / 2.0, min_boundary)
    
    # Gradient
    grad = np.zeros_like(pos)
    
    # If boundary is the limiting factor
    if min_boundary <= min_pair / 2.0:
        # Find which point(s) are at the boundary
        for i in range(n):
            if dist_boundary[i] <= min_boundary + 1e-10:
                # Gradient of (0.5 - max(|x|, |y|))
                ax = abs(pos[i, 0])
                ay = abs(pos[i, 1])
                if ax >= ay:
                    grad[i, 0] = -np.sign(pos[i, 0]) if pos[i, 0] != 0 else 1
                else:
                    grad[i, 1] = -np.sign(pos[i, 1]) if pos[i, 1] != 0 else 1
    
    # If pair distance is the limiting factor
    if min_pair / 2.0 <= min_boundary:
        # Find the closest pair
        dist_mat = dist.copy()
        np.fill_diagonal(dist_mat, np.inf)
        idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        i, j = idx
        if i != j:
            # Gradient of |p_i - p_j| / 2
            dvec = pos[i] - pos[j]
            dnorm = np.linalg.norm(dvec)
            if dnorm > 1e-15:
                unit = dvec / dnorm
                grad[i] += unit / 2.0
                grad[j] -= unit / 2.0
    
    return -grad  # Negative because we're minimizing -r

def _dynamics_refine(pos, time_budget=30.0):
    """Refine using rigid-body dynamics with repulsive forces."""
    start = time.time()
    pos = pos.copy()
    vel = np.zeros_like(pos)
    
    # Parameters
    dt = 0.001
    damping = 0.98
    repulsion_strength = 0.1
    wall_strength = 0.5
    target_r = _compute_radius(pos)
    
    while time.time() - start < time_budget:
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dvec = pos[i] - pos[j]
                dist = np.linalg.norm(dvec)
                if dist < 2 * target_r * 1.5:  # Only interact when close
                    if dist > 1e-10:
                        # Inverse square repulsion
                        force_mag = repulsion_strength / (dist**2 + 1e-6)
                        forces[i] += force_mag * dvec / dist
                        forces[j] -= force_mag * dvec / dist
        
        # Wall repulsion
        for i in range(N):
            for axis in range(2):
                coord = pos[i, axis]
                if coord > 0.5 - target_r:
                    forces[i, axis] -= wall_strength * (coord - (0.5 - target_r))**2
                elif coord < -0.5 + target_r:
                    forces[i, axis] += wall_strength * ((-0.5 + target_r) - coord)**2
        
        # Update velocities and positions
        vel = damping * vel + dt * forces
        pos = pos + dt * vel
        
        # Keep within bounds
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Check convergence
        if np.max(np.abs(vel)) < 1e-8:
            break
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # Try multiple initializations
    init_methods = [
        ("hex", _initial_hex_grid(N)),
        ("sq", _initial_square_grid(N)),
        ("rand1", _initial_random(N, seed=1)),
        ("rand2", _initial_random(N, seed=2)),
        ("rand3", _initial_random(N, seed=42)),
        ("rand4", _initial_random(N, seed=123)),
    ]
    
    for name, pos in init_methods:
        if time.time() > deadline - 20:
            break
        
        pos = np.clip(pos, -0.49, 0.49)
        current_r = _compute_radius(pos)
        
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Use scipy L-BFGS-B optimization
        pos_flat = pos.flatten()
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            return _negative_radius(x)
        
        def jac(x):
            if time.time() > deadline - 5:
                return np.zeros_like(x)
            return _grad_softmin_neg(x)
        
        try:
            result = minimize(
                objective,
                pos_flat,
                jac=jac,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
            )
            
            if result.success or result.fun < 0:
                opt_pos = result.x.reshape(-1, 2)
                opt_pos = np.clip(opt_pos, -0.4999, 0.4999)
                opt_r = _compute_radius(opt_pos)
                
                if opt_r > best_r:
                    best_r = opt_r
                    best_pos = opt_pos.copy()
        except Exception:
            pass
        
        # Dynamics refinement if time permits
        if time.time() < deadline - 15 and best_pos is not None:
            dyn_pos = _dynamics_refine(best_pos, time_budget=10.0)
            dyn_r = _compute_radius(dyn_pos)
            if dyn_r > best_r:
                best_r = dyn_r
                best_pos = dyn_pos.copy()
    
    # Final local refinement
    if time.time() < deadline - 5 and best_pos is not None:
        pos = best_pos.copy()
        current_r = best_r
        
        # Small gradient steps
        step_size = 0.001
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            
            grad = _grad_softmin_neg(pos.flatten())
            grad = grad.reshape(-1, 2)
            
            new_pos = pos + step_size * grad
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            new_r = _compute_radius(new_pos)
            
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
    
    if best_pos is None:
        return _initial_hex_grid(N)
    
    return best_pos
