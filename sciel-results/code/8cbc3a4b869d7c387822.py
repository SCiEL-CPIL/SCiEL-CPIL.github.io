"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        n = centers.shape[0]
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _objective_delaunay(params, penalty_weight=50.0):
    """
    Objective using Delaunay-based constraints.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edge_set = set(edges)
    
    # Overlap penalties for Delaunay neighbors
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2
    
    # Also check a few nearest neighbors not in Delaunay to be safe
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            # Only check if potentially close
            max_r = max(r[i], r[j])
            if dist_sq < (4.0 * max_r * max_r):
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight=100.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _physics_relax(xyz, n_iters=200, time_budget=15.0):
    """
    Physics-based PBD relaxation:
    - Circles are particles with mass ~ r^2
    - Global outward pressure force pushes circles apart and toward boundaries
    - Gauss-Seidel relaxation for distance constraints and boundary containment
    - Radii are grown iteratively as space allows
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    n = xyz.shape[0]
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Mass proportional to radius squared (with a floor to avoid zero mass)
    mass = np.maximum(r ** 2, 1e-6)
    inv_mass = 1.0 / mass
    
    # Adaptive time step
    dt = 0.05
    damping = 0.9
    
    # Velocity arrays
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion (soft spring for overlaps)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Overlap: strong repulsive force
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    nx = dx / dist
                    ny = dy / dist
                    fx[i] += force * nx
                    fy[i] += force * ny
                    fx[j] -= force * nx
                    fy[j] -= force * ny
                else:
                    # Slight attraction to pack tightly (very weak)
                    gap = dist - min_dist
                    if gap < 0.05:
                        force = -0.1 * gap
                        nx = dx / dist
                        ny = dy / dist
                        fx[i] += force * nx
                        fy[i] += force * ny
                        fx[j] -= force * nx
                        fy[j] -= force * ny
        
        # Boundary forces: push circles toward center if too close to edge
        for i in range(n):
            margin = 0.01
            if x[i] - r[i] < margin:
                fx[i] += 5.0 * (margin - (x[i] - r[i]))
            if x[i] + r[i] > 1 - margin:
                fx[i] -= 5.0 * ((x[i] + r[i]) - (1 - margin))
            if y[i] - r[i] < margin:
                fy[i] += 5.0 * (margin - (y[i] - r[i]))
            if y[i] + r[i] > 1 - margin:
                fy[i] -= 5.0 * ((y[i] + r[i]) - (1 - margin))
        
        # Update velocities with damping
        vx = damping * vx + dt * fx * inv_mass
        vy = damping * vy + dt * fy * inv_mass
        
        # Update positions
        x += dt * vx
        y += dt * vy
        
        # Gauss-Seidel constraint projection: enforce non-overlap
        for _ in range(3):  # Multiple passes for convergence
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Move apart proportional to inverse mass
                        wi = inv_mass[i] / (inv_mass[i] + inv_mass[j])
                        wj = inv_mass[j] / (inv_mass[i] + inv_mass[j])
                        x[i] += overlap * nx * wi
                        y[i] += overlap * ny * wi
                        x[j] -= overlap * nx * wj
                        y[j] -= overlap * ny * wj
            
            # Enforce boundary containment
            for i in range(n):
                if x[i] - r[i] < 0:
                    x[i] = r[i]
                if x[i] + r[i] > 1:
                    x[i] = 1 - r[i]
                if y[i] - r[i] < 0:
                    y[i] = r[i]
                if y[i] + r[i] > 1:
                    y[i] = 1 - r[i]
        
        # Try to grow radii slightly if there's room
        # Check if we can increase any radius by a small amount
        grow_amount = 0.001 * (1.0 - it / n_iters)  # Decreasing growth
        for i in range(n):
            # Check boundary constraints
            max_r_boundary = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] + grow_amount > max_r_boundary:
                continue
            
            # Check pairwise constraints
            can_grow = True
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] + grow_amount:
                    can_grow = False
                    break
            
            if can_grow:
                r[i] += grow_amount
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
    
    return xyz

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _optimize_delaunay(xyz, time_budget=20.0):
    """
    Optimize using Delaunay-based constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Delaunay-based optimization
    try:
        result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0,),
                          options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, 50.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.6:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(100.0,),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < _objective_full(best_params, 100.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.22 * np.cos(angle)
        y = 0.5 + 0.22 * np.sin(angle)
        positions.append([x, y, 0.09])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.08])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.48 * np.cos(angle)
        y = 0.5 + 0.48 * np.sin(angle)
        positions.append([x, y, 0.07])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Apply physics relaxation first to get a better starting point
        time_for_physics = min(8.0, remaining * 0.25)
        xyz_phys = _physics_relax(xyz0, n_iters=150, time_budget=time_for_physics)
        
        # Then run the Delaunay-based optimizer
        time_for_opt = min(20.0, remaining - 2.0 - time_for_physics)
        if time_for_opt < 2.0:
            break
        
        xyz_opt = _optimize_delaunay(xyz_phys, time_budget=time_for_opt)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score and _validate(xyz_opt):
            best_score = score
            best_xyz = xyz_opt
    
    # Final high-precision polish on the best configuration
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 5.0:
        try:
            params = best_xyz.flatten()
            bounds = []
            for i in range(N):
                bounds.append((0, 1))
                bounds.append((0, 1))
                bounds.append((0, 0.5))
            
            result = minimize(_objective_full, params, method='L-BFGS-B',
                              bounds=bounds, args=(200.0,),
                              options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < _objective_full(params, 200.0):
                polished = result.x.reshape(N, 3)
                if _validate(polished) and np.sum(polished[:, 2]) > best_score:
                    best_score = np.sum(polished[:, 2])
                    best_xyz = polished
        except Exception:
            pass
    
    # Final check
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
