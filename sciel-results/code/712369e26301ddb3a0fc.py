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
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    This is the primary optimization engine - it naturally handles
    circle packing constraints through repulsion forces.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Track best valid configuration
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.005
    velocity_damp = 0.95
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly to push packing tighter
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Strong repulsion proportional to overlap
                    force = (min_dist - dist) * 50.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces - push circles away from walls
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 100.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 100.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 100.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 100.0
        
        # Update positions with velocity damping
        x += dt * fx
        y += dt * fy
        
        # Clip to valid positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check validity and track best
        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
    
    return best_xyz

def _objective_full(params, penalty_weight=1000.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Full pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _polish_full(xyz, time_budget=5.0):
    """
    Final high-precision polish using full pairwise constraints.
    Uses very tight tolerances and high penalty weights for precision.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_full(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _objective_full(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _local_refine(xyz, time_budget=5.0):
    """Refine a configuration using L-BFGS-B with Delaunay-based constraint reduction."""
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    for penalty_weight in [10.0, 50.0, 100.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            n = N
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            penalty = 0.0
            for i in range(n):
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
            
            pts = np.column_stack([x, y])
            edges = _get_delaunay_edges(pts)
            
            for i, j in edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
            
            return -np.sum(r) + penalty_weight * penalty
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < obj(best_params):
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

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

def _initial_config_hex():
    """Generate an initial configuration using a hexagonal-like arrangement."""
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

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(5):
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
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration using physics relaxation + local refine + polish
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 8.0:
            break
        
        # Phase 1: Physics relaxation
        phys_time = min(5.0, remaining * 0.4)
        if phys_time < 2.0:
            break
        
        try:
            xyz_phys = _physics_relax(xyz0, time_limit=phys_time)
            
            # Phase 2: Local refinement
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            local_time = min(5.0, remaining * 0.3)
            if local_time < 1.0:
                xyz_local = xyz_phys
            else:
                xyz_local = _local_refine(xyz_phys, time_budget=local_time)
            
            # Phase 3: Final polish
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            polish_time = min(5.0, remaining * 0.3)
            if polish_time < 1.0:
                xyz_opt = xyz_local
            else:
                xyz_opt = _polish_full(xyz_local, time_budget=polish_time)
            
            # Validate and fix if necessary
            if not _validate(xyz_opt):
                xyz_fixed = _enforce_validity(xyz_opt)
                xyz_opt = xyz_fixed
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
        except Exception:
            pass
    
    # Final check
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
