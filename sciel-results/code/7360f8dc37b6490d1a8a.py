# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check bounds: each square must be inside [-tol, 1+tol]^2
    half = sides / 2.0
    if np.any(centers[:, 0] - half < -tol) or np.any(centers[:, 0] + half > 1 + tol):
        return False
    if np.any(centers[:, 1] - half < -tol) or np.any(centers[:, 1] + half > 1 + tol):
        return False
    
    # Check non-overlap: for i < j, not (|cx_i - cx_j| < t AND |cy_i - cy_j| < t)
    # where t = (s_i + s_j)/2 - tol
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _physics_relax(centers, sides, max_iters=2000, start_time=None, deadline=None):
    """
    Physics-based relaxation: treat squares as particles with repulsive forces
    when overlapping and attractive forces to grow. This helps escape local minima.
    """
    n = len(sides)
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    # Force parameters
    repulsion_strength = 50.0
    growth_force = 0.1
    damping = 0.95
    dt = 0.01
    tol = 1e-7
    
    for it in range(max_iters):
        if start_time is not None and deadline is not None:
            if time.time() > deadline:
                break
        
        forces_cx = np.zeros(n)
        forces_cy = np.zeros(n)
        forces_s = np.zeros(n)
        
        # Repulsive forces between overlapping squares
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                adx = abs(dx)
                ady = abs(dy)
                
                # Check if overlapping in both dimensions
                if adx < t and ady < t:
                    # Compute penetration depth
                    pen_x = t - adx
                    pen_y = t - ady
                    
                    # Apply repulsive forces
                    if pen_x > 0:
                        f = repulsion_strength * pen_x
                        if dx > 0:
                            forces_cx[i] += f
                            forces_cx[j] -= f
                        elif dx < 0:
                            forces_cx[i] -= f
                            forces_cx[j] += f
                        else:
                            # Random push if perfectly aligned
                            forces_cx[i] += f * 0.1
                            forces_cx[j] -= f * 0.1
                    
                    if pen_y > 0:
                        f = repulsion_strength * pen_y
                        if dy > 0:
                            forces_cy[i] += f
                            forces_cy[j] -= f
                        elif dy < 0:
                            forces_cy[i] -= f
                            forces_cy[j] += f
                        else:
                            forces_cy[i] += f * 0.1
                            forces_cy[j] -= f * 0.1
        
        # Growth forces: try to expand each square
        for i in range(n):
            half = sides[i] / 2.0
            # Check if can grow in each direction
            can_grow = True
            # Check bounds
            if centers[i, 0] - half <= tol or centers[i, 0] + half >= 1 - tol:
                can_grow = False
            if centers[i, 1] - half <= tol or centers[i, 1] + half >= 1 - tol:
                can_grow = False
            
            if can_grow:
                forces_s[i] = growth_force
        
        # Apply forces with damping
        centers[:, 0] += forces_cx * dt
        centers[:, 1] += forces_cy * dt
        sides += forces_s * dt
        
        # Damping on velocities (implicit)
        # Clip to valid range
        half = sides / 2.0
        centers[:, 0] = np.clip(centers[:, 0], half, 1 - half)
        centers[:, 1] = np.clip(centers[:, 1], half, 1 - half)
        sides = np.maximum(sides, 1e-6)
        
        # Check convergence
        if it > 100 and np.max(np.abs(forces_cx)) < 1e-6 and np.max(np.abs(forces_cy)) < 1e-6:
            break
    
    return centers, sides

def _local_refine(centers, sides, start_time=None, deadline=None):
    """
    Local refinement using gradient-based optimization.
    """
    n = len(sides)
    tol = 1e-7
    
    def objective(params):
        sides_p = params[:10]
        cx = params[10:20]
        cy = params[20:30]
        
        if np.any(sides_p <= 0):
            return 1e10
        
        penalty = 0.0
        half = sides_p / 2.0
        penalty += np.sum(np.maximum(0, tol - (cx - half)) * 5000)
        penalty += np.sum(np.maximum(0, (cx + half) - (1 - tol)) * 5000)
        penalty += np.sum(np.maximum(0, tol - (cy - half)) * 5000)
        penalty += np.sum(np.maximum(0, (cy + half) - (1 - tol)) * 5000)
        
        for i in range(10):
            for j in range(i + 1, 10):
                t = (sides_p[i] + sides_p[j]) / 2.0 - tol
                overlap_x = np.maximum(0, t - abs(cx[i] - cx[j]))
                overlap_y = np.maximum(0, t - abs(cy[i] - cy[j]))
                if overlap_x > 0 and overlap_y > 0:
                    penalty += (overlap_x * overlap_y) * 5000
        
        return -np.sum(sides_p) + penalty
    
    params0 = np.concatenate([sides, centers[:, 0], centers[:, 1]])
    
    try:
        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if result.success:
            params_opt = result.x
            sides_opt = np.maximum(params_opt[:10], 1e-6)
            cx_opt = np.clip(params_opt[10:20], 0, 1)
            cy_opt = np.clip(params_opt[20:30], 0, 1)
            centers_opt = np.stack([cx_opt, cy_opt], axis=1)
            
            if _is_valid(centers_opt, sides_opt):
                return centers_opt, sides_opt
    except Exception:
        pass
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares in the unit square.
    
    Uses physics-based relaxation to escape local minima, followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10-second safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Strategy 1: Physics-based relaxation from various initial configurations
    rng = np.random.RandomState(42)
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Grid 3x3 + 1 (9 squares in 3x3 grid, 1 in center)
    k = 3
    side = 1.0 / k
    for i in range(9):
        cols = i % k
        rows = i // k
        initial_configs.append({
            'cx': (cols + 0.5) * side,
            'cy': (rows + 0.5) * side,
            's': side
        })
    # 10th square in the middle
    initial_configs.append({'cx': 0.5, 'cy': 0.5, 's': 0.1})
    
    # 2. Random initial configurations
    for _ in range(5):
        cx = rng.uniform(0.2, 0.8, size=10)
        cy = rng.uniform(0.2, 0.8, size=10)
        s = rng.uniform(0.1, 0.3, size=10)
        for i in range(10):
            initial_configs.append({'cx': cx[i], 'cy': cy[i], 's': s[i]})
    
    # 3. Perturbed grid
    k = 3
    side = 1.0 / k
    for i in range(9):
        cols = i % k
        rows = i // k
        cx = (cols + 0.5) * side + rng.normal(0, 0.05)
        cy = (rows + 0.5) * side + rng.normal(0, 0.05)
        s = side + rng.normal(0, 0.02)
        initial_configs.append({'cx': np.clip(cx, 0.05, 0.95), 'cy': np.clip(cy, 0.05, 0.95), 's': max(0.05, s)})
    initial_configs.append({'cx': 0.5, 'cy': 0.5, 's': 0.1})
    
    # Try each initial configuration
    for config_idx in range(len(initial_configs)):
        if time.time() > deadline:
            break
        
        if config_idx < 10:
            # Grid-based
            centers = np.array([[initial_configs[i]['cx'], initial_configs[i]['cy']] for i in range(10)], dtype=np.float64)
            sides = np.array([initial_configs[i]['s'] for i in range(10)], dtype=np.float64)
        else:
            # Random-based
            base_idx = config_idx - 10
            if base_idx < 5:
                centers = np.array([[initial_configs[10 + base_idx]['cx'], initial_configs[10 + base_idx]['cy']] for i in range(10)], dtype=np.float64)
                sides = np.array([initial_configs[10 + base_idx]['s'] for i in range(10)], dtype=np.float64)
            else:
                # Perturbed grid
                centers = np.array([[initial_configs[15 + base_idx - 5]['cx'], initial_configs[15 + base_idx - 5]['cy']] for i in range(10)], dtype=np.float64)
                sides = np.array([initial_configs[15 + base_idx - 5]['s'] for i in range(10)], dtype=np.float64)
        
        # Physics relaxation
        centers_relaxed, sides_relaxed = _physics_relax(centers, sides, max_iters=1500, start_time=start_time, deadline=deadline)
        
        # Check if valid after relaxation
        update_best(centers_relaxed, sides_relaxed)
        
        # Local refinement
        if time.time() < deadline - 30:
            centers_refined, sides_refined = _local_refine(centers_relaxed, sides_relaxed, start_time=start_time, deadline=deadline)
            update_best(centers_refined, sides_refined)
    
    # Strategy 2: Multi-start gradient optimization
    if time.time() < deadline - 60:
        def objective(params):
            sides_p = params[:10]
            cx = params[10:20]
            cy = params[20:30]
            
            if np.any(sides_p <= 0):
                return 1e10
            
            penalty = 0.0
            half = sides_p / 2.0
            penalty += np.sum(np.maximum(0, tol - (cx - half)) * 5000)
            penalty += np.sum(np.maximum(0, (cx + half) - (1 - tol)) * 5000)
            penalty += np.sum(np.maximum(0, tol - (cy - half)) * 5000)
            penalty += np.sum(np.maximum(0, (cy + half) - (1 - tol)) * 5000)
            
            for i in range(10):
                for j in range(i + 1, 10):
                    t = (sides_p[i] + sides_p[j]) / 2.0 - tol
                    overlap_x = np.maximum(0, t - abs(cx[i] - cx[j]))
                    overlap_y = np.maximum(0, t - abs(cy[i] - cy[j]))
                    if overlap_x > 0 and overlap_y > 0:
                        penalty += (overlap_x * overlap_y) * 5000
            
            return -np.sum(sides_p) + penalty
        
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Start from best found so far
                if best_centers is not None:
                    params_init = np.concatenate([best_sides, best_centers[:, 0], best_centers[:, 1]])
                else:
                    k = 3
                    side = 1.0 / k
                    idx = np.arange(9)
                    cols = idx % k
                    rows = idx // k
                    cx0 = np.concatenate([(cols + 0.5) * side, [0.5]])
                    cy0 = np.concatenate([(rows + 0.5) * side, [0.5]])
                    sides0 = np.concatenate([np.full(9, side), [0.1]])
                    params_init = np.concatenate([sides0, cx0, cy0])
            else:
                # Random initialization
                sides_init = rng.uniform(0.05, 0.35, size=10)
                cx_init = rng.uniform(0.1, 0.9, size=10)
                cy_init = rng.uniform(0.1, 0.9, size=10)
                params_init = np.concatenate([sides_init, cx_init, cy_init])
            
            params_init = np.clip(params_init, 0.01, None)
            params_init[10:30] = np.clip(params_init[10:30], 0.0, 1.0)
            
            try:
                result = minimize(
                    objective,
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                params_opt = result.x
                sides_opt = np.maximum(params_opt[:10], 1e-6)
                cx_opt = np.clip(params_opt[10:20], 0, 1)
                cy_opt = np.clip(params_opt[20:30], 0, 1)
                centers_opt = np.stack([cx_opt, cy_opt], axis=1)
                
                update_best(centers_opt, sides_opt)
            except Exception:
                continue
    
    # If no valid solution found, return the grid baseline
    if best_centers is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(9)
        cols = idx % k
        rows = idx // k
        cx = np.concatenate([(cols + 0.5) * side, [0.5]])
        cy = np.concatenate([(rows + 0.5) * side, [0.5]])
        sides = np.concatenate([np.full(9, side), [0.1]])
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = sides.astype(np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
