"""Packing 200 equal circles in a unit square: Force-Directed + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
DEADLINE = 140.0  # safety margin before 150s kill


def _score(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(walls)
    
    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)


def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    best = None
    best_r = -1
    
    for orientation in [0, np.pi/6, np.pi/3, np.pi/12, np.pi/4]:
        m = int(np.ceil(np.sqrt(n)))
        
        for scale in np.linspace(0.85, 1.15, 30):
            s = scale / m
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m):
                for j in range(m):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            max_abs = np.max(np.abs(pts))
            if max_abs > 0.48:
                pts = pts * (0.48 / max_abs)
            
            r = _score(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    return best if best is not None else _square_init(n)


def _random_init(n, seed=42):
    """Random initialization within the square."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.48, 0.48, size=(n, 2))
    return pts


def _square_init(n):
    """Fallback square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _force_directed(centers, time_limit=10.0):
    """
    Force-directed relaxation: push overlapping circles apart, pull toward center if near walls.
    Uses inverse-square repulsion between circles and wall repulsion.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    # Adaptive parameters
    repulsion_strength = 1.0
    wall_strength = 0.5
    damping = 0.9
    velocity = np.zeros_like(centers)
    
    # Estimate typical spacing
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    typical_dist = np.median(dists[:, 1])
    if typical_dist < 1e-6:
        typical_dist = 0.05
    
    # Scale forces relative to typical distance
    force_scale = 1.0 / (typical_dist * typical_dist)
    
    while time.time() - start_time < time_limit:
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        tree = cKDTree(centers)
        threshold = 3 * typical_dist
        pairs = tree.query_pairs(threshold)
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random nudge
                forces[i] += np.array([np.random.randn(), np.random.randn()]) * 0.01
                continue
            # Inverse square repulsion, stronger when closer
            repulsion = force_scale * repulsion_strength / (dist * dist)
            forces[i] += repulsion * diff / dist
            forces[j] -= repulsion * diff / dist
        
        # Wall repulsion
        for i in range(n):
            for axis in range(2):
                val = centers[i, axis]
                # Distance to nearest wall in this axis
                dist_to_wall = min(0.5 - val, val + 0.5)
                if dist_to_wall < typical_dist:
                    # Push away from wall
                    if val > 0:
                        forces[i, axis] -= wall_strength * force_scale * (1.0 - dist_to_wall/typical_dist)
                    else:
                        forces[i, axis] += wall_strength * force_scale * (1.0 - dist_to_wall/typical_dist)
        
        # Update velocity with damping
        velocity = damping * velocity + forces * 0.01
        centers += velocity
        
        # Clip to square with margin
        centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
        centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
        
        # Decay forces over time
        repulsion_strength *= 0.999
        wall_strength *= 0.999
    
    return centers


def _soft_min_neg(centers, temperature=0.01):
    """
    Compute a smooth approximation of the negative of the minimum constraint.
    The constraint is: min(wall_dist, pair_dist/2) >= 0
    We want to minimize the violation: max(0, -min_constraint)
    
    Using soft min: -log(sum(exp(-x_i/T))) * T approximates min(x_i)
    """
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pair distances / 2
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([walls, pair_dists])
    
    # Soft min approximation
    # To avoid overflow, shift by min
    c_min = np.min(all_constraints)
    shifted = all_constraints - c_min
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = c_min - temperature * log_sum
    
    # We want to minimize violation = max(0, -soft_min)
    # For optimization, we minimize -soft_min (i.e., maximize soft_min)
    # But L-BFGS-B minimizes, so objective = -soft_min
    return -soft_min


def _soft_min_neg_grad(centers, temperature=0.01):
    """
    Compute gradient of the soft-min objective.
    """
    n = len(centers)
    grad = np.zeros(2 * n)
    
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pair distances / 2
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_constraints = np.concatenate([walls, pair_dists])
    
    c_min = np.min(all_constraints)
    shifted = all_constraints - c_min
    weights = np.exp(-shifted / temperature)
    weights /= np.sum(weights)
    
    # Gradient of soft_min = sum(weights_i * grad(constraint_i))
    # But we're minimizing -soft_min, so grad_objective = -sum(weights_i * grad(constraint_i))
    
    # Wall gradients
    for i in range(n):
        # Right wall: 0.5 - x
        # Left wall: x + 0.5
        # Top wall: 0.5 - y
        # Bottom wall: y + 0.5
        x, y = centers[i]
        w_right = 0.5 - x
        w_left = x + 0.5
        w_top = 0.5 - y
        w_bottom = y + 0.5
        
        # The wall constraint is min of these 4
        # We already computed walls[i] = min of these
        # For gradient, we need to know which wall is active
        # Simplification: use the soft-min weights over all constraints
        
        # For now, approximate: if center is near a wall, push away
        if w_right < 0.1:
            grad[2*i] += weights[i] * (-1.0)  # d/dx (0.5-x) = -1, but we want to increase wall dist
            # Actually, we want to maximize wall dist, so if w_right is small, we want to decrease x
            # grad of objective (-soft_min) w.r.t. x: if right wall is active, increasing x decreases wall dist
            # So d(-soft_min)/dx = -weights * d(soft_min)/dx = -weights * (d(w_right)/dx) = -weights * (-1) = weights
            # But this is getting complicated. Let's use a simpler approach.
    
    # This gradient computation is complex. Let's use numerical differentiation instead
    # for reliability, or use a simpler objective.
    
    # Alternative: use the actual minimum constraint and its gradient
    # Find which constraint is minimum
    min_idx = np.argmin(all_constraints)
    
    if min_idx < n:
        # Wall constraint
        i = min_idx
        x, y = centers[i]
        w_right = 0.5 - x
        w_left = x + 0.5
        w_top = 0.5 - y
        w_bottom = y + 0.5
        
        # Find which wall is the minimum
        wall_vals = [w_right, w_left, w_top, w_bottom]
        wall_grads = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        min_wall = np.argmin(wall_vals)
        
        # Gradient of -soft_min w.r.t. centers[i]
        # d(-soft_min)/d(centers[i]) = -weights[min_idx] * wall_grads[min_wall]
        # But since we're using hard min approximation, weights[min_idx] ≈ 1
        gx, gy = wall_grads[min_wall]
        grad[2*i] = -gx
        grad[2*i+1] = -gy
    else:
        # Pair constraint
        pair_idx = min_idx - n
        i = pair_idx
        j = idx[i, 1]
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            unit = diff / dist
            # d(pair_dist)/d(centers[i]) = unit
            # d(-soft_min)/d(centers[i]) = -unit
            grad[2*i:2*i+2] = -unit
            # d(pair_dist)/d(centers[j]) = -unit
            # d(-soft_min)/d(centers[j]) = unit
            grad[2*j:2*j+2] = unit
    
    return grad


def _optimize_lbfgs(centers, time_limit=30.0, temperature=0.01):
    """
    Use L-BFGS-B to minimize the soft-min objective.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Timeout")
        c = x.reshape(N, 2)
        return _soft_min_neg(c, temperature)
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Timeout")
        c = x.reshape(N, 2)
        return _soft_min_neg_grad(c, temperature)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        new_centers = result.x.reshape(N, 2)
        return new_centers
        
    except Exception:
        return centers


def _active_set_polish(centers, time_limit=20.0):
    """
    Active set polish: identify the binding constraints and optimize only those.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _score(best_centers)
    n = len(centers)
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        current_r = _score(best_centers)
        
        # Find binding constraints
        walls = np.minimum(
            np.minimum(0.5 - best_centers[:, 0], best_centers[:, 0] + 0.5),
            np.minimum(0.5 - best_centers[:, 1], best_centers[:, 1] + 0.5)
        )
        tree = cKDTree(best_centers)
        dists, idx = tree.query(best_centers, k=2)
        pair_dists = dists[:, 1] / 2.0
        
        # Find pairs with distance close to 2*current_r
        active_pairs = np.where(dists[:, 1] < 2 * current_r * 1.01)[0]
        active_walls = np.where(walls < current_r * 1.01)[0]
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Try to move active points to increase the minimum
        improved = False
        
        # Move pairs apart
        for i in active_pairs:
            j = idx[i, 1]
            diff = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                continue
            step = 0.001
            new_i = best_centers[i] + step * diff / norm
            new_j = best_centers[j] - step * diff / norm
            
            # Check if this improves the score
            test_centers = best_centers.copy()
            test_centers[i] = new_i
            test_centers[j] = new_j
            test_r = _score(test_centers)
            
            if test_r > best_r + 1e-12:
                best_centers[i] = new_i
                best_centers[j] = new_j
                best_r = test_r
                improved = True
        
        # Move wall-bound points away from wall
        for i in active_walls:
            x, y = best_centers[i]
            # Determine direction away from nearest wall
            dx = 0
            dy = 0
            if x > 0.4:
                dx = -1
            elif x < -0.4:
                dx = 1
            if y > 0.4:
                dy = -1
            elif y < -0.4:
                dy = 1
            
            if dx == 0 and dy == 0:
                # Corner case
                dx = -np.sign(x) if x != 0 else 1
                dy = -np.sign(y) if y != 0 else 1
            
            norm = np.sqrt(dx*dx + dy*dy)
            step = 0.001 / norm
            new_pos = best_centers[i] + step * np.array([dx, dy])
            
            test_centers = best_centers.copy()
            test_centers[i] = new_pos
            test_r = _score(test_centers)
            
            if test_r > best_r + 1e-12:
                best_centers[i] = new_pos
                best_r = test_r
                improved = True
        
        if not improved:
            break
    
    return best_centers, best_r


def _high_precision_polish(centers, time_limit=10.0):
    """
    High-precision polish using very small steps to squeeze out last bits of radius.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _score(best_centers)
    
    step_size = 0.0001
    
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Find the bottleneck
        walls = np.minimum(
            np.minimum(0.5 - best_centers[:, 0], best_centers[:, 0] + 0.5),
            np.minimum(0.5 - best_centers[:, 1], best_centers[:, 1] + 0.5)
        )
        wall_min = np.min(walls)
        wall_argmin = np.argmin(walls)
        
        tree = cKDTree(best_centers)
        dists, idx = tree.query(best_centers, k=2)
        pair_min = np.min(dists[:, 1]) / 2.0
        pair_argmin = np.argmin(dists[:, 1])
        
        current_r = min(wall_min, pair_min)
        
        if current_r > best_r + 1e-14:
            best_r = current_r
        
        # Try to improve the bottleneck
        if wall_min <= pair_min:
            i = wall_argmin
            x, y = best_centers[i]
            # Move away from nearest wall
            dx = 0
            dy = 0
            if 0.5 - x < x + 0.5:
                dx = -1
            else:
                dx = 1
            if 0.5 - y < y + 0.5:
                dy = -1
            else:
                dy = 1
            
            norm = np.sqrt(dx*dx + dy*dy)
            new_pos = best_centers[i] + (step_size / norm) * np.array([dx, dy])
            
            test_centers = best_centers.copy()
            test_centers[i] = new_pos
            test_r = _score(test_centers)
            
            if test_r > best_r:
                best_centers[i] = new_pos
                best_r = test_r
        else:
            i = pair_argmin
            j = idx[i, 1]
            diff = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                break
            
            new_i = best_centers[i] + step_size * diff / norm
            new_j = best_centers[j] - step_size * diff / norm
            
            test_centers = best_centers.copy()
            test_centers[i] = new_i
            test_centers[j] = new_j
            test_r = _score(test_centers)
            
            if test_r > best_r:
                best_centers[i] = new_i
                best_centers[j] = new_j
                best_r = test_r
        
        # Gradually decrease step size
        step_size *= 0.999
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    
    # Phase 1: Generate multiple initializations and pick the best
    candidates = []
    
    # Hexagonal lattice
    hex_centers = _hex_init(N)
    candidates.append(hex_centers)
    
    # Random initializations
    for seed in [42, 123, 456]:
        random_centers = _random_init(N, seed=seed)
        # Quick force-directed relaxation
        fd_centers = _force_directed(random_centers, time_limit=3.0)
        candidates.append(fd_centers)
    
    # Pick the best candidate
    best_centers = None
    best_r = -1
    for cand in candidates:
        r = _score(cand)
        if r > best_r:
            best_r = r
            best_centers = cand.copy()
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 2: Force-directed relaxation on the best candidate
    if remaining > 15:
        fd_time = min(remaining * 0.3, 15.0)
        fd_centers = _force_directed(best_centers, time_limit=fd_time)
        fd_r = _score(fd_centers)
        if fd_r > best_r:
            best_r = fd_r
            best_centers = fd_centers
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 3: L-BFGS-B optimization with different temperatures
    if remaining > 20:
        temps = [0.05, 0.02, 0.01, 0.005]
        for temp in temps:
            if time.time() - start_time > DEADLINE - 10:
                break
            lbfgs_time = min(remaining * 0.25, 20.0)
            lbfgs_centers = _optimize_lbfgs(best_centers, time_limit=lbfgs_time, temperature=temp)
            lbfgs_r = _score(lbfgs_centers)
            if lbfgs_r > best_r:
                best_r = lbfgs_r
                best_centers = lbfgs_centers
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 4: Active set polish
    if remaining > 10:
        asp_time = min(remaining * 0.5, 20.0)
        asp_centers, asp_r = _active_set_polish(best_centers, time_limit=asp_time)
        if asp_r > best_r:
            best_r = asp_r
            best_centers = asp_centers
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 5: High-precision polish
    if remaining > 5:
        hp_time = min(remaining - 5, 10.0)
        hp_centers, hp_r = _high_precision_polish(best_centers, time_limit=hp_time)
        if hp_r > best_r:
            best_r = hp_r
            best_centers = hp_centers
    
    # Final validation and clipping
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    return best_centers
