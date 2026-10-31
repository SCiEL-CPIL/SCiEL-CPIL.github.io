"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        # Better hexagonal packing
        # 5 rows of 5, but adjust to fit better
        # Standard hex packing for 25 circles is dense.
        # Let's try a specific layout known to be good for 26.
        # Corners are usually large.
        
        # Corner circles
        r_corner = 0.15
        pos[0] = [r_corner, r_corner]
        radii[0] = r_corner
        pos[1] = [1.0 - r_corner, r_corner]
        radii[1] = r_corner
        pos[2] = [r_corner, 1.0 - r_corner]
        radii[2] = r_corner
        pos[3] = [1.0 - r_corner, 1.0 - r_corner]
        radii[3] = r_corner
        
        # Edge circles
        r_edge = 0.12
        # Top edge
        pos[4] = [0.5, 1.0 - r_edge]
        radii[4] = r_edge
        # Bottom edge
        pos[5] = [0.5, r_edge]
        radii[5] = r_edge
        # Left edge
        pos[6] = [r_edge, 0.5]
        radii[6] = r_edge
        # Right edge
        pos[7] = [1.0 - r_edge, 0.5]
        radii[7] = r_edge
        
        # Center and remaining
        # We have 18 left.
        # Place them in a loose hex pattern in the center
        remaining = N - 8
        # Simple grid for the rest to start, then optimize
        side = int(np.sqrt(remaining)) + 1
        for i in range(side):
            for j in range(side):
                idx = 8 + i * side + j
                if idx < N:
                    # Scale to fit in the central region [0.2, 0.8]
                    pos[idx] = [0.2 + 0.6 * (i + 0.5) / side, 0.2 + 0.6 * (j + 0.5) / side]
                    radii[idx] = 0.08
        
        return pos, radii
    
    elif strategy == 'center_ring':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        pos[0] = [0.5, 0.5]
        radii[0] = 0.15
        r_inner = 0.12
        dist_inner = 0.3
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
            radii[i+1] = r_inner
        r_outer = 0.08
        dist_outer = 0.42
        for i in range(17):
            angle = 2 * np.pi * i / 17
            pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
            radii[i+9] = r_outer
        return pos, radii
    
    elif strategy == 'random':
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.98)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.98))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.98))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.98))
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both to resolve overlap
                    # Scale factor such that r_i + r_j = dist * 0.98
                    # r_i' = r_i * scale, r_j' = r_j * scale
                    # (r_i + r_j) * scale = dist * 0.98
                    # scale = dist * 0.98 / (r_i + r_j)
                    if min_dist > 1e-9:
                        scale = (dist * 0.98) / min_dist
                    else:
                        scale = 0.5
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii

def _objective_and_grad(x):
    """
    Objective: -sum(radii) + penalty(violations)
    x = [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    obj = -np.sum(radii)
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0
    
    penalty = 0.0
    pen_grad = np.zeros_like(x)
    
    # Boundary violations
    # pos[i, 0] >= r[i]  => v = r[i] - pos[i, 0]
    # pos[i, 0] <= 1 - r[i] => v = pos[i, 0] + r[i] - 1
    for i in range(n):
        for coord in range(2):
            # Left/Bottom
            v = radii[i] - pos[i, coord]
            if v > 0:
                penalty += v * v
                pen_grad[2*n+i] += 2 * v
                pen_grad[2*i+coord] -= 2 * v
            
            # Right/Top
            v = pos[i, coord] + radii[i] - 1.0
            if v > 0:
                penalty += v * v
                pen_grad[2*i+coord] += 2 * v
                pen_grad[2*n+i] += 2 * v

    # Overlap violations
    # dist(i,j) >= r[i] + r[j]
    # v = (r[i] + r[j]) - dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            
            min_dist = radii[i] + radii[j]
            v = min_dist - dist
            
            if v > 0:
                penalty += v * v
                
                # d(v)/d(r_i) = 1, d(v)/d(r_j) = 1
                # d(v)/d(pos_i) = - (dx/dist, dy/dist)
                # d(v)/d(pos_j) = (dx/dist, dy/dist)
                
                factor = 2 * v
                pen_grad[2*n+i] += factor
                pen_grad[2*n+j] += factor
                
                gx = dx / dist
                gy = dy / dist
                
                pen_grad[2*i] -= factor * gx
                pen_grad[2*i+1] -= factor * gy
                pen_grad[2*j] += factor * gx
                pen_grad[2*j+1] += factor * gy

    # Scale penalty to dominate objective initially, then relax
    # A fixed scale of 100 is usually good for L-BFGS-B
    scale = 100.0
    obj += scale * penalty
    grad += scale * pen_grad
    
    return obj, grad

def _run_optimization(pos, radii, deadline):
    """Run L-BFGS-B optimization."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.4)) # Max radius
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
        )
        
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = x_opt[2*N:]
        
        # Post-process to ensure strict validity
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None

def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.01 * np.exp(-0.005 * it)
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            margin = 0.005
            if pos[i, 0] < radii[i] + margin:
                forces[i, 0] += 10.0 * (radii[i] + margin - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - margin:
                forces[i, 0] -= 10.0 * (pos[i, 0] - (1.0 - radii[i] - margin))
            if pos[i, 1] < radii[i] + margin:
                forces[i, 1] += 10.0 * (radii[i] + margin - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - margin:
                forces[i, 1] -= 10.0 * (pos[i, 1] - (1.0 - radii[i] - margin))
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 15.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.2 and dist > 1e-6:
                    force_mag = 2.0 * (min_dist * 1.2 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.85 * vel + 0.15 * forces
        pos += vel * 0.008 + rng.normal(0, temp, size=pos.shape)
        
        # Clip positions
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Grow radii where possible
        for i in range(N):
            max_grow = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0005:
                radii[i] += max_grow * 0.6
        
        # Ensure minimum radius
        radii = np.maximum(radii, 0.002)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['hexagonal', 'grid', 'center_ring', 'random']
    
    # Phase 1: Multi-start with gradient optimization
    for strategy in strategies:
        if time.time() > deadline - 30:
            break
        
        # Try 2 seeds for each strategy
        for seed in range(2):
            if time.time() > deadline - 25:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # Quick optimization
            pos_opt, radii_opt = _run_optimization(pos, radii, deadline)
            
            if pos_opt is not None:
                score = np.sum(radii_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = radii_opt.copy()
    
    # Phase 2: Force-directed annealing from best result
    if best_pos is not None and time.time() < deadline - 20:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=500, deadline=deadline - 15)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: Final gradient polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_pol, radii_pol = _run_optimization(best_pos, best_radii, deadline)
        if pos_pol is not None:
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
