import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26
DEADLINE = 145.0  # Safety margin before 150s kill

def _validity_check(pos, radii, tol=1e-8):
    """Check if configuration is valid."""
    for i in range(N):
        if pos[i][0] - radii[i] < -tol or pos[i][0] + radii[i] > 1 + tol:
            return False
        if pos[i][1] - radii[i] < -tol or pos[i][1] + radii[i] > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _enforce_validity(pos, radii):
    """Make configuration valid by shrinking radii if necessary."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions
    pos = np.clip(pos, 0, 1)
    
    # Clamp radii to wall distances
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(500):
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                overlap = radii[i] + radii[j] - d
                if overlap > 1e-10:
                    overlap_found = True
                    # Shrink the smaller circle
                    if radii[i] <= radii[j]:
                        shrink = min(overlap, radii[i] - 1e-6)
                        radii[i] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[j] -= min(rem, radii[j] - 1e-6)
                    else:
                        shrink = min(overlap, radii[j] - 1e-6)
                        radii[j] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[i] -= min(rem, radii[i] - 1e-6)
        if not overlap_found:
            break
    
    # Final wall check
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    return pos, radii

def _score_valid(pos, radii):
    """Return score if valid, else -inf."""
    if _validity_check(pos, radii):
        return np.sum(radii)
    return -np.inf

def _hex_initial(rng):
    """Initialize with hexagonal packing pattern."""
    # Hexagonal grid with offset rows
    spacing_x = 1.0 / 5.5
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    centers = []
    for row in range(6):
        for col in range(6):
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            if row % 2 == 1:
                x += spacing_x / 2
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
    
    centers = np.array(centers)
    if len(centers) < N:
        # Fallback to grid
        return _grid_initial(rng)
    
    # Select N most central points
    d2c = np.sum((centers - 0.5)**2, axis=1)
    idx = np.argsort(d2c)[:N]
    pos = centers[idx].copy()
    
    # Add small jitter
    pos += rng.uniform(-0.005, 0.005, size=pos.shape)
    pos = np.clip(pos, 0.02, 0.98)
    
    return pos

def _grid_initial(rng, cols=6, rows=5):
    """Initialize with grid packing."""
    xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    centers = []
    for x in xs:
        for y in ys:
            centers.append([x, y])
    centers = np.array(centers)
    
    if len(centers) < N:
        # Fallback
        return np.random.RandomState(42).uniform(0.1, 0.9, size=(N, 2))
    
    # Select N most central points
    d2c = np.sum((centers - 0.5)**2, axis=1)
    idx = np.argsort(d2c)[:N]
    pos = centers[idx].copy()
    
    # Add small jitter
    pos += rng.uniform(-0.005, 0.005, size=pos.shape)
    pos = np.clip(pos, 0.02, 0.98)
    
    return pos

def _voronoi_initial(rng):
    """Initialize using Voronoi-based approach."""
    # Start with random points and use Voronoi to estimate max radii
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    
    try:
        vor = Voronoi(pos)
        # For each point, find the minimum distance to Voronoi vertices
        radii = np.zeros(N)
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or len(region) == 0:
                radii[i] = 0.02
                continue
            vertices = vor.vertices[region]
            # Filter to vertices within [0,1]^2
            valid = (vertices[:, 0] >= 0) & (vertices[:, 0] <= 1) & \
                    (vertices[:, 1] >= 0) & (vertices[:, 1] <= 1)
            if np.sum(valid) == 0:
                radii[i] = 0.02
            else:
                d = np.min(np.linalg.norm(vertices[valid] - pos[i], axis=1))
                radii[i] = max(d * 0.95, 0.01)
        
        # Also consider wall distances
        for i in range(N):
            wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            radii[i] = min(radii[i], wall_dist * 0.95)
        
        # Ensure no overlaps by iterative shrinking
        pos, radii = _enforce_validity(pos, radii)
        return pos
        
    except Exception:
        return _grid_initial(rng)

def _penalty_objective(params, radii, weight):
    """Penalty objective: minimize negative sum of radii + penalty for violations."""
    pos = params.reshape(N, 2)
    score = np.sum(radii)
    
    penalty = 0.0
    
    # Pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                penalty += overlap**2
    
    # Wall violations
    for i in range(N):
        pen = max(0, radii[i] - pos[i][0])
        penalty += pen**2
        pen = max(0, pos[i][0] + radii[i] - 1)
        penalty += pen**2
        pen = max(0, radii[i] - pos[i][1])
        penalty += pen**2
        pen = max(0, pos[i][1] + radii[i] - 1)
        penalty += pen**2
    
    return -score + weight * penalty

def _penalty_grad(params, radii, weight):
    """Gradient of penalty objective."""
    pos = params.reshape(N, 2)
    grad = np.zeros_like(params)
    
    # Score gradient: d(-sum(radii))/dpos = 0 (radii are fixed in this formulation)
    # Actually we want to maximize sum(radii), but radii are fixed here.
    # The optimization is over positions only, with radii fixed.
    # So the objective is -sum(radii) + penalty, and gradient is just penalty gradient.
    
    # Pairwise overlap gradients
    for i in range(N):
        for j in range(i+1, N):
            d_vec = pos[i] - pos[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                d = 1e-10
                d_vec = np.array([1e-10, 0])
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                force = 2 * weight * overlap * (d_vec / d)
                grad[i*2:i*2+2] += force
                grad[j*2:j*2+2] -= force
    
    # Wall violation gradients
    for i in range(N):
        # Left wall
        pen = radii[i] - pos[i][0]
        if pen > 0:
            grad[i*2] += 2 * weight * pen
        # Right wall
        pen = pos[i][0] + radii[i] - 1
        if pen > 0:
            grad[i*2] -= 2 * weight * pen
        # Bottom wall
        pen = radii[i] - pos[i][1]
        if pen > 0:
            grad[i*2+1] += 2 * weight * pen
        # Top wall
        pen = pos[i][1] + radii[i] - 1
        if pen > 0:
            grad[i*2+1] -= 2 * weight * pen
    
    return grad

def _optimize_positions(pos, radii, deadline, max_iter=500):
    """Optimize positions to maximize sum of radii (fixed) subject to constraints."""
    # Actually, we want to grow radii too. Let's use a different approach.
    # We'll use L-BFGS-B to minimize penalty while gradually increasing radii.
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Start with current radii
    current_radii = radii.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Try to grow radii slightly
        growth = 1.001
        trial_radii = current_radii * growth
        
        # Check if valid
        if _validity_check(pos, trial_radii):
            current_radii = trial_radii
            if np.sum(current_radii) > best_score:
                best_score = np.sum(current_radii)
                best_pos = pos.copy()
                best_radii = current_radii.copy()
            continue
        
        # If not valid, optimize positions to make it valid
        x0 = pos.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        # Use moderate penalty weight
        weight = 500.0
        
        try:
            res = minimize(
                _penalty_objective,
                x0,
                args=(current_radii, weight),
                jac=lambda p: _penalty_grad(p, current_radii, weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_pos = res.x.reshape(N, 2)
            
            # Check if valid
            if _validity_check(new_pos, current_radii):
                pos = new_pos
                # Try to grow again
                if np.sum(current_radii) > best_score:
                    best_score = np.sum(current_radii)
                    best_pos = pos.copy()
                    best_radii = current_radii.copy()
            elif _validity_check(pos, current_radii):
                # Keep old position if still valid
                pass
            else:
                # Fallback: enforce validity
                pos, current_radii = _enforce_validity(pos, current_radii)
                if np.sum(current_radii) > best_score:
                    best_score = np.sum(current_radii)
                    best_pos = pos.copy()
                    best_radii = current_radii.copy()
                    
        except Exception:
            # Fallback
            pos, current_radii = _enforce_validity(pos, current_radii)
            if np.sum(current_radii) > best_score:
                best_score = np.sum(current_radii)
                best_pos = pos.copy()
                best_radii = current_radii.copy()
    
    return best_pos, best_radii, best_score

def _simulated_annealing(pos, radii, deadline, max_iter=1000):
    """Simulated annealing to escape local minima."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_pos = pos.copy()
    current_radii = radii.copy()
    
    temp = 0.01
    cooling = 0.995
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation
        perturbation = np.random.normal(0, temp, size=current_pos.shape)
        new_pos = current_pos + perturbation
        new_pos = np.clip(new_pos, 0.01, 0.99)
        
        # Try to grow radii
        growth = 1.002
        trial_radii = current_radii * growth
        
        new_score = _score_valid(new_pos, trial_radii)
        current_score = _score_valid(current_pos, current_radii)
        
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(temp, 1e-8)):
            current_pos = new_pos
            current_radii = trial_radii
            if new_score > best_score:
                best_score = new_score
                best_pos = current_pos.copy()
                best_radii = current_radii.copy()
        else:
            # Keep current, but try to enforce validity
            current_pos, current_radii = _enforce_validity(current_pos, current_radii)
        
        temp *= cooling
        if temp < 0.001:
            break
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Multiple initialization strategies and seeds
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 1111, 2222, 3333, 4444, 5555]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        
        rng = np.random.RandomState(seed)
        
        # Try different initializations
        inits = [
            _hex_initial(rng),
            _grid_initial(rng, 6, 5),
            _grid_initial(rng, 5, 6),
            _grid_initial(rng, 7, 4),
            _voronoi_initial(rng),
        ]
        
        for pos in inits:
            if time.time() > deadline - 10:
                break
            
            # Compute initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist = min(min_dist, d)
                min_dist = min(min_dist, pos[i][0], 1-pos[i][0], pos[i][1], 1-pos[i][1])
            
            radii = np.full(N, min_dist * 0.85)
            pos, radii = _enforce_validity(pos, radii)
            
            # Optimize positions and grow radii
            pos, radii, score1 = _optimize_positions(pos, radii, deadline, max_iter=300)
            
            # Simulated annealing for additional improvement
            pos, radii, score2 = _simulated_annealing(pos, radii, deadline, max_iter=200)
            
            if score2 > best_score:
                best_score = score2
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final validity check
    if best_pos is None or not _validity_check(best_pos, best_radii):
        # Fallback: tiny circles
        rng = np.random.RandomState(42)
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.01)
        best_pos, best_radii = _enforce_validity(pos, radii)
    
    result = np.column_stack((best_pos, best_radii))
    return result
