"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, eps=1e-9):
    """Check if all circles are inside [0,1]^2 and have disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < -eps or x > 1 + eps or y < -eps or y > 1 + eps:
            return False
        if r < -eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - eps:
                return False
    return True

def _score(circles):
    """Sum of radii."""
    return float(np.sum(circles[:, 2]))

def _initial_grid(n):
    """Create a grid of n circles in the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:n])

def _max_radius_for_circle(circles, idx, eps=1e-10):
    """Find the maximum radius for circle idx given other circles and boundaries."""
    x, y, _ = circles[idx]
    # Boundary constraints
    max_r_boundary = min(x, 1 - x, y, 1 - y)
    
    max_r = max_r_boundary
    for j in range(len(circles)):
        if j == idx:
            continue
        xj, yj, rj = circles[j]
        dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
        # Need dist >= r + rj, so r <= dist - rj
        candidate = dist - rj
        if candidate < max_r:
            max_r = candidate
    return max(0, max_r)

def _physics_relaxation(circles, max_iters=5000, dt=0.001, grow_rate=0.01):
    """Physics-based relaxation: resolve overlaps and grow radii."""
    circles = circles.copy()
    n = len(circles)
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Repulsive forces for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    # Random nudge
                    dx = np.random.randn()
                    dy = np.random.randn()
                    dist = np.sqrt(dx * dx + dy * dy)
                
                overlap = ri + rj - dist
                if overlap > 0:
                    # Force proportional to overlap
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i] -= fx
                    forces[i] -= fy
                    forces[j] += fx
                    forces[j] += fy
        
        # Boundary forces
        for i in range(n):
            x, y, r = circles[i]
            if x < r:
                forces[i, 0] += (r - x) * 10
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10
            if y < r:
                forces[i, 1] += (r - y) * 10
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10
        
        # Update positions
        for i in range(n):
            circles[i, 0] += forces[i, 0] * dt
            circles[i, 1] += forces[i, 1] * dt
            # Clamp to unit square
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
        
        # Grow radii
        for i in range(n):
            max_r = _max_radius_for_circle(circles, i)
            if max_r > circles[i, 2]:
                circles[i, 2] = min(max_r, circles[i, 2] + grow_rate)
            else:
                circles[i, 2] = max_r
    
    return circles

def _local_search(circles, max_iters=10000, step_size=0.01):
    """Local search: try small perturbations to improve the score."""
    circles = circles.copy()
    n = len(circles)
    best_score = _score(circles)
    
    for it in range(max_iters):
        # Pick a random circle and perturb it
        i = np.random.randint(n)
        old = circles[i].copy()
        
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0, step_size)
        circles[i, 0] += dist * np.cos(angle)
        circles[i, 1] += dist * np.sin(angle)
        
        # Try to grow radius
        max_r = _max_radius_for_circle(circles, i)
        circles[i, 2] = max_r
        
        if _is_valid(circles):
            new_score = _score(circles)
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert
                circles[i] = old
        else:
            circles[i] = old
    
    return circles

def _random_restart(n, seed=None):
    """Generate a random initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    
    circles = []
    for _ in range(n):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.02, 0.1)
        circles.append([x, y, r])
    return np.array(circles)

def _optimize_single(circles, time_limit=100):
    """Optimize a single configuration within time limit."""
    start_time = time.time()
    
    # Phase 1: Physics relaxation
    circles = _physics_relaxation(circles, max_iters=2000, dt=0.002, grow_rate=0.02)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.5:
        return circles
    
    # Phase 2: Local search
    remaining = time_limit - elapsed
    if remaining > 5:
        circles = _local_search(circles, max_iters=5000, step_size=0.005)
    
    return circles

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s
    
    # Start with grid
    best_circles = _initial_grid(N)
    best_score = _score(best_circles)
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Generate random initial config
        np.random.seed(42 + restart)
        circles = _random_restart(N, seed=42 + restart)
        
        # Optimize
        per_restart_time = max(10, (time_limit - elapsed) / (n_restarts - restart))
        optimized = _optimize_single(circles, time_limit=per_restart_time)
        
        # Check validity and score
        if _is_valid(optimized):
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_circles = optimized
    
    # Final local search on best if time remains
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        best_circles = _local_search(best_circles, max_iters=3000, step_size=0.003)
        if _is_valid(best_circles):
            best_score = _score(best_circles)
    
    return best_circles

# EVOLVE-BLOCK-END
