"""Circle packing in unit square – 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _clamp_positions(pos, radii, eps=1e-10):
    """Clamp positions so circles stay inside [0,1]^2."""
    x = np.clip(pos[0], radii, 1.0 - radii)
    y = np.clip(pos[1], radii, 1.0 - radii)
    return x, y

def _is_valid(circles, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < eps:
            return False
        if x - r < -eps or x + r > 1.0 + eps:
            return False
        if y - r < -eps or y + r > 1.0 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True

def _score(circles):
    """Sum of radii; return -inf if invalid."""
    if not _is_valid(circles):
        return -np.inf
    return sum(c[2] for c in circles)

def _initial_grid():
    """Create an initial grid arrangement of N circles."""
    # Use a 5x6 grid (30 cells), take first 26
    cols, rows = 5, 6
    side = 1.0 / cols
    cells = []
    for i in range(cols):
        for j in range(rows):
            cx = (i + 0.5) * side
            cy = (j + 0.5) * side
            r = 0.5 * side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])

def _objective_flat(flat, circles_init, eps=1e-10):
    """
    Objective for optimization: maximize sum of radii subject to constraints.
    We minimize negative sum of radii.
    Variables: x_i, y_i, r_i for i=0..N-1 (3N variables)
    """
    n = N
    x = flat[0:n]
    y = flat[n:2*n]
    r = flat[2*n:3*n]
    
    # Clamp radii to be positive
    r = np.maximum(r, 1e-8)
    
    # Check bounds
    for i in range(n):
        if x[i] - r[i] < -eps or x[i] + r[i] > 1.0 + eps:
            return 1e10  # Large penalty
        if y[i] - r[i] < -eps or y[i] + r[i] > 1.0 + eps:
            return 1e10
    
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - eps:
                return 1e10
    
    return -np.sum(r)

def _gradient_ascent(circles, max_iter=5000, lr=0.01, deadline=None):
    """
    Gradient ascent on sum of radii with projection onto feasible set.
    We use a simple approach: try to increase radii and adjust positions.
    """
    n = N
    best = circles.copy()
    best_score = _score(best)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient: for each circle, we want to increase radius
        # The constraint is that circles don't overlap and stay in bounds.
        # We use a simple iterative scheme:
        # 1. Try to increase all radii slightly
        # 2. Resolve conflicts by moving circles apart
        
        # Increase radii
        dr = 0.001
        new_r = best[:, 2] + dr
        new_x = best[:, 0].copy()
        new_y = best[:, 1].copy()
        
        # Resolve overlaps
        for _ in range(50):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = new_x[j] - new_x[i]
                    dy = new_y[j] - new_y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = new_r[i] + new_r[j]
                    if dist < min_dist and dist > 1e-12:
                        # Push apart
                        overlap = min_dist - dist
                        ux = dx / dist
                        uy = dy / dist
                        push = overlap / 2.0
                        new_x[i] -= ux * push
                        new_y[i] -= uy * push
                        new_x[j] += ux * push
                        new_y[j] += uy * push
                        moved = True
                    elif dist <= 1e-12:
                        # Random small push
                        angle = np.random.uniform(0, 2*np.pi)
                        push = min_dist / 2.0
                        new_x[i] -= np.cos(angle) * push
                        new_y[i] -= np.sin(angle) * push
                        new_x[j] += np.cos(angle) * push
                        new_y[j] += np.sin(angle) * push
                        moved = True
            
            # Clamp to bounds
            for i in range(n):
                new_x[i] = np.clip(new_x[i], new_r[i], 1.0 - new_r[i])
                new_y[i] = np.clip(new_y[i], new_r[i], 1.0 - new_r[i])
            
            if not moved:
                break
        
        new_circles = np.column_stack([new_x, new_y, new_r])
        new_score = _score(new_circles)
        
        if new_score > best_score:
            best = new_circles
            best_score = new_score
        else:
            # Try smaller step
            dr = 0.0001
            new_r = best[:, 2] + dr
            new_x = best[:, 0].copy()
            new_y = best[:, 1].copy()
            for _ in range(30):
                moved = False
                for i in range(n):
                    for j in range(i+1, n):
                        dx = new_x[j] - new_x[i]
                        dy = new_y[j] - new_y[i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = new_r[i] + new_r[j]
                        if dist < min_dist and dist > 1e-12:
                            overlap = min_dist - dist
                            ux = dx / dist
                            uy = dy / dist
                            push = overlap / 2.0
                            new_x[i] -= ux * push
                            new_y[i] -= uy * push
                            new_x[j] += ux * push
                            new_y[j] += uy * push
                            moved = True
                        elif dist <= 1e-12:
                            angle = np.random.uniform(0, 2*np.pi)
                            push = min_dist / 2.0
                            new_x[i] -= np.cos(angle) * push
                            new_y[i] -= np.sin(angle) * push
                            new_x[j] += np.cos(angle) * push
                            new_y[j] += np.sin(angle) * push
                            moved = True
                for i in range(n):
                    new_x[i] = np.clip(new_x[i], new_r[i], 1.0 - new_r[i])
                    new_y[i] = np.clip(new_y[i], new_r[i], 1.0 - new_r[i])
                if not moved:
                    break
            new_circles = np.column_stack([new_x, new_y, new_r])
            new_score = _score(new_circles)
            if new_score > best_score:
                best = new_circles
                best_score = new_score
            else:
                break
    
    return best, best_score

def _local_search(circles, deadline=None):
    """
    Local search: try random perturbations and accept improvements.
    """
    n = N
    best = circles.copy()
    best_score = _score(best)
    
    for it in range(2000):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        new_circles = best.copy()
        idx = np.random.randint(0, n)
        delta = 0.01 * np.random.randn(3)
        new_circles[idx, 0] += delta[0]
        new_circles[idx, 1] += delta[1]
        new_circles[idx, 2] += delta[2]
        
        # Clamp
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], new_circles[idx, 2], 1.0 - new_circles[idx, 2])
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], new_circles[idx, 2], 1.0 - new_circles[idx, 2])
        new_circles[idx, 2] = max(new_circles[idx, 2], 1e-6)
        
        new_score = _score(new_circles)
        if new_score > best_score:
            best = new_circles
            best_score = new_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with grid
    circles = _initial_grid()
    best_circles = circles.copy()
    best_score = _score(best_circles)
    
    # Gradient ascent
    circles, score = _gradient_ascent(circles, max_iter=3000, deadline=deadline)
    if score > best_score:
        best_circles = circles
        best_score = score
    
    if time.time() > deadline:
        return best_circles
    
    # Local search
    circles, score = _local_search(circles, deadline=deadline)
    if score > best_score:
        best_circles = circles
        best_score = score
    
    if time.time() > deadline:
        return best_circles
    
    # More gradient ascent
    circles, score = _gradient_ascent(circles, max_iter=3000, deadline=deadline)
    if score > best_score:
        best_circles = circles
        best_score = score
    
    # Final local search
    circles, score = _local_search(circles, deadline=deadline)
    if score > best_score:
        best_circles = circles
        best_score = score
    
    # Ensure validity
    if not _is_valid(best_circles):
        # Fallback to grid
        best_circles = _initial_grid()
    
    return best_circles
# EVOLVE-BLOCK-END
