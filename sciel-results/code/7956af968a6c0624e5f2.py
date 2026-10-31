"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution
from scipy.spatial import cKDTree
import itertools

N = 26
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _is_valid(circles):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    if n == 0:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - 1e-9:
                return False
    return True

def _sum_radii(circles):
    return sum(c[2] for c in circles)

def _construct_grid():
    """Simple grid packing as initial solution."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append((x, y, r))
    return np.array(cells[:N])

def _construct_hexagonal():
    """Hexagonal packing pattern."""
    # Try a hexagonal arrangement
    circles = []
    # Use a grid with hexagonal offset
    rows = 5
    cols = 6
    # Calculate radius for hexagonal packing in unit square
    # For hexagonal packing, the spacing is r*2 in one direction and r*sqrt(3) in the other
    # Let's try different configurations
    
    best = None
    best_sum = 0
    
    for r in np.linspace(0.03, 0.12, 50):
        # Hexagonal spacing
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        # Try to fit as many circles as possible
        positions = []
        for row in range(10):
            for col in range(10):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                        # Check if this position doesn't overlap with existing circles
                        valid = True
                        for px, py, pr in positions:
                            if np.sqrt((x - px)**2 + (y - py)**2) < r + pr - 1e-9:
                                valid = False
                                break
                        if valid:
                            positions.append((x, y, r))
        
        if len(positions) >= N:
            positions = positions[:N]
            s = sum(p[2] for p in positions)
            if s > best_sum:
                best_sum = s
                best = np.array(positions)
    
    if best is not None:
        return best
    return None

def _gradient_ascent(circles, steps=500, lr=0.01):
    """Simple gradient ascent to improve packing."""
    circles = circles.copy()
    n = len(circles)
    
    for step in range(steps):
        if _time_left() < 10:
            break
        
        # Compute forces
        forces = np.zeros_like(circles)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 1e-10:
                    dx, dy = 0.01, 0.01
                    dist = np.sqrt(dx**2 + dy**2)
                overlap = r1 + r2 - dist
                if overlap > 0:
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        # Attraction to maximize radii (push circles apart to allow larger radii)
        for i in range(n):
            x, y, r = circles[i]
            # Push away from boundaries
            if x - r < 0:
                forces[i, 0] += (0 - (x - r)) * 0.5
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 0.5
            if y - r < 0:
                forces[i, 1] += (0 - (y - r)) * 0.5
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 0.5
        
        # Update positions
        for i in range(n):
            circles[i, 0] += lr * forces[i, 0]
            circles[i, 1] += lr * forces[i, 1]
            # Keep in bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Try to increase radii slightly
        for i in range(n):
            x, y, r = circles[i]
            # Check if we can increase radius
            min_dist_to_boundary = min(x, 1 - x, y, 1 - y)
            min_dist_to_other = np.inf
            for j in range(n):
                if i == j:
                    continue
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_dist_to_other = min(min_dist_to_other, dist - r2)
            max_r = min(min_dist_to_boundary, min_dist_to_other)
            if max_r > r:
                circles[i, 2] = min(max_r, r + 0.001)
    
    return circles

def _lp_optimize_radii(circles):
    """Use LP to optimize radii given positions."""
    n = len(circles)
    positions = circles[:, :2]
    
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r_i)
    # Subject to:
    # r_i >= 0 for all i
    # r_i + r_j <= dist(i,j) for all i < j
    # r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
    
    c = -np.ones(n)  # Minimize -sum(r_i)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    
    # Boundary constraints
    for i in range(n):
        max_r = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(max_r)
    
    if not A_ub:
        return circles
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            new_radii = res.x
            # Check if valid
            new_circles = circles.copy()
            new_circles[:, 2] = new_radii
            if _is_valid(new_circles):
                return new_circles
    except:
        pass
    
    return circles

def _local_search(circles, num_restarts=5):
    """Local search with random perturbations."""
    best = circles.copy()
    best_sum = _sum_radii(best)
    
    for restart in range(num_restarts):
        if _time_left() < 5:
            break
        
        current = best.copy()
        # Random perturbation
        for i in range(len(current)):
            current[i, 0] += np.random.uniform(-0.01, 0.01)
            current[i, 1] += np.random.uniform(-0.01, 0.01)
            current[i, 0] = np.clip(current[i, 0], 0, 1)
            current[i, 1] = np.clip(current[i, 1], 0, 1)
        
        # Gradient ascent
        current = _gradient_ascent(current, steps=200, lr=0.005)
        
        # LP optimization
        current = _lp_optimize_radii(current)
        
        if _is_valid(current):
            s = _sum_radii(current)
            if s > best_sum:
                best_sum = s
                best = current.copy()
    
    return best

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    
    # Initial solutions
    candidates = []
    
    # Grid
    grid = _construct_grid()
    candidates.append(grid)
    
    # Hexagonal
    hex_pack = _construct_hexagonal()
    if hex_pack is not None:
        candidates.append(hex_pack)
    
    # Random initializations
    for _ in range(3):
        if _time_left() < 5:
            break
        rand_circles = np.zeros((N, 3))
        rand_circles[:, 0] = np.random.uniform(0.05, 0.95, N)
        rand_circles[:, 1] = np.random.uniform(0.05, 0.95, N)
        rand_circles[:, 2] = np.random.uniform(0.02, 0.06, N)
        candidates.append(rand_circles)
    
    best = None
    best_sum = 0
    
    for circ in candidates:
        if _time_left() < 10:
            break
        
        # Validate and repair
        circ = circ.copy()
        # Clip to valid positions
        for i in range(len(circ)):
            r = circ[i, 2]
            circ[i, 0] = np.clip(circ[i, 0], r, 1 - r)
            circ[i, 1] = np.clip(circ[i, 1], r, 1 - r)
        
        # Gradient ascent
        circ = _gradient_ascent(circ, steps=300, lr=0.008)
        
        # LP optimization
        circ = _lp_optimize_radii(circ)
        
        # Local search
        circ = _local_search(circ, num_restarts=3)
        
        if _is_valid(circ):
            s = _sum_radii(circ)
            if s > best_sum:
                best_sum = s
                best = circ
    
    if best is None:
        # Fallback to grid
        return _construct_grid()
    
    return best
