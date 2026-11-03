"""Circle packing in unit square - best-fit discrete search with diverse initializations."""
import numpy as np
import time
import math

N = 26

def _is_valid(circles, tol=1e-9):
    """Check if all circles are inside unit square and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _max_radius_for_circle(x, y, circles, idx, tol=1e-10):
    """Find max radius for circle at (x,y) given all other circles."""
    r_max = min(x, 1 - x, y, 1 - y)
    for j in range(len(circles)):
        if j == idx:
            continue
        dx = x - circles[j][0]
        dy = y - circles[j][1]
        dist = math.sqrt(dx * dx + dy * dy)
        r_max = min(r_max, dist - circles[j][2])
    return max(0.0, r_max)

def _grid_init(n, seed=0):
    side = int(math.ceil(math.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])

def _hex_init(n, seed=0):
    """Hexagonal packing initialization."""
    circles = []
    r = 0.08
    dx = 2 * r
    dy = r * math.sqrt(3)
    row = 0
    while len(circles) < n:
        offset = (dx / 2) if row % 2 else 0
        col = 0
        while len(circles) < n:
            x = offset + col * dx + r
            y = row * dy + r
            if x + r <= 1.0 and y + r <= 1.0:
                circles.append([x, y, r])
            col += 1
        row += 1
    return np.array(circles[:n])

def _random_init(n, seed=0):
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(n):
        r = 0.05 + rng.random() * 0.05
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)

def _corner_init(n, seed=0):
    """Initialize circles biased toward corners and edges."""
    rng = np.random.RandomState(seed)
    circles = []
    # Place some in corners
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    for cx, cy in corners:
        if len(circles) < n:
            circles.append([cx, cy, 0.07])
    # Fill rest randomly but biased
    for _ in range(n - len(circles)):
        r = 0.06 + rng.random() * 0.04
        # Bias toward edges
        edge = rng.randint(4)
        if edge == 0:
            x = rng.uniform(r, 1 - r)
            y = rng.uniform(r, 0.3)
        elif edge == 1:
            x = rng.uniform(r, 1 - r)
            y = rng.uniform(0.7, 1 - r)
        elif edge == 2:
            x = rng.uniform(r, 0.3)
            y = rng.uniform(r, 1 - r)
        else:
            x = rng.uniform(0.7, 1 - r)
            y = rng.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles[:n])

def _optimize_positions_and_radii(circles, max_time, deadline):
    """Iteratively optimize positions and radii using best-fit discrete search."""
    circles = np.array(circles, dtype=float)
    n = len(circles)
    best_score = _total_radius(circles)
    best_circles = circles.copy()
    
    # Multiple step sizes for discrete search
    step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    # Directions: 8 neighbors + 4 diagonals + more
    directions = []
    for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        directions.append((math.cos(angle), math.sin(angle)))
    
    while time.time() < deadline:
        improved = False
        for step in step_sizes:
            for i in range(n):
                if time.time() >= deadline:
                    break
                x, y, r = circles[i]
                # Try all directions and step sizes
                best_move = None
                best_move_score = _total_radius(circles)
                
                for dx_dir, dy_dir in directions:
                    for s in [step, step * 0.5, step * 2]:
                        new_x = x + dx_dir * s
                        new_y = y + dy_dir * s
                        # Check bounds
                        if new_x - r < -1e-9 or new_x + r > 1 + 1e-9:
                            continue
                        if new_y - r < -1e-9 or new_y + r > 1 + 1e-9:
                            continue
                        # Check overlap with others
                        valid = True
                        for j in range(n):
                            if j == i:
                                continue
                            dx = new_x - circles[j][0]
                            dy = new_y - circles[j][1]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < r + circles[j][2] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Compute new total radius (position change doesn't change radius directly,
                        # but we can try to grow radius after moving)
                        # For now, just track position feasibility
                        # Score improvement comes from ability to grow radii later
                        # Use a heuristic: prefer moves that create more space
                        new_score = _total_radius(circles)
                        if new_score > best_move_score + 1e-12:
                            best_move_score = new_score
                            best_move = (new_x, new_y)
                
                # Also try to grow radius in place
                new_r = _max_radius_for_circle(x, y, circles, i)
                if new_r > r + 1e-10:
                    circles[i][2] = new_r
                    improved = True
                    new_score = _total_radius(circles)
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = circles.copy()
                
                if best_move is not None:
                    circles[i][0] = best_move[0]
                    circles[i][1] = best_move[1]
                    improved = True
                    # Try to grow radius after moving
                    new_r = _max_radius_for_circle(best_move[0], best_move[1], circles, i)
                    if new_r > circles[i][2] + 1e-10:
                        circles[i][2] = new_r
                    new_score = _total_radius(circles)
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = circles.copy()
            
            if time.time() >= deadline:
                break
        
        if not improved:
            break
    
    return best_circles, best_score

def _local_search_refine(circles, deadline):
    """Refine using local search with multiple step sizes."""
    circles = np.array(circles, dtype=float)
    n = len(circles)
    best_score = _total_radius(circles)
    best_circles = circles.copy()
    
    # Adaptive step sizes
    base_steps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    iteration = 0
    while time.time() < deadline:
        improved = False
        for step in base_steps:
            for i in range(n):
                if time.time() >= deadline:
                    break
                x, y, r = circles[i]
                best_move = None
                best_move_r = r
                best_move_score = _total_radius(circles)
                
                # Try 16 directions
                for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    dx_dir = math.cos(angle)
                    dy_dir = math.sin(angle)
                    for s in [step, step * 0.5, step * 1.5]:
                        new_x = x + dx_dir * s
                        new_y = y + dy_dir * s
                        if new_x - r < -1e-9 or new_x + r > 1 + 1e-9:
                            continue
                        if new_y - r < -1e-9 or new_y + r > 1 + 1e-9:
                            continue
                        valid = True
                        for j in range(n):
                            if j == i:
                                continue
                            dx = new_x - circles[j][0]
                            dy = new_y - circles[j][1]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < r + circles[j][2] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Try to grow radius at new position
                        new_r = _max_radius_for_circle(new_x, new_y, circles, i)
                        new_score = _total_radius(circles) - r + new_r
                        if new_score > best_move_score + 1e-12:
                            best_move_score = new_score
                            best_move = (new_x, new_y, new_r)
                
                if best_move is not None:
                    circles[i][0] = best_move[0]
                    circles[i][1] = best_move[1]
                    circles[i][2] = best_move[2]
                    improved = True
                    new_score = _total_radius(circles)
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = circles.copy()
            
            if time.time() >= deadline:
                break
        
        # Also try growing all radii
        for i in range(n):
            if time.time() >= deadline:
                break
            x, y, r = circles[i]
            new_r = _max_radius_for_circle(x, y, circles, i)
            if new_r > r + 1e-10:
                circles[i][2] = new_r
                improved = True
                new_score = _total_radius(circles)
                if new_score > best_score:
                    best_score = new_score
                    best_circles = circles.copy()
        
        if not improved:
            break
        iteration += 1
    
    return best_circles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -1
    best_circles = None
    
    # Diverse initializations
    inits = [
        ("grid", _grid_init(N, 0)),
        ("hex", _hex_init(N, 0)),
        ("random1", _random_init(N, 42)),
        ("random2", _random_init(N, 123)),
        ("random3", _random_init(N, 999)),
        ("corner", _corner_init(N, 0)),
        ("corner2", _corner_init(N, 7)),
    ]
    
    for name, init in inits:
        if time.time() >= deadline:
            break
        
        # First pass: optimize positions and radii
        circles, score = _optimize_positions_and_radii(init, 20, deadline)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        if time.time() >= deadline:
            break
    
    # Second pass: refine the best solution
    if best_circles is not None and time.time() < deadline:
        circles, score = _local_search_refine(best_circles, deadline)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # Final radius growth
    if best_circles is not None and time.time() < deadline:
        for i in range(N):
            if time.time() >= deadline:
                break
            x, y, r = best_circles[i]
            new_r = _max_radius_for_circle(x, y, best_circles, i)
            if new_r > r + 1e-10:
                best_circles[i][2] = new_r
    
    if best_circles is None:
        # Fallback to grid
        best_circles = _grid_init(N, 0)
    
    return best_circles
