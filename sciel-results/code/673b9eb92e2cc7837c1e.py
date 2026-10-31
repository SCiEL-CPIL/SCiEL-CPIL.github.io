"""Circle packing in perimeter-4 rectangle using unified penalty-based optimization."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def _validate(circles, tol=1e-9):
    """Check that circles are valid: non-negative radii, disjoint interiors, bounding rect perimeter <= 4."""
    if len(circles) != N:
        return False
    for c in circles:
        if c[2] < -tol:
            return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    # Check bounding rectangle perimeter
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2 * (w + h)
    if perim > 4 + tol:
        return False
    return True


def _score(circles):
    """Compute the sum of radii if valid, else 0."""
    if not _validate(circles):
        return 0.0
    return np.sum(circles[:, 2])


def _initial_grid():
    """Create initial grid placement."""
    side = 5  # 5x5 grid = 25 cells, use 21
    # Rectangle with perimeter 4: w + h = 2
    # Start with square-ish: w = h = 1
    w = 1.0
    h = 1.0
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            x = (i + 0.5) / side * w
            y = (j + 0.5) / side * h
            r = min(0.5 / side * w, 0.5 / side * h) * 0.95
            circles.append([x, y, r])
    return np.array(circles[:N])


def _force_directed(circles, w, h, steps=5000, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to maximize sum of radii."""
    circles = circles.copy()
    lr = lr_init
    best_score = _score(circles)
    best_circles = circles.copy()
    
    for step in range(steps):
        # Compute forces
        forces = np.zeros_like(circles)
        
        # Pairwise repulsion/attraction based on overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = circles[i][2] + circles[j][2] - dist
                if overlap > 0:
                    # Push apart
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # Boundary forces: try to expand radii by pushing toward center
        # and adjust positions to fit
        for i in range(N):
            # Distance to boundaries
            d_left = circles[i][0] - circles[i][2]
            d_right = w - (circles[i][0] + circles[i][2])
            d_bottom = circles[i][1] - circles[i][2]
            d_top = h - (circles[i][1] + circles[i][2])
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d < 0:
                # Circle is outside, push inward
                if d_left == min_d:
                    forces[i][0] += -min_d
                if d_right == min_d:
                    forces[i][0] += -min_d
                if d_bottom == min_d:
                    forces[i][1] += -min_d
                if d_top == min_d:
                    forces[i][1] += -min_d
            
            # Try to grow radius slightly
            forces[i][2] += 0.001
        
        # Apply forces
        circles += lr * forces
        lr *= lr_decay
        
        # Ensure radii are non-negative
        circles[:, 2] = np.maximum(circles[:, 2], 0.0)
        
        # Check score periodically
        if step % 100 == 0:
            s = _score(circles)
            if s > best_score:
                best_score = s
                best_circles = circles.copy()
    
    return best_circles, best_score


def _optimize_aspect(circles, steps=2000):
    """Optimize aspect ratio by adjusting w and h while maintaining perimeter 4."""
    best_score = _score(circles)
    best_circles = circles.copy()
    
    # Try different aspect ratios
    for w in np.linspace(0.5, 1.5, 21):
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Scale positions to fit in new rectangle
        # Find bounding box of current circles
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        cur_w = max_x - min_x
        cur_h = max_y - min_y
        
        if cur_w < 1e-9 or cur_h < 1e-9:
            continue
        
        # Scale to fit
        scale_x = w / cur_w
        scale_y = h / cur_h
        scale = min(scale_x, scale_y)
        
        new_circles = circles.copy()
        new_circles[:, 0] = (circles[:, 0] - min_x) * scale + 0
        new_circles[:, 1] = (circles[:, 1] - min_y) * scale + 0
        new_circles[:, 2] = circles[:, 2] * scale
        
        # Relax
        relaxed, s = _force_directed(new_circles, w, h, steps=steps // 20)
        if s > best_score:
            best_score = s
            best_circles = relaxed.copy()
    
    return best_circles, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial grid
    circles = _initial_grid()
    best_circles = circles.copy()
    best_score = _score(circles)
    
    # Multiple restarts with different initial conditions
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Create perturbed initial condition
        w = 1.0
        h = 1.0
        circles = np.zeros((N, 3))
        
        # Random grid-like placement with some jitter
        for i in range(N):
            row = i // 5
            col = i % 5
            x = (col + 0.5) / 5 * w + rng.normal(0, 0.02)
            y = (row + 0.5) / 5 * h + rng.normal(0, 0.02)
            r = 0.19  # Initial radius
            circles[i] = [x, y, r]
        
        # Relax
        relaxed, s = _force_directed(circles, w, h, steps=3000)
        if s > best_score:
            best_score = s
            best_circles = relaxed.copy()
        
        # Try aspect ratio optimization
        if time.time() < deadline - 10:
            opt_circles, opt_s = _optimize_aspect(relaxed, steps=1000)
            if opt_s > best_score:
                best_score = opt_s
                best_circles = opt_circles.copy()
    
    return best_circles
# EVOLVE-BLOCK-END
