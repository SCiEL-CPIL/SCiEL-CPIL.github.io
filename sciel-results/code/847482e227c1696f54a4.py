"""Improved circle packing: force-directed + gradient refinement with deadline guard."""
N = 21

import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(circles):
    """Check circles are disjoint and inside a perimeter-4 rectangle."""
    n = len(circles)
    if n != N:
        return False
    # Find bounding box
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    width = max_x - min_x
    height = max_y - min_y
    if width + height > 2.0 + 1e-9:  # perimeter = 2*(w+h) <= 4
        return False
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - 1e-7:
                return False
    return True


def _compute_value(circles):
    """Compute sum of radii if valid, else -inf."""
    if not _is_valid(circles):
        return -np.inf
    return np.sum(circles[:, 2])


def _force_directed_refine(circles, n_iter=3000, dt=0.002, temp_start=0.05):
    """Force-directed refinement: repulsion between circles, attraction to center,
    and a soft constraint to keep perimeter <= 4."""
    circles = circles.copy().astype(np.float64)
    n = len(circles)
    
    for it in range(n_iter):
        # Compute forces
        forces = np.zeros_like(circles)
        
        # Pairwise repulsion (if overlapping, push apart; if not, slight attraction to pack tighter)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < 1e-10:
                    # Random small push
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx = np.cos(angle) * 0.1
                    fy = np.sin(angle) * 0.1
                else:
                    fx = dx / dist
                    fy = dy / dist
                
                if dist < min_dist:
                    # Strong repulsion when overlapping
                    overlap = min_dist - dist
                    f_mag = 5.0 * overlap
                else:
                    # Weak attraction to pack circles closer (helps maximize radii)
                    f_mag = -0.01 * (dist - min_dist) if dist < min_dist * 1.5 else 0.0
                
                forces[i, 0] += fx * f_mag
                forces[i, 1] += fy * f_mag
                forces[j, 0] -= fx * f_mag
                forces[j, 1] -= fy * f_mag
        
        # Centering force (keep circles near origin)
        for i in range(n):
            forces[i, 0] -= 0.1 * circles[i, 0]
            forces[i, 1] -= 0.1 * circles[i, 1]
        
        # Perimeter constraint: if bounding box perimeter > 4, push inward
        min_x = np.min(circles[:, 0] - circles[:, 2])
        max_x = np.max(circles[:, 0] + circles[:, 2])
        min_y = np.min(circles[:, 1] - circles[:, 2])
        max_y = np.max(circles[:, 1] + circles[:, 2])
        w = max_x - min_x
        h = max_y - min_y
        perim = 2 * (w + h)
        
        if perim > 4.0:
            # Scale positions toward center
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            for i in range(n):
                forces[i, 0] += 2.0 * (cx - circles[i, 0])
                forces[i, 1] += 2.0 * (cy - circles[i, 1])
        
        # Anisotropic scaling: if w > h, compress x; if h > w, compress y
        if w > h * 1.1:
            for i in range(n):
                forces[i, 0] += 0.5 * (w - h) * (0 - (circles[i, 0] - (min_x + max_x)/2))
        elif h > w * 1.1:
            for i in range(n):
                forces[i, 1] += 0.5 * (h - w) * (0 - (circles[i, 1] - (min_y + max_y)/2))
        
        # Update positions with temperature cooling
        temp = temp_start * (1.0 - it / n_iter) + 0.001
        for i in range(n):
            circles[i, 0] += forces[i, 0] * dt + np.random.normal(0, temp * 0.01)
            circles[i, 1] += forces[i, 1] * dt + np.random.normal(0, temp * 0.01)
    
    return circles


def _gradient_refine(circles, n_iter=2000, lr_start=0.005):
    """Gradient-based refinement to maximize sum of radii subject to constraints.
    Uses penalty method for non-overlap and perimeter."""
    circles = circles.copy().astype(np.float64)
    n = len(circles)
    
    for it in range(n_iter):
        lr = lr_start * (1.0 - it / n_iter) + 0.0001
        grad = np.zeros_like(circles)
        
        # Objective: maximize sum of radii
        # grad of sum(r_i) w.r.t. r_i is 1
        grad[:, 2] = 1.0
        
        # Penalty for overlap: for each pair, penalty = max(0, r_i + r_j - dist_ij)^2
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < 1e-10:
                    continue
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Gradient of penalty w.r.t. positions
                    fx = dx / dist
                    fy = dy / dist
                    pen_grad = 2.0 * overlap * 10.0  # penalty weight
                    grad[i, 0] -= fx * pen_grad
                    grad[i, 1] -= fy * pen_grad
                    grad[j, 0] += fx * pen_grad
                    grad[j, 1] += fy * pen_grad
                    # Gradient w.r.t. radii
                    grad[i, 2] -= pen_grad
                    grad[j, 2] -= pen_grad
        
        # Perimeter penalty
        min_x = np.min(circles[:, 0] - circles[:, 2])
        max_x = np.max(circles[:, 0] + circles[:, 2])
        min_y = np.min(circles[:, 1] - circles[:, 2])
        max_y = np.max(circles[:, 1] + circles[:, 2])
        w = max_x - min_x
        h = max_y - min_y
        perim = 2 * (w + h)
        
        if perim > 4.0:
            excess = perim - 4.0
            # Find which circles contribute to bounds
            for i in range(n):
                left = circles[i, 0] - circles[i, 2]
                right = circles[i, 0] + circles[i, 2]
                bottom = circles[i, 1] - circles[i, 2]
                top = circles[i, 1] + circles[i, 2]
                
                if left <= min_x + 1e-9:
                    grad[i, 0] += 5.0 * excess
                    grad[i, 2] -= 5.0 * excess
                if right >= max_x - 1e-9:
                    grad[i, 0] -= 5.0 * excess
                    grad[i, 2] -= 5.0 * excess
                if bottom <= min_y + 1e-9:
                    grad[i, 1] += 5.0 * excess
                    grad[i, 2] -= 5.0 * excess
                if top >= max_y - 1e-9:
                    grad[i, 1] -= 5.0 * excess
                    grad[i, 2] -= 5.0 * excess
        
        # Update
        for i in range(n):
            circles[i, 0] += lr * grad[i, 0]
            circles[i, 1] += lr * grad[i, 1]
            circles[i, 2] += lr * grad[i, 2]
            # Ensure radii positive
            circles[i, 2] = max(circles[i, 2], 0.001)
    
    return circles


def _hex_init():
    """Initialize circles in a hexagonal pattern."""
    # Use a grid with hexagonal offset
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            x = (i + 0.5 + (0.5 if j % 2 == 1 else 0)) / (side + 0.5)
            y = (j + 0.5) / (side + 0.5)
            r = 0.45 / (side + 0.5)
            circles.append([x, y, r])
    return np.array(circles[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try multiple initializations and refinement strategies
    best_circles = None
    best_value = -np.inf
    
    # Initial configurations to try
    inits = []
    
    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.48 / side) 
                  for i in range(side) for j in range(side)]
    inits.append(np.array(grid_cells[:N]))
    
    # 2. Hexagonal initialization
    inits.append(_hex_init())
    
    # 3. Random initialization in unit square
    for seed in range(5):
        rng = np.random.RandomState(seed)
        xs = rng.uniform(0.2, 0.8, N)
        ys = rng.uniform(0.2, 0.8, N)
        rs = np.full(N, 0.1)
        inits.append(np.column_stack([xs, ys, rs]))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force-directed refinement
        refined = _force_directed_refine(init, n_iter=2000, dt=0.002, temp_start=0.03)
        
        # Scale to fit perimeter 4
        min_x = np.min(refined[:, 0] - refined[:, 2])
        max_x = np.max(refined[:, 0] + refined[:, 2])
        min_y = np.min(refined[:, 1] - refined[:, 2])
        max_y = np.max(refined[:, 1] + refined[:, 2])
        w = max_x - min_x
        h = max_y - min_y
        if w + h > 2.0:
            scale = 2.0 / (w + h)
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            for i in range(N):
                refined[i, 0] = cx + (refined[i, 0] - cx) * scale
                refined[i, 1] = cy + (refined[i, 1] - cy) * scale
                refined[i, 2] *= scale
        
        val = _compute_value(refined)
        if val > best_value:
            best_value = val
            best_circles = refined.copy()
        
        if time.time() > deadline:
            break
        
        # Gradient refinement
        grad_refined = _gradient_refine(refined, n_iter=1500, lr_start=0.003)
        val2 = _compute_value(grad_refined)
        if val2 > best_value:
            best_value = val2
            best_circles = grad_refined.copy()
    
    # Final polish: try to increase radii slightly
    if best_circles is not None and time.time() < deadline:
        for _ in range(500):
            if time.time() > deadline:
                break
            # Try increasing all radii by small amount
            test = best_circles.copy()
            test[:, 2] *= 1.001
            if _is_valid(test):
                best_circles = test
            else:
                break
    
    # If no valid result, return the seed
    if best_circles is None or not _is_valid(best_circles):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
