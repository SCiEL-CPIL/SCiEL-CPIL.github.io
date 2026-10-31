"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _objective(params, w, h, n):
    """Objective: minimize negative sum of radii + penalty for constraint violations."""
    # params layout: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    xs = params[0:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]
    
    # Ensure positive radii
    rs = np.maximum(rs, 1e-8)
    
    # Wall constraints: circles must be inside rectangle
    # x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    wall_viol = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            wall_viol += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            wall_viol += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            wall_viol += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            wall_viol += (ys[i] - (h - rs[i])) ** 2
    
    # Pairwise constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    pair_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist:
                pair_viol += (min_dist * min_dist - dist_sq) ** 2
    
    penalty = 100.0
    return -np.sum(rs) + penalty * (wall_viol + pair_viol)


def _construct_for_aspect(aspect_ratio, n, deadline):
    """Try to construct a packing for a given aspect ratio."""
    # w + h = 2, aspect_ratio = w/h
    # w = aspect_ratio * h, so aspect_ratio * h + h = 2 => h = 2/(1+aspect_ratio)
    h = 2.0 / (1.0 + aspect_ratio)
    w = aspect_ratio * h
    
    # Max possible radius is min(w, h) / 2
    max_r = min(w, h) / 2.0
    
    # Try multiple initializations
    best_result = None
    best_obj = np.inf
    
    # Initialization strategies
    inits = []
    
    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(n)))
    for i in range(side):
        for j in range(side):
            if len(inits) >= n:
                break
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.4 * min(w, h) / side  # initial radius estimate
            inits.append([x, y, r])
    
    # 2. Hexagonal-like initialization
    rows = int(np.ceil(np.sqrt(n * h / w))) if w > 0 else n
    count = 0
    for row in range(rows):
        offset = 0.0 if row % 2 == 0 else w / (2 * max(1, int(np.sqrt(n * w / h))))
        cols = int(np.sqrt(n * w / h)) + 1
        for col in range(cols):
            if count >= n:
                break
            x = (col + 0.5 + offset) * w / cols
            y = (row + 0.5) * h / rows
            r = 0.35 * min(w, h) / max(rows, cols)
            inits.append([x, y, r])
            count += 1
    
    # 3. Random initialization
    np.random.seed(42)
    for _ in range(n):
        x = np.random.uniform(0.1 * max_r, w - 0.1 * max_r)
        y = np.random.uniform(0.1 * max_r, h - 0.1 * max_r)
        r = np.random.uniform(0.05 * max_r, 0.3 * max_r)
        inits.append([x, y, r])
    
    # 4. Concentric ring initialization
    for i in range(n):
        angle = 2 * np.pi * i / n
        radius_from_center = max_r * 0.7 * (i / max(1, n - 1))
        x = w / 2 + radius_from_center * np.cos(angle)
        y = h / 2 + radius_from_center * np.sin(angle)
        r = 0.2 * max_r
        inits.append([x, y, r])
    
    # Deduplicate and ensure we have at least n initializations
    unique_inits = []
    seen = set()
    for init in inits:
        key = (round(init[0], 4), round(init[1], 4), round(init[2], 4))
        if key not in seen:
            seen.add(key)
            unique_inits.append(init)
    
    # Pad if needed
    while len(unique_inits) < n:
        unique_inits.append([w / 2, h / 2, 0.1 * max_r])
    
    unique_inits = unique_inits[:n]
    
    # Try optimization from each initialization
    for init in unique_inits:
        if time.time() > deadline:
            break
        
        x0 = []
        for i in range(n):
            x0.append(init[i][0])
        for i in range(n):
            x0.append(init[i][1])
        for i in range(n):
            x0.append(init[i][2])
        x0 = np.array(x0)
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-6, w - 1e-6))  # x
        for i in range(n):
            bounds.append((1e-6, h - 1e-6))  # y
        for i in range(n):
            bounds.append((1e-6, max_r))  # r
        
        try:
            result = minimize(_objective, x0, args=(w, h, n), method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success or result.fun < best_obj:
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        return None
    
    # Extract and validate
    xs = best_result[0:n]
    ys = best_result[n:2*n]
    rs = best_result[2*n:3*n]
    
    # Clip to valid range
    rs = np.maximum(rs, 1e-8)
    xs = np.clip(xs, rs, w - rs)
    ys = np.clip(ys, rs, h - rs)
    
    # Verify no overlaps (with small tolerance)
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to fix by reducing radii
        scale = 0.95
        rs = rs * scale
        xs = np.clip(xs, rs, w - rs)
        ys = np.clip(ys, rs, h - rs)
    
    return np.column_stack([xs, ys, rs])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = N
    best_packing = None
    best_score = -np.inf
    
    # Try a range of aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        packing = _construct_for_aspect(ar, n, deadline)
        if packing is not None:
            # Score is sum of radii
            score = np.sum(packing[:, 2])
            if score > best_score:
                best_score = score
                best_packing = packing
    
    # If we didn't find anything, fall back to simple grid
    if best_packing is None:
        side = int(np.ceil(np.sqrt(n)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_packing = np.array(cells[:n])
    
    return best_packing
# EVOLVE-BLOCK-END

