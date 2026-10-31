"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 138.0


def _validity(circles, W, H):
    """Check disjoint interiors and bounds. Returns (ok, perimeter, sum_radii)."""
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
            if d < circles[i][2] + circles[j][2] - 1e-7:
                return False, per, 0.0
    s = sum(c[2] for c in circles)
    return True, per, s


def _scale_to_perimeter(circles, target_per=PERIMETER):
    """Scale the whole packing so bounding-box perimeter equals target_per."""
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per <= 0:
        return circles
    s = target_per / per
    out = []
    for (x, y, r) in circles:
        nx = (x - minx) * s
        ny = (y - miny) * s
        nr = r * s
        out.append((nx, ny, nr))
    return out


def _penalty_objective(x, W, H, mu=100.0):
    """
    Objective function: Minimize -sum(r) + penalty.
    x = [cx, cy, r]
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    # Ensure radii are positive for the objective calculation, 
    # though bounds will handle it. 
    r_eff = np.maximum(r, 1e-6)
    
    val = -np.sum(r_eff)
    
    # Wall violations (soft)
    # x < r  => x - r < 0
    # x > W - r => x - (W - r) > 0
    wall_pen = 0.0
    for i in range(N):
        # Left
        if cx[i] - r_eff[i] < 0:
            wall_pen += (cx[i] - r_eff[i]) ** 2
        # Right
        if cx[i] - (W - r_eff[i]) > 0:
            wall_pen += (cx[i] - (W - r_eff[i])) ** 2
        # Bottom
        if cy[i] - r_eff[i] < 0:
            wall_pen += (cy[i] - r_eff[i]) ** 2
        # Top
        if cy[i] - (H - r_eff[i]) > 0:
            wall_pen += (cy[i] - (H - r_eff[i])) ** 2
            
    val += mu * wall_pen
    
    # Overlap violations (soft)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r_eff[i] + r_eff[j]
            if d < min_d:
                overlap_pen += (min_d - d) ** 2
                
    val += mu * overlap_pen
    
    return val


def _hexagonal_layout(W, H, seed=0):
    """Generate a hexagonal lattice layout scaled to fit WxH."""
    rng = random.Random(seed)
    # Estimate number of circles per row
    # Area per circle ~ pi * r^2. Total area W*H. 
    # Rough estimate for r: sqrt(W*H / N)
    r_est = np.sqrt(W * H / N)
    
    # Hexagonal packing:
    # Row 0: x = 0, r_est, 2*r_est...
    # Row 1: x = r_est, 3*r_est... y = r_est * sqrt(3)
    
    points = []
    r = r_est * 0.9 # Slightly smaller to avoid immediate boundary issues
    
    y = r
    row = 0
    while y < H - r:
        x = r
        if row % 2 == 1:
            x += r
        while x < W - r:
            points.append((x, y, r))
            x += 2 * r
        y += r * np.sqrt(3)
        row += 1
        
    # If we have too many points, remove some (e.g., random or farthest from center)
    # If we have too few, we just use what we have and let optimization expand them.
    # Actually, we need exactly N points.
    # If len(points) < N, add random points.
    # If len(points) > N, keep the N closest to the center? Or just truncate?
    # Truncating might break the lattice structure.
    
    if len(points) > N:
        # Sort by distance to center (W/2, H/2) and keep closest N?
        # Or just take the first N? Taking first N is biased to top-left.
        # Let's shuffle and take N? No, that destroys lattice.
        # Let's keep the ones closest to the center.
        center_x, center_y = W/2, H/2
        points.sort(key=lambda p: (p[0]-center_x)**2 + (p[1]-center_y)**2)
        points = points[:N]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            points.append((rng.uniform(r, W-r), rng.uniform(r, H-r), r))
            
    return points


def _optimize_aspect(W, H, seed=0):
    """Optimize circle positions and radii for a fixed aspect ratio W:H."""
    rng = random.Random(seed)
    best_circles = None
    best_val = -float('inf')
    
    # Initial configuration: Hexagonal
    points = _hexagonal_layout(W, H, seed)
    cx = np.array([p[0] for p in points])
    cy = np.array([p[1] for p in points])
    r = np.array([p[2] for p in points])
    
    x0 = np.concatenate([cx, cy, r])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-4, W - 1e-4)) # cx
    for i in range(N):
        bounds.append((1e-4, H - 1e-4)) # cy
    for i in range(N):
        bounds.append((1e-4, min(W, H) / 2)) # r
    
    # Optimization
    # Try L-BFGS-B first, it's faster for this smooth-ish objective
    try:
        res = minimize(_penalty_objective, x0, args=(W, H, 100.0), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9, 'maxls': 50})
        
        if res.fun < _penalty_objective(x0, W, H, 100.0):
            cx_opt = res.x[:N]
            cy_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Post-processing: Ensure validity by clipping and resolving overlaps
            # The penalty method might leave slight overlaps if mu is not high enough.
            # We do a "hard" relaxation step here.
            
            circles = list(zip(cx_opt, cy_opt, r_opt))
            
            # Hard relaxation: Push circles apart if overlapping
            for _ in range(50):
                changed = False
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dx = xi - xj
                        dy = yi - yj
                        d = np.hypot(dx, dy)
                        min_d = ri + rj
                        if d < min_d - 1e-8:
                            # Push apart
                            if d < 1e-8:
                                dx = 1.0
                                dy = 0.0
                                d = 1.0
                            push = (min_d - d) / 2.0 + 1e-4
                            nx_i = xi + (dx/d)*push
                            ny_i = yi + (dy/d)*push
                            nx_j = xj - (dx/d)*push
                            ny_j = yj - (dy/d)*push
                            
                            # Check bounds
                            if nx_i >= ri and nx_i <= W-ri and ny_i >= ri and ny_i <= H-ri:
                                circles[i] = (nx_i, ny_i, ri)
                            if nx_j >= rj and nx_j <= W-rj and ny_j >= rj and ny_j <= H-rj:
                                circles[j] = (nx_j, ny_j, rj)
                            changed = True
                if not changed:
                    break
            
            ok, per, s = _validity(circles, W, H)
            if ok and s > best_val:
                best_val = s
                best_circles = circles
                
    except Exception as e:
        pass
    
    return best_circles, best_val


def _tighten_local_search(circles, W, H, steps=1000):
    """
    Local search to maximize sum of radii.
    Iteratively:
    1. For each circle, increase radius as much as possible (limited by walls and neighbors).
    2. For each circle, move it to a position that allows a larger radius.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(42)
    
    for step in range(steps):
        if time.time() > DEADLINE - 5:
            break
            
        improved = False
        
        # 1. Increase radii
        for i in range(n):
            x, y, r = circles[i]
            # Max r allowed by walls
            max_r_wall = min(x, W-x, y, H-y)
            # Max r allowed by neighbors
            max_r_neighbor = max_r_wall
            for j in range(n):
                if i == j: continue
                xj, yj, rj = circles[j]
                d = np.hypot(x-xj, y-yj)
                if d > 0:
                    max_r_neighbor = min(max_r_neighbor, d - rj)
            
            if max_r_neighbor > r + 1e-8:
                circles[i] = [x, y, max_r_neighbor]
                improved = True
        
        # 2. Move circles to allow larger radii
        # Pick a random circle, try small moves
        i = rng.randrange(n)
        x, y, r = circles[i]
        best_r = r
        best_x, best_y = x, y
        
        # Try 10 random small moves
        for _ in range(10):
            dx = rng.uniform(-0.05, 0.05)
            dy = rng.uniform(-0.05, 0.05)
            nx, ny = x + dx, y + dy
            
            # Check bounds for current radius
            if nx < r or nx > W-r or ny < r or ny > H-r:
                continue
                
            # Check if this position allows a larger radius
            # The new radius is limited by:
            # 1. Walls: min(nx, W-nx, ny, H-ny)
            # 2. Neighbors: d(nx, ny, xj, yj) - rj
            nr = min(nx, W-nx, ny, H-ny)
            for j in range(n):
                if i == j: continue
                xj, yj, rj = circles[j]
                d = np.hypot(nx-xj, ny-yj)
                nr = min(nr, d - rj)
            
            if nr > best_r:
                best_r = nr
                best_x, best_y = nx, ny
                
        if best_r > r + 1e-8:
            circles[i] = [best_x, best_y, best_r]
            improved = True
            
        if not improved:
            # Try a larger move? Or break?
            # If no improvement in a step, we might be stuck.
            # But we alternate radius increase and move.
            pass

    return circles


def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    # Try a range of aspects
    aspect_choices = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.15, 0.85]
    random.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE - 10:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        circles, val = _optimize_aspect(W, H, seed=int(W * 100))
        if circles is not None and val > best_val:
            # Refine with local search
            circles = _tighten_local_search(circles, W, H, steps=2000)
            # Scale to exact perimeter 4
            circles = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles
    
    if best is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
