"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_score(circles):
    """Compute the sum of radii for a valid packing, 0 if invalid."""
    n = len(circles)
    if n == 0:
        return 0.0
    valid = True
    for i in range(n):
        x, y, r = circles[i]
        if r < 1e-10:
            continue
        # Check boundary
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
    if not valid:
        return 0.0
    # Check pairwise
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - 1e-9:
                valid = False
                break
        if not valid:
            break
    if not valid:
        return 0.0
    return float(np.sum(circles[:, 2]))


def _local_search(circles, max_iter=2000, step_init=0.01, step_decay=0.999):
    """
    Simple local search: iteratively adjust each circle to maximize radius
    while maintaining non-overlap and boundary constraints.
    """
    n = len(circles)
    circles = circles.copy()
    step = step_init
    best_score = _compute_score(circles)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            x, y, r = circles[i]
            # Try to increase radius
            new_r = r + step * 0.5
            # Check if we can increase radius
            feasible = True
            # Boundary check
            if x - new_r < -1e-9 or x + new_r > 1 + 1e-9 or y - new_r < -1e-9 or y + new_r > 1 + 1e-9:
                # Try to move center to accommodate larger radius
                feasible = False
            
            if feasible:
                # Check pairwise
                for j in range(n):
                    if j == i:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.hypot(x - xj, y - yj)
                    if dist < new_r + rj - 1e-9:
                        feasible = False
                        break
            
            if feasible:
                circles[i, 2] = new_r
                improved = True
            else:
                # Try small random perturbations
                for _ in range(5):
                    dx = random.gauss(0, step * 0.1)
                    dy = random.gauss(0, step * 0.1)
                    nx, ny = x + dx, y + dy
                    nr = r + step * 0.3
                    # Check feasibility
                    ok = True
                    if nx - nr < -1e-9 or nx + nr > 1 + 1e-9 or ny - nr < -1e-9 or ny + nr > 1 + 1e-9:
                        ok = False
                    else:
                        for j in range(n):
                            if j == i:
                                continue
                            xj, yj, rj = circles[j]
                            dist = np.hypot(nx - xj, ny - yj)
                            if dist < nr + rj - 1e-9:
                                ok = False
                                break
                    if ok:
                        circles[i, 0] = nx
                        circles[i, 1] = ny
                        circles[i, 2] = nr
                        improved = True
                        break
        if not improved:
            step *= step_decay
            if step < 1e-6:
                break
    
    return circles


def _scipy_optimize(circles, deadline):
    """
    Use scipy.optimize.minimize with L-BFGS-B to maximize sum of radii.
    Variables: [x1, y1, r1, x2, y2, r2, ...]
    Objective: -sum(r_i)
    Constraints: 
      - Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - Pairwise: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    We use penalty method or convert to inequality constraints.
    """
    n = len(circles)
    x0 = circles.flatten().copy()
    
    def objective(v):
        return -np.sum(v[2::3])
    
    def objective_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g
    
    # Use penalty approach: add large penalty for violations
    def penalized_objective(v, penalty=1000.0):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        obj = -np.sum(r)
        # Boundary penalties
        for i in range(n):
            obj += penalty * max(0, r[i] - x[i])
            obj += penalty * max(0, r[i] - (1 - x[i]))
            obj += penalty * max(0, r[i] - y[i])
            obj += penalty * max(0, r[i] - (1 - y[i]))
        # Pairwise penalties
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(x[i] - x[j], y[i] - y[j])
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    obj += penalty * (min_dist - dist) ** 2
        return obj
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 0.5))  # x
        bounds.append((1e-6, 0.5))  # y
        bounds.append((1e-6, 0.5))  # r
    
    try:
        res = minimize(penalized_objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            new_circles = res.x.reshape(n, 3)
            # Clamp to valid range
            new_circles[:, 0] = np.clip(new_circles[:, 0], 0.01, 0.99)
            new_circles[:, 1] = np.clip(new_circles[:, 1], 0.01, 0.99)
            new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, 0.5)
            # Validate and repair
            for i in range(n):
                xi, yi, ri = new_circles[i]
                # Ensure boundary
                ri = min(ri, xi, 1 - xi, yi, 1 - yi)
                new_circles[i, 2] = ri
            # Ensure pairwise by reducing radii
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.hypot(new_circles[i, 0] - new_circles[j, 0],
                                    new_circles[i, 1] - new_circles[j, 1])
                    min_r = max(0.001, (dist - 1e-6) / 2)
                    new_circles[i, 2] = min(new_circles[i, 2], min_r)
                    new_circles[j, 2] = min(new_circles[j, 2], min_r)
            # Re-check boundary after pairwise reduction
            for i in range(n):
                xi, yi, ri = new_circles[i]
                ri = min(ri, xi, 1 - xi, yi, 1 - yi)
                new_circles[i, 2] = ri
            return new_circles
    except Exception:
        pass
    return circles


def _generate_initial_packing(seed):
    """Generate an initial packing using a grid or random placement."""
    rng = np.random.RandomState(seed)
    
    # Try a few different initial configurations
    best_circles = None
    best_score = 0
    
    # Configuration 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
    circles_grid = np.array(cells[:N])
    score = _compute_score(circles_grid)
    if score > best_score:
        best_score = score
        best_circles = circles_grid.copy()
    
    # Configuration 2: Random with rejection
    for attempt in range(10):
        circles = []
        rng2 = np.random.RandomState(seed * 1000 + attempt)
        for _ in range(N):
            for _ in range(100):
                x = rng2.uniform(0.05, 0.95)
                y = rng2.uniform(0.05, 0.95)
                r = rng2.uniform(0.02, 0.15)
                # Check boundary
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    continue
                # Check overlap
                ok = True
                for cx, cy, cr in circles:
                    if np.hypot(x - cx, y - cy) < r + cr - 1e-9:
                        ok = False
                        break
                if ok:
                    circles.append([x, y, r])
                    break
        if len(circles) == N:
            circles_arr = np.array(circles)
            score = _compute_score(circles_arr)
            if score > best_score:
                best_score = score
                best_circles = circles_arr.copy()
    
    if best_circles is None:
        # Fallback: very small circles
        best_circles = np.array([[0.5, 0.5, 0.01]] * N)
    
    return best_circles


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_circles = None
    best_score = 0
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        # Generate initial packing
        circles = _generate_initial_packing(seed)
        
        # Local search
        circles = _local_search(circles, max_iter=1000)
        score = _compute_score(circles)
        
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        # Scipy optimization
        if time.time() < deadline - 10:
            circles_scipy = _scipy_optimize(circles, deadline)
            score_scipy = _compute_score(circles_scipy)
            if score_scipy > best_score:
                best_score = score_scipy
                best_circles = circles_scipy.copy()
            
            # Another local search round
            circles_final = _local_search(circles_scipy, max_iter=500)
            score_final = _compute_score(circles_final)
            if score_final > best_score:
                best_score = score_final
                best_circles = circles_final.copy()
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final validation and repair
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 0.001)
        best_circles[i] = [x, y, r]
    
    # Final pairwise repair
    for i in range(len(best_circles)):
        for j in range(i + 1, len(best_circles)):
            dist = np.hypot(best_circles[i, 0] - best_circles[j, 0],
                            best_circles[i, 1] - best_circles[j, 1])
            min_r = max(0.001, (dist - 1e-6) / 2)
            best_circles[i, 2] = min(best_circles[i, 2], min_r)
            best_circles[j, 2] = min(best_circles[j, 2], min_r)
    
    # Final boundary check
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 0.001)
        best_circles[i, 2] = r
    
    return best_circles

# EVOLVE-BLOCK-END
