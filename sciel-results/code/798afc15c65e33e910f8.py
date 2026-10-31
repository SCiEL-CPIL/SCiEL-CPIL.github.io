"""Rewritten circle-packing program: 26 circles in [0,1]^2 maximizing sum of radii."""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-8):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-7):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    # 5 rows with varying counts: 5,5,5,5,6
    rows_counts = [5, 5, 5, 5, 6]
    circles = []
    total = sum(rows_counts)
    # Scale to fit
    # For hexagonal: row spacing = r * sqrt(3), col spacing = 2*r
    # We want total width ~ 2*r*(max_cols-1) + 2*r <= 1
    # Let's just place them and scale later
    r = 0.08
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows_counts:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    # If we have more than N, truncate; if less, add
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.45 / side
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.02, 0.08)
        circles.append((x, y, r))
    return circles


def _grow_all(circles, num_steps=200, step_size=0.005):
    """Iteratively grow all circles by a small amount, then fix overlaps."""
    circles = [list(c) for c in circles]
    for step in range(num_steps):
        # Try to grow all circles
        for i in range(N):
            circles[i][2] += step_size
        # Fix overlaps by pushing apart and shrinking
        for it in range(50):
            max_overlap = 0.0
            max_i, max_j = -1, -1
            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    overlap = ri + rj - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_i, max_j = i, j
            # Check bounds
            for i in range(N):
                x, y, r = circles[i]
                if x < r or x > 1 - r or y < r or y > 1 - r:
                    max_overlap = max(max_overlap, 0.01)
                    max_i, max_j = i, i
            
            if max_i == -1 or max_overlap < 1e-10:
                break
            
            if max_i != max_j:
                xi, yi, ri = circles[max_i]
                xj, yj, rj = circles[max_j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < 1e-10:
                    dx, dy = 0.01, 0.01
                else:
                    dx = (xi - xj) / dist
                    dy = (yi - yj) / dist
                # Push apart
                push = max_overlap / 2
                circles[max_i][0] += dx * push
                circles[max_i][1] += dy * push
                circles[max_j][0] -= dx * push
                circles[max_j][1] -= dy * push
            # Shrink both slightly
            circles[max_i][2] -= max_overlap * 0.6
            if max_i != max_j:
                circles[max_j][2] -= max_overlap * 0.6
            # Clamp
            for i in range(N):
                circles[i][2] = max(0.001, circles[i][2])
                circles[i][0] = min(1.0 - circles[i][2], max(circles[i][2], circles[i][0]))
                circles[i][1] = min(1.0 - circles[i][2], max(circles[i][2], circles[i][1]))
    
    return [tuple(c) for c in circles]


def _local_search(circles, deadline, max_iters=200):
    """Local search: try moving/shrinking/growing individual circles."""
    circles = [list(c) for c in circles]
    best_sum = _sum_radii(circles)
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(N):
            x, y, r = circles[i]
            # Try perturbations
            perturbations = []
            for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dr in [-0.01, 0.005, 0.01, 0.02]:
                        if dx == 0 and dy == 0 and dr == 0:
                            continue
                        perturbations.append((dx, dy, dr))
            random.shuffle(perturbations)
            
            for dx, dy, dr in perturbations:
                new_x = x + dx
                new_y = y + dy
                new_r = max(0.005, r + dr)
                # Bounds check
                if new_x - new_r < -1e-9 or new_x + new_r > 1 + 1e-9:
                    continue
                if new_y - new_r < -1e-9 or new_y + new_r > 1 + 1e-9:
                    continue
                # Overlap check
                valid = True
                for j in range(N):
                    if j == i:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((new_x - xj)**2 + (new_y - yj)**2)
                    if dist < new_r + rj - 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                new_sum = best_sum - r + new_r
                if new_sum > best_sum + 1e-12:
                    circles[i] = [new_x, new_y, new_r]
                    best_sum = new_sum
                    improved = True
                    break
        if not improved:
            break
    
    return [tuple(c) for c in circles], best_sum


def _slsqp_polish(circles, deadline):
    """SLSQP polish with pairwise constraints."""
    def params_to_circles(params):
        return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    
    def objective(params):
        return -sum(params[3*i+2] for i in range(N))
    
    def _jac_overlap(p, i, j):
        jac = np.zeros(3 * N)
        xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
        xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
        jac[3*i] = 2 * (xi - xj)
        jac[3*j] = -2 * (xi - xj)
        jac[3*i+1] = 2 * (yi - yj)
        jac[3*j+1] = -2 * (yi - yj)
        jac[3*i+2] = -2 * (ri + rj)
        jac[3*j+2] = -2 * (ri + rj)
        return jac
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i+2 else 0.0 for k in range(3*N)])})
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
            })
    
    params = np.array([val for circle in circles for val in circle])
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        result = minimize(
            objective, params, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 80, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = params_to_circles(result.x)
            if _is_valid(new_circles):
                return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    best_circles = _init_hexagonal()
    best_sum = _sum_radii(best_circles)
    
    # Try multiple initialization strategies
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
    ]
    
    for init_circles in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Grow all circles
        grown = _grow_all(init_circles, num_steps=100, step_size=0.003)
        grown_sum = _sum_radii(grown)
        
        # Local search
        refined, refined_sum = _local_search(grown, deadline, max_iters=150)
        
        if refined_sum > best_sum and _is_valid(refined):
            best_circles = refined
            best_sum = refined_sum
    
    # SLSQP polish on best
    if time.time() < deadline - 30.0:
        polished = _slsqp_polish(best_circles, deadline)
        polished_sum = _sum_radii(polished)
        if polished_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = polished_sum
    
    # Another round of local search
    if time.time() < deadline - 15.0:
        refined, refined_sum = _local_search(best_circles, deadline, max_iters=100)
        if refined_sum > best_sum and _is_valid(refined):
            best_circles = refined
            best_sum = refined_sum
    
    # Final post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback to a simple valid configuration
        best_circles = _init_grid()
        best_circles = _post_process(best_circles)
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
