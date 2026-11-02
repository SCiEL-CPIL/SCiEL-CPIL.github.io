import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def _check_validity(x, y, r, tol=1e-7):
    """Check if the configuration is valid (no overlaps, within bounds)."""
    # Boundary check
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _score(x, y, r):
    """Return the sum of radii if valid, else -inf."""
    if _check_validity(x, y, r):
        return np.sum(r)
    return -np.inf

def _hex_initial():
    """Generate a hexagonal packing initial guess."""
    # Hexagonal close packing: rows offset by half a spacing
    # For N=26, we can use a 5x5 grid with some removed, or 6x5
    # Let's try a hexagonal arrangement
    rows = 5
    cols = 6
    spacing = 1.0 / (cols + 0.5)  # account for offset
    offset = spacing * 0.5
    
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing
            if i % 2 == 1:
                x += offset
            if x < 1.0 and y < 1.0:
                points.append((x, y))
    
    # If we have too many, take first N
    # If too few, add more
    if len(points) >= N:
        points = points[:N]
    else:
        # Add random points
        while len(points) < N:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            points.append((x, y))
    
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Compute minimum distance between any pair
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            min_dist = min(min_dist, d)
    
    # Also check distance to boundaries
    for i in range(N):
        min_dist = min(min_dist, xs[i], 1 - xs[i], ys[i], 1 - ys[i])
    
    # Set radius to half the minimum distance (with safety margin)
    r = np.full(N, min_dist * 0.45)
    
    return xs, ys, r

def _grid_initial():
    """Generate a grid initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < N:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Add small jitter
    jitter = 0.02
    xs = xs + np.random.uniform(-jitter, jitter, N)
    ys = ys + np.random.uniform(-jitter, jitter, N)
    xs = np.clip(xs, 0, 1)
    ys = np.clip(ys, 0, 1)
    
    # Compute minimum distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            min_dist = min(min_dist, d)
    for i in range(N):
        min_dist = min(min_dist, xs[i], 1 - xs[i], ys[i], 1 - ys[i])
    
    r = np.full(N, min_dist * 0.45)
    return xs, ys, r

def _simulated_annealing(x, y, r, time_budget=30.0):
    """Run simulated annealing to escape local minima."""
    start = time.time()
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    current_score = best_score
    T = 0.05  # initial temperature
    cooling = 0.995
    
    while time.time() - start < time_budget and T > 1e-6:
        # Random move: perturb one circle's position and/or radius
        idx = np.random.randint(N)
        move_type = np.random.random()
        
        if move_type < 0.6:
            # Move position
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            x_new = x.copy()
            y_new = y.copy()
            x_new[idx] += dx
            y_new[idx] += dy
            r_new = r.copy()
        elif move_type < 0.85:
            # Change radius
            r_new = r.copy()
            r_new[idx] *= (1 + np.random.normal(0, 0.1))
            r_new[idx] = np.clip(r_new[idx], 0.005, 0.5)
            x_new = x.copy()
            y_new = y.copy()
        else:
            # Large random move (to escape local minima)
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            x_new[idx] = np.random.uniform(0, 1)
            y_new[idx] = np.random.uniform(0, 1)
            r_new[idx] = np.random.uniform(0.01, 0.2)
        
        # Clip to bounds
        x_new = np.clip(x_new, 0, 1)
        y_new = np.clip(y_new, 0, 1)
        
        new_score = _score(x_new, y_new, r_new)
        
        # Accept if better, or with probability exp((new - current)/T)
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            x, y, r = x_new, y_new, r_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= cooling
    
    return best_x, best_y, best_r, best_score

def _slsqp_polish(x, y, r, maxiter=200):
    """Final SLSQP polish pass with tight tolerance."""
    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    def objective(params):
        return -np.sum(params[2::3])
    
    def constraints(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        c_x_min = x_ - r_
        c_x_max = 1.0 - x_ - r_
        c_y_min = y_ - r_
        c_y_max = 1.0 - y_ - r_
        
        dx = x_[:, None] - x_[None, :]
        dy = y_[:, None] - y_[None, :]
        dist_sq = dx**2 + dy**2
        r_sum = r_[:, None] + r_[None, :]
        overlap_constr = dist_sq - r_sum**2
        
        iu = np.triu_indices(N, k=1)
        pair_constr = overlap_constr[iu]
        
        return np.concatenate([c_x_min, c_x_max, c_y_min, c_y_max, pair_constr])
    
    cons = ({'type': 'ineq', 'fun': constraints},)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        res = minimize(
            objective, params, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        if res.success or res.status == 0:
            x_res = res.x[0::3]
            y_res = res.x[1::3]
            r_res = res.x[2::3]
            if _check_validity(x_res, y_res, r_res, tol=1e-6):
                return x_res, y_res, r_res
    except Exception:
        pass
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple initializations
    init_funcs = [_hex_initial, _grid_initial]
    
    for k in range(10):
        if time.time() > deadline - 40:  # Leave time for SA and polish
            break
        
        # Choose initialization
        if k < len(init_funcs):
            x, y, r = init_funcs[k]()
        else:
            # Random initialization
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.05)
        
        # Run simulated annealing
        remaining = deadline - time.time() - 20  # Leave 20s for polish
        if remaining > 5:
            x, y, r, sa_score = _simulated_annealing(x, y, r, time_budget=min(remaining, 30))
        
        # Final SLSQP polish
        if time.time() < deadline - 10:
            x, y, r = _slsqp_polish(x, y, r, maxiter=150)
        
        score = _score(x, y, r)
        if score > best_score:
            best_score = score
            best_result = np.column_stack((x, y, r))
    
    if best_result is not None:
        return best_result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append((x, y, r))
        return np.array(cells)

# EVOLVE-BLOCK-END
