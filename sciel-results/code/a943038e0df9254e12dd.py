"""Improved circle packing in unit square using two-phase optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _max_radii_for_positions(positions, n, max_iter=200):
    """
    Given fixed positions, find maximum radii such that all circles are
    disjoint and within bounds.
    
    For each circle i, r_i <= min(
        x_i, 1-x_i, y_i, 1-y_i,
        min_{j!=i} (dist(i,j) - r_j) / 2  -- but this is circular
    )
    
    Better: solve r_i such that for all pairs, r_i + r_j <= dist(i,j)
    and r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    This is a linear program! Maximize sum(r_i) subject to:
      r_i + r_j <= dist(i,j) for all i<j
      r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      r_i >= 0
    """
    from scipy.optimize import linprog
    
    # Bounds from walls
    wall_bounds = np.minimum(
        np.minimum(positions[:, 0], 1 - positions[:, 0]),
        np.minimum(positions[:, 1], 1 - positions[:, 1])
    )
    
    # Build LP: maximize sum(r_i) s.t. A_ub @ r <= b_ub, 0 <= r <= wall_bounds
    # Constraints: r_i + r_j <= dist(i,j)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
    
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    c = -np.ones(n)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, wall_bounds[i]) for i in range(n)],
                  method='highs')
    
    if res.success:
        return -res.fun, res.x  # (max_sum, radii)
    else:
        # Fallback: use half the minimum wall bound
        r = wall_bounds * 0.5
        return np.sum(r), r

def _local_search_positions(positions, n, max_time=30.0):
    """
    Local search over positions using gradient-based optimization.
    Objective: maximize sum of radii (computed via LP at each step).
    """
    start_time = time.time()
    best_positions = positions.copy()
    best_sum, _ = _max_radii_for_positions(positions, n)
    
    # Use Nelder-Mead for derivative-free local search
    # Perturb positions and see if sum of radii improves
    
    step_size = 0.01
    n_iter = 0
    max_iter = 500
    
    while time.time() - start_time < max_time and n_iter < max_iter:
        n_iter += 1
        # Random perturbation
        delta = np.random.normal(0, step_size, size=positions.shape)
        new_positions = positions + delta
        # Clip to valid range
        new_positions = np.clip(new_positions, 0.05, 0.95)
        
        new_sum, _ = _max_radii_for_positions(new_positions, n)
        
        if new_sum > best_sum:
            positions = new_positions
            best_sum = new_sum
            step_size *= 0.95  # Reduce step on improvement
        else:
            step_size *= 1.05  # Increase step on failure
            step_size = min(step_size, 0.1)
    
    return best_positions, best_sum

def _hexagonal_initial(n, seed=42):
    """Generate hexagonal packing initial positions."""
    np.random.seed(seed)
    positions = []
    # Hexagonal lattice
    r_hex = 0.12  # approximate hex radius
    dx = 2 * r_hex
    dy = r_hex * np.sqrt(3)
    
    row = 0
    while len(positions) < n:
        col = 0
        while len(positions) < n:
            x = col * dx + (0.5 * dx if row % 2 == 1 else 0)
            y = row * dy
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                # Add small jitter
                x += np.random.uniform(-0.005, 0.005)
                y += np.random.uniform(-0.005, 0.005)
                positions.append([x, y])
            col += 1
        row += 1
    
    return np.array(positions[:n])

def _random_initial(n, seed=42):
    """Generate random initial positions."""
    np.random.seed(seed)
    positions = np.random.uniform(0.1, 0.9, size=(n, 2))
    return positions

def _grid_initial(n, seed=42):
    """Generate grid initial positions."""
    np.random.seed(seed)
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            x += np.random.uniform(-0.3/side, 0.3/side)
            y += np.random.uniform(-0.3/side, 0.3/side)
            positions.append([x, y])
    return np.array(positions[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_result = None
    best_score = -np.inf
    
    def update_best(positions, radii):
        nonlocal best_result, best_score
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            result = np.zeros((N, 3))
            result[:, 0] = positions[:, 0]
            result[:, 1] = positions[:, 1]
            result[:, 2] = radii
            best_result = result
    
    # Try different initial layouts
    initial_layouts = [
        ("hex", _hexagonal_initial),
        ("grid", _grid_initial),
        ("random", _random_initial),
    ]
    
    for layout_name, layout_func in initial_layouts:
        if time.time() > deadline:
            break
        
        # Try multiple seeds for each layout type
        for seed in range(5):
            if time.time() > deadline:
                break
            
            positions = layout_func(N, seed=seed + 100)
            
            # Phase 1: Compute max radii for initial positions
            sum_r, radii = _max_radii_for_positions(positions, N)
            update_best(positions, radii)
            
            # Phase 2: Local search to improve positions
            if time.time() > deadline:
                break
            
            improved_positions, improved_sum = _local_search_positions(
                positions, N, max_time=min(20.0, deadline - time.time())
            )
            
            if improved_sum > sum_r:
                _, improved_radii = _max_radii_for_positions(improved_positions, N)
                update_best(improved_positions, improved_radii)
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
