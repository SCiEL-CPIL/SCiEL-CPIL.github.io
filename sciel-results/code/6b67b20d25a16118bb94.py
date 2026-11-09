import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

N = 26

def _solve_radii(positions):
    """
    Given positions, solve for maximum radii using Linear Programming.
    Maximize sum(r)
    Subject to:
      r_i + r_j <= d_ij for all i < j
      r_i <= 1 - x_i
      r_i <= x_i
      r_i <= 1 - y_i
      r_i <= y_i
      r_i >= 0
    """
    n = len(positions)
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # 1. Pairwise constraints: r_i + r_j <= d_ij
    # 2. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    # Calculate distances
    dists = cdist(positions, positions)
    
    A_rows = []
    b_rows = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_rows.append(dists[i, j])
            
    # Boundary constraints
    for i in range(n):
        x, y = positions[i]
        max_r_boundary = min(x, 1 - x, y, 1 - y)
        if max_r_boundary < 0:
            return None # Invalid position, circle cannot exist
            
        # r_i <= max_r_boundary
        # This can be split into 4 constraints or just one. One is sufficient for LP.
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_rows.append(max_r_boundary)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds for r: [0, None]
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun # Sum of radii
        else:
            return 0.0
    except Exception:
        return 0.0

def _score(positions):
    """Calculate the total radius sum for given positions."""
    return _solve_radii(positions)

def _local_search(positions, time_limit=10.0):
    """
    Perform local search on positions to maximize sum of radii.
    Uses a simple random perturbation / hill climbing approach.
    """
    best_pos = positions.copy()
    best_score = _score(best_pos)
    if best_score is None:
        best_score = 0.0
    
    start_time = time.time()
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Generate a random perturbation
        # Reduce step size if stuck
        if no_improve_count > 50:
            step_size *= 0.9
            no_improve_count = 0 # Reset to allow more exploration with smaller steps
            
        new_pos = best_pos + np.random.normal(0, step_size, size=best_pos.shape)
        
        # Clamp positions to valid range? 
        # The LP will handle boundaries by setting r=0 if out of bounds, 
        # but it's better to keep centers within [0,1] to avoid trivial r=0 solutions.
        new_pos = np.clip(new_pos, 0, 1)
        
        new_score = _score(new_pos)
        
        if new_score is None:
            new_score = 0.0
        
        if new_score > best_score + 1e-6:
            best_pos = new_pos
            best_score = new_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            break
            
    return best_pos, best_score

def _make_initial_grid():
    """Create a grid of 26 circles."""
    # 5x5 grid = 25, add 1 more? Or 4x7? 
    # Let's try 5x5 and place the 26th in the center of one cell, splitting it? 
    # Or just 5x5 grid and randomize the 26th?
    # Better: 5x5 grid, take first 25, and for the 26th, pick a random spot.
    # Or use a hexagonal packing initialization?
    
    # Simple Grid 5x5
    positions = []
    side = 5
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
            
    # We have 25. Need 1 more.
    # Place it at (0.5, 0.5) ? It will overlap.
    # Let's just randomize the 26th position.
    positions.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
    
    return np.array(positions)

def _make_initial_random():
    """Create a random set of positions."""
    return np.random.uniform(0.1, 0.9, size=(N, 2))

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    np.random.seed(42)
    random.seed(42)
    
    best_positions = None
    best_score = 0.0
    
    # 1. Try Grid Initialization
    pos_grid = _make_initial_grid()
    pos_opt, score_opt = _local_search(pos_grid, time_limit=30.0)
    if score_opt > best_score:
        best_score = score_opt
        best_positions = pos_opt.copy()
        
    # 2. Try Random Initializations with Local Search
    restarts = 0
    while time.time() < deadline:
        pos_rand = _make_initial_random()
        
        # Shorter time limit per restart to allow more restarts
        time_per_restart = 10.0
        if time.time() + time_per_restart > deadline:
            time_per_restart = max(1.0, deadline - time.time())
            
        pos_opt, score_opt = _local_search(pos_rand, time_limit=time_per_restart)
        
        if score_opt > best_score:
            best_score = score_opt
            best_positions = pos_opt.copy()
            restarts += 1
            
        # If we've found a good solution, we can stop earlier?
        # Or continue until deadline.
        
    # Final Polish: Run a longer local search on the best found so far
    if best_positions is not None and time.time() < deadline - 5:
        time_remaining = deadline - time.time() - 5
        pos_opt, score_opt = _local_search(best_positions, time_limit=time_remaining)
        if score_opt > best_score:
            best_score = score_opt
            best_positions = pos_opt.copy()
            
    # Construct the result array
    # We need the radii corresponding to best_positions
    # _solve_radii returns the score (sum), but we need individual radii.
    # Let's re-solve the LP to get the radii vector.
    
    def _get_radii(positions):
        n = len(positions)
        c = -np.ones(n)
        dists = cdist(positions, positions)
        A_rows = []
        b_rows = []
        
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_rows.append(row)
                b_rows.append(dists[i, j])
                
        for i in range(n):
            x, y = positions[i]
            max_r_boundary = min(x, 1 - x, y, 1 - y)
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(max_r_boundary)
            
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(n)
        except Exception:
            return np.zeros(n)
            
    radii = _get_radii(best_positions)
    
    # Combine into (x, y, r) array
    result = np.hstack([best_positions, radii.reshape(-1, 1)])
    
    # Final validation
    # Ensure no negative radii
    result[:, 2] = np.maximum(result[:, 2], 0)
    
    return result

