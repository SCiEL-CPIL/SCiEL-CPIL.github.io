import numpy as np
import time
from scipy.optimize import linprog
import random

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        row = k // 4
        col = k % 4
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _get_topology(centers, sides):
    """
    Extracts a valid topology from a given packing.
    Returns a list of relations for each pair (i, j) with i < j.
    Relation: 0 (left), 1 (right), 2 (below), 3 (above).
    """
    n = len(sides)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    relations = []
    for i, j in pairs:
        cx_i, cy_i = centers[i]
        cx_j, cy_j = centers[j]
        s_i, s_j = sides[i], sides[j]
        
        # Check which non-overlap condition is satisfied.
        # We need to pick one that is true.
        # If they touch, multiple might be true. We pick the first one found.
        
        # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
        if cx_i + s_i/2 <= cx_j - s_j/2 + 1e-9:
            relations.append(0)
        # i right of j: cx_i - s_i/2 >= cx_j + s_j/2
        elif cx_i - s_i/2 >= cx_j + s_j/2 - 1e-9:
            relations.append(1)
        # i below j: cy_i + s_i/2 <= cy_j - s_j/2
        elif cy_i + s_i/2 <= cy_j - s_j/2 + 1e-9:
            relations.append(2)
        # i above j: cy_i - s_i/2 >= cy_j + s_j/2
        elif cy_i - s_i/2 >= cy_j + s_j/2 - 1e-9:
            relations.append(3)
        else:
            # Should not happen if packing is valid.
            # Default to left
            relations.append(0)
            
    return relations

def _solve_lp(topology, n=10, deadline=None):
    """
    Solves the LP for a given topology.
    Returns (centers, sides, value) or None if infeasible or timeout.
    """
    # Variables: cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...
    # Index: 3*i + 0 for cx, 3*i + 1 for cy, 3*i + 2 for s
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i + 2] = -1.0
        
    # Bounds
    # cx, cy in [0, 1]
    # s in [0.01, 1.0] (0.01 to avoid 0)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s
        
    # Constraints A_ub x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Boundary constraints
    # cx_i - s_i/2 >= 0  => -cx_i + 0.5*s_i <= 0
    # 1 - cx_i - s_i/2 >= 0 => cx_i + 0.5*s_i <= 1
    # Same for y
    
    for i in range(n):
        # -cx_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # cx_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # -cy_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # cy_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
    # 2. Topology constraints
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for k, (i, j) in enumerate(pairs):
        rel = topology[k]
        row = np.zeros(3*n)
        
        if rel == 0: # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
            # cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0
            row[3*i] = 1.0
            row[3*j] = -1.0
            row[3*i+2] = 0.5
            row[3*j+2] = 0.5
            b_val = 0.0
        elif rel == 1: # i right of j: cx_i - s_i/2 >= cx_j + s_j/2
            # -cx_i + cx_j - 0.5*s_i - 0.5*s_j <= 0
            row[3*i] = -1.0
            row[3*j] = 1.0
            row[3*i+2] = -0.5
            row[3*j+2] = -0.5
            b_val = 0.0
        elif rel == 2: # i below j: cy_i + s_i/2 <= cy_j - s_j/2
            # cy_i - cy_j + 0.5*s_i + 0.5*s_j <= 0
            row[3*i+1] = 1.0
            row[3*j+1] = -1.0
            row[3*i+2] = 0.5
            row[3*j+2] = 0.5
            b_val = 0.0
        elif rel == 3: # i above j: cy_i - s_i/2 >= cy_j + s_j/2
            # -cy_i + cy_j - 0.5*s_i - 0.5*s_j <= 0
            row[3*i+1] = -1.0
            row[3*j+1] = 1.0
            row[3*i+2] = -0.5
            row[3*j+2] = -0.5
            b_val = 0.0
        else:
            continue
            
        A_ub_rows.append(row)
        b_ub_vals.append(b_val)
        
    if not A_ub_rows:
        A_ub = np.array([])
        b_ub = np.array([])
    else:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None
        
    if not res.success:
        return None
        
    x = res.x
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    for i in range(n):
        centers[i, 0] = x[3*i]
        centers[i, 1] = x[3*i+1]
        sides[i] = x[3*i+2]
        
    val = np.sum(sides)
    return centers, sides, val

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 620
    
    # Initial best
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    # Get initial topology
    current_topology = _get_topology(best_centers, best_sides)
    current_val = best_val
    
    # Simulated Annealing on Topologies
    # We define a move as changing one relation in the topology.
    # We solve the LP for the new topology.
    # We accept the move based on SA criteria.
    
    T_sa = 0.1 # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-4
    
    # To speed up, we can limit the number of SA steps or time.
    # Each LP solve takes ~10-50ms. 600s allows ~10,000-30,000 solves.
    # We will run SA until time runs out.
    
    n_pairs = len(current_topology)
    
    while time.time() < deadline and T_sa > min_T:
        # Pick a random pair
        k = random.randint(0, n_pairs - 1)
        original_rel = current_topology[k]
        
        # Pick a new relation different from original
        new_rel = random.choice([r for r in range(4) if r != original_rel])
        
        # Create new topology
        new_topology = current_topology.copy()
        new_topology[k] = new_rel
        
        # Solve LP
        res = _solve_lp(new_topology, n, deadline)
        
        if res is None:
            # Infeasible, reject
            continue
            
        c_new, s_new, val_new = res
        
        # Check validity just in case
        if not _is_valid_vectorized(c_new, s_new, tol=tol):
            continue
            
        # Calculate delta
        delta = val_new - current_val
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Boltzmann factor
            # If T is very small, exp(delta/T) will be 0
            try:
                prob = np.exp(delta / T_sa)
            except OverflowError:
                prob = 0.0
            accept = random.random() < prob
            
        if accept:
            current_topology = new_topology
            current_val = val_new
            # Update current centers/sides? 
            # We only need them if we want to check validity or restart.
            # For the next move, we just need the topology.
            
            if val_new > best_val:
                best_val = val_new
                best_centers = c_new
                best_sides = s_new
        else:
            # Reject move, keep current topology and value
            pass
            
        # Cool down
        T_sa *= cooling_rate
        
    # Final check
    if not _is_valid_vectorized(best_centers, best_sides, tol=tol):
        # Fallback
        best_centers, best_sides = _initial_packing_grid()
        best_val = 3.0
        
    return best_centers, best_sides
