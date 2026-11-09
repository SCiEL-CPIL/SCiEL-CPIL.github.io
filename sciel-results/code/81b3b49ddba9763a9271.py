# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import itertools

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    deadline = time.time() + 570  # safety margin before 600s limit
    
    # Variables: x_0, y_0, s_0, ..., x_9, y_9, s_9
    # Indexing: 3*i -> x_i, 3*i+1 -> y_i, 3*i+2 -> s_i
    
    # Objective: Maximize sum(s_i) -> Minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i + 2] = -1.0
        
    # Bounds
    # x_i, y_i in [0, 1], s_i in [1e-6, 1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 1)) # s
        
    # Inequality constraints: A_ub x <= b_ub
    # 1. Boundary constraints
    # x_i - s_i/2 >= 0  => -x_i + 0.5*s_i <= 0
    # x_i + s_i/2 <= 1  => x_i + 0.5*s_i <= 1
    # y_i - s_i/2 >= 0  => -y_i + 0.5*s_i <= 0
    # y_i + s_i/2 <= 1  => y_i + 0.5*s_i <= 1
    
    A_ub_list = []
    b_ub_list = []
    
    for i in range(n):
        # -x_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub_list.append(row)
        b_ub_list.append(0.0)
        
        # x_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub_list.append(row)
        b_ub_list.append(1.0)
        
        # -y_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub_list.append(row)
        b_ub_list.append(0.0)
        
        # y_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub_list.append(row)
        b_ub_list.append(1.0)
        
    base_A = np.array(A_ub_list)
    base_b = np.array(b_ub_list)
    
    best_obj = -np.inf
    best_x = None
    
    # Generate random topologies
    # For each pair (i, j), choose one of 4 separation constraints:
    # 0: i left of j (x_i + s_i/2 <= x_j - s_j/2)
    # 1: j left of i (x_j + s_j/2 <= x_i - s_i/2)
    # 2: i below j (y_i + s_i/2 <= y_j - s_j/2)
    # 3: j below i (y_j + s_j/2 <= y_i - s_i/2)
    
    pairs = list(itertools.combinations(range(n), 2))
    rng = np.random.default_rng(42)
    
    # Pre-calculate rows for each pair and option
    pair_rows = {}
    for i, j in pairs:
        rows = []
        # 0: i left of j
        r0 = np.zeros(3*n)
        r0[3*i] = 1.0
        r0[3*i+2] = 0.5
        r0[3*j] = -1.0
        r0[3*j+2] = 0.5
        rows.append((r0, 0.0))
        
        # 1: j left of i
        r1 = np.zeros(3*n)
        r1[3*j] = 1.0
        r1[3*j+2] = 0.5
        r1[3*i] = -1.0
        r1[3*i+2] = 0.5
        rows.append((r1, 0.0))
        
        # 2: i below j
        r2 = np.zeros(3*n)
        r2[3*i+1] = 1.0
        r2[3*i+2] = 0.5
        r2[3*j+1] = -1.0
        r2[3*j+2] = 0.5
        rows.append((r2, 0.0))
        
        # 3: j below i
        r3 = np.zeros(3*n)
        r3[3*j+1] = 1.0
        r3[3*j+2] = 0.5
        r3[3*i+1] = -1.0
        r3[3*i+2] = 0.5
        rows.append((r3, 0.0))
        
        pair_rows[(i, j)] = rows
        
    # Run as many topologies as possible
    start_time = time.time()
    num_topologies_attempted = 0
    
    while time.time() < deadline - 5:
        # Randomly select a constraint for each pair
        A_current = [base_A]
        b_current = [base_b]
        
        for i, j in pairs:
            choice = rng.integers(0, 4)
            row, b_val = pair_rows[(i, j)][choice]
            A_current.append(row.reshape(1, -1))
            b_current.append(b_val)
            
        A_full = np.vstack(A_current)
        b_full = np.concatenate(b_current)
        
        try:
            res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
            if res.success:
                obj_val = -res.fun
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_x = res.x
        except Exception:
            pass
            
        num_topologies_attempted += 1
        # Optional: break if we've done enough
        if num_topologies_attempted > 50000:
            break

    if best_x is None:
        # Fallback
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            centers[i] = [0.1, 0.1]
            sides[i] = 0.05
        return centers, sides
    
    # Extract results
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        sides[i] = best_x[3*i+2]
        
    # Clip to ensure strict feasibility if numerical errors occurred
    tol = 1e-7
    for i in range(n):
        s = sides[i]
        centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
        centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
        
    return centers, sides

# EVOLVE-BLOCK-END
