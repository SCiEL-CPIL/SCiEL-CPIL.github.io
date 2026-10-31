"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import LinearConstraint

N = 26

# EVOLVE-BLOCK-START
def _make_grid_initial(n, side):
    """Create a grid of circles with equal radii."""
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side
            cells.append((x, y, r))
    return np.array(cells[:n])


def _compute_objective(params, n):
    """Compute sum of radii from params (2n values: x_i, y_i for each circle)."""
    # params is flattened: [x0, y0, x1, y1, ..., xn-1, yn-1]
    # We need to infer radii from positions.
    # Actually, for this problem, we'll use a different approach:
    # Use scipy.optimize with variables being centers and radii.
    pass


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize sum of radii.
    Returns array of shape (26, 3): rows (x, y, radius).
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin from 120s limit
    
    # Initial configuration: 5x5 grid, use first 25, add 1 more
    # Better: use a 5x6 grid and pick 26
    side = 5
    # Create 5x5 grid = 25 circles, then add 26th
    # Let's use a different approach: start with equal circles in a grid
    
    # For 26 circles, we can use a 5x6 arrangement or similar
    # Let's try: 5 columns x 6 rows = 30 positions, use 26
    # Or better: use optimization
    
    # Simple approach: start with 5x5 grid (25 circles) + 1 extra
    # Each circle has radius 0.5/5 = 0.1
    # Sum = 25 * 0.1 + r_26
    
    # Let's use scipy to optimize
    from scipy.optimize import minimize
    
    n = 26
    
    # Initial guess: 5x5 grid + 1 extra
    # 5x5 grid: positions at (0.1, 0.3, 0.5, 0.7, 0.9) x (0.1, 0.3, 0.5, 0.7, 0.9)
    # That's 25 circles. Add 26th somewhere.
    
    # Actually, let's use a better initial configuration.
    # Use a hexagonal-like packing or just start with a grid.
    
    # Let's parameterize: for each circle i, variables are (x_i, y_i, r_i)
    # Total variables: 3 * 26 = 78
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    # Constraints:
    # 1. Each circle inside unit square: x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    # 2. Circles don't overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    # This is a non-linear optimization problem. Let's use SLSQP or trust-constr.
    
    # Initial guess:
    # 5x5 grid for first 25, then place 26th
    xs = []
    ys = []
    rs = []
    
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            xs.append((i + 0.5) / 5.0)
            ys.append((j + 0.5) / 5.0)
            rs.append(0.5 / 5.0)
    
    # 26th circle: place it in a gap or just add it
    # Let's place it at a position that doesn't overlap
    # Actually, let's just start with 26 circles in a slightly perturbed grid
    # Or use a 5x6 grid and drop 4
    
    # Better initial: use 26 circles in a 5x6 grid pattern
    # 5 columns, 6 rows = 30 positions, use 26
    # But that might not be optimal.
    
    # Let's just use the 5x5 grid + 1 extra for now
    # Place 26th at (0.5, 0.5) with small radius
    xs.append(0.5)
    ys.append(0.5)
    rs.append(0.05)
    
    x0 = np.array(xs + ys + rs)  # [x0, x1, ..., x25, y0, y1, ..., y25, r0, r1, ..., r25]
    
    # Actually, let's reorganize: x0, y0, r0, x1, y1, r1, ...
    x0 = np.array([xs[i] for i in range(n)] + [ys[i] for i in range(n)] + [rs[i] for i in range(n)])
    
    # Hmm, let's use a cleaner parameterization
    # Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    
    def unpack(params):
        """Unpack parameters into x, y, r arrays."""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return x, y, r
    
    def objective(params):
        """Negative sum of radii (to minimize)."""
        _, _, r = unpack(params)
        return -np.sum(r)
    
    def constraints(params):
        """Return list of constraint values (each >= 0 for feasibility)."""
        x, y, r = unpack(params)
        cons = []
        
        # Boundary constraints: x_i >= r_i, x_i <= 1 - r_i, etc.
        for i in range(n):
            cons.append(x[i] - r[i])       # x_i >= r_i
            cons.append((1 - r[i]) - x[i]) # x_i <= 1 - r_i
            cons.append(y[i] - r[i])       # y_i >= r_i
            cons.append((1 - r[i]) - y[i]) # y_i <= 1 - r_i
        
        # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                cons.append(dx*dx + dy*dy - (r[i] + r[j])**2)
        
        return cons
    
    # Use SLSQP
    # This might be slow for 78 variables. Let's try.
    
    # Actually, SLSQP might struggle with this. Let's try a different approach.
    # Use a simpler heuristic or a different optimizer.
    
    # Let's try using scipy.optimize.minimize with method='SLSQP'
    
    # But first, let's check if we have enough time
    if time.time() - start_time > deadline * 0.1:
        # Return initial grid
        return np.array(list(zip(xs, ys, rs)))
    
    try:
        # Scale the problem to make it easier for the optimizer
        # Normalize radii to be in [0, 0.5]
        
        # Let's try a simpler approach: use a greedy algorithm or simulated annealing
        # Or use a library like scipy.optimize with a better method
        
        # For now, let's just return a reasonable grid configuration
        # 5x5 grid with 25 circles + 1 extra
        # Sum = 25 * 0.1 + 0.05 = 2.55
        
        # Can we do better?
        # If we use 26 circles with average radius ~0.09, sum ~2.34
        # The 5x5 grid gives sum = 2.5 + 0.05 = 2.55
        
        # Let's try to optimize using a simple local search
        
        best_params = x0.copy()
        best_score = -objective(x0)
        
        # Simple local search
        step_size = 0.01
        n_iter = 10000
        
        for it in range(n_iter):
            if time.time() - start_time > deadline:
                break
            
            # Random perturbation
            idx = np.random.randint(0, 3 * n)
            delta = np.random.uniform(-step_size, step_size)
            
            new_params = best_params.copy()
            new_params[idx] += delta
            
            # Check feasibility
            x, y, r = unpack(new_params)
            feasible = True
            
            # Check boundaries
            for i in range(n):
                if r[i] < 0 or x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                    feasible = False
                    break
            
            if feasible:
                # Check overlaps
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        if dx*dx + dy*dy < (r[i] + r[j])**2 - 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        break
            
            if feasible:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_params = new_params
                    step_size *= 0.99  # Decrease step size
            else:
                step_size *= 1.01  # Increase step size slightly
        
        x, y, r = unpack(best_params)
        result = np.array(list(zip(x, y, r)))
        
        # Verify feasibility
        feasible = True
        for i in range(n):
            if r[i] < 0 or x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
                feasible = False
                break
        
        if feasible:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    if dx*dx + dy*dy < (r[i] + r[j])**2 - 1e-9:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if feasible:
            return result
        else:
            # Return initial grid
            return np.array(list(zip(xs, ys, rs)))
    
    except Exception as e:
        # Return initial grid on error
        return np.array(list(zip(xs, ys, rs)))
# EVOLVE-BLOCK-END
