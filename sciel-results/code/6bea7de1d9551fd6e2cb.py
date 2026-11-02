"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Initial guess: grid placement
    side = 5
    # 5x5 grid = 25 circles, we need 26. 
    # Let's use a better initial heuristic.
    # Place 25 circles in a 5x5 grid, and 1 in the center? No, overlaps.
    # Let's just use a random or grid start and optimize.
    
    # Heuristic start: 
    # 26 circles. 
    # A 5x5 grid has 25 spots. 
    # Maybe 5x5 grid of radius 0.19, and one smaller? 
    # Or just random start with optimization.
    
    def initial_guess():
        # Create a 5x5 grid
        coords = []
        for i in range(5):
            for j in range(5):
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                coords.append([x, y])
        # Add 26th circle? 
        # The 5x5 grid is full. 
        # Let's try a different layout.
        # 26 circles is roughly sqrt(26) ~ 5.1.
        # Maybe 5 rows: 5, 5, 5, 5, 6?
        # Or just random initialization for the optimizer to handle.
        
        # Random initialization is often robust for global optimization if we have multiple restarts.
        # But let's try a structured one first.
        # 26 circles.
        # Let's place them in a grid but slightly jittered.
        rng = np.random.RandomState(42)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.1)
        return np.column_stack((x, y, r))

    def objective(params):
        # params: (N, 3) -> x, y, r
        # We want to MAXIMIZE sum(r)
        # So minimize -sum(r)
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Constraints:
        # 1. Inside square: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # 2. Disjoint: dist(i, j) >= r_i + r_j
        
        # Penalty method for constraints in objective
        # Or use scipy.optimize.minimize with constraints.
        
        # Let's use a penalty approach for simplicity and speed, or SLSQP with constraints.
        # SLSQP is good for this size.
        
        # Objective: -sum(r)
        val = -np.sum(r)
        
        # Penalties (soft)
        # Boundary violations
        pen = 0.0
        # x < r
        pen += np.sum(np.maximum(0, r - x)**2)
        # x > 1-r
        pen += np.sum(np.maximum(0, x - (1 - r))**2)
        # y < r
        pen += np.sum(np.maximum(0, r - y)**2)
        # y > 1-r
        pen += np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Overlaps
        # For each pair i, j
        # dist^2 < (r_i + r_j)^2
        # Penalty: max(0, (r_i + r_j)^2 - dist^2)^2
        
        # Vectorized overlap penalty
        # N=26, so 325 pairs.
        # dx = x[:, None] - x[None, :]
        # dy = y[:, None] - y[None, :]
        # dist2 = dx**2 + dy**2
        # r_sum = r[:, None] + r[None, :]
        # overlap = np.maximum(0, r_sum**2 - dist2)
        # pen += np.sum(overlap**2)
        
        # This is expensive if done every step in a loop, but N is small.
        # However, scipy minimize calls objective many times.
        # Let's use explicit constraints instead for better convergence.
        
        return val + 100 * pen # Scale penalty

    def jac(params):
        # Gradient of objective
        # This is complex with penalties. 
        # Let's rely on numerical gradient or analytic if simple.
        # Given the complexity, numerical is safer for correctness first.
        # But it's slow.
        # Let's stick to SLSQP with constraints.
        pass

    # Let's use SLSQP with constraints
    def optimize(start):
        x0 = start.flatten()
        
        # Constraints:
        # Inequality: g(x) >= 0
        # 1. x_i - r_i >= 0
        # 2. 1 - x_i - r_i >= 0
        # 3. y_i - r_i >= 0
        # 4. 1 - y_i - r_i >= 0
        # 5. dist_ij^2 - (r_i + r_j)^2 >= 0  (for all i<j)
        
        cons = []
        
        # Boundary constraints
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p: p[0::3] - p[2::3]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p: 1 - p[0::3] - p[2::3]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p: p[1::3] - p[2::3]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p: 1 - p[1::3] - p[2::3]})
        
        # Overlap constraints
        # This is O(N^2) constraints. 325 constraints.
        # SLSQP might struggle with many constraints.
        # Alternative: Penalty method is often faster for this N.
        
        # Let's switch to Penalty Method with BFGS/L-BFGS-B.
        # It's more robust to many constraints if penalty is tuned.
        
        def pen_objective(p):
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            obj = -np.sum(r)
            
            # Boundary
            pen = 0
            pen += np.sum(np.maximum(0, r - x)**2)
            pen += np.sum(np.maximum(0, x + r - 1)**2)
            pen += np.sum(np.maximum(0, r - y)**2)
            pen += np.sum(np.maximum(0, y + r - 1)**2)
            
            # Overlap
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist2 = dx**2 + dy**2
            r_sum = r[:, None] + r[None, :]
            # Only upper triangle
            iu = np.triu_indices(N, k=1)
            overlap = np.maximum(0, r_sum[iu]**2 - dist2[iu])
            pen += np.sum(overlap**2)
            
            return obj + 10 * pen
            
        def pen_jac(p):
            # Numerical gradient is too slow?
            # Analytic gradient for penalty:
            # d/dx (max(0, r-x)^2) = -2(r-x) if r>x else 0
            # d/dx (max(0, x+r-1)^2) = 2(x+r-1) if x+r>1 else 0
            # d/dr (max(0, r-x)^2) = 2(r-x) if r>x else 0
            # d/dr (max(0, x+r-1)^2) = 2(x+r-1) if x+r>1 else 0
            
            # Overlap:
            # f = max(0, (ri+rj)^2 - dist2)^2
            # Let u = (ri+rj)^2 - dist2
            # if u > 0:
            #   df/du = 2u
            #   du/dri = 2(ri+rj)
            #   du/dxi = -2(xi - xj)
            #   etc.
            
            # This is implementable but tedious.
            # Let's use finite difference for now, or just no jac (scipy will use numerical).
            # For N=26, 78 variables. Numerical jac is 79 evaluations.
            # Each evaluation is O(N^2) ~ 676 ops.
            # 79 * 676 ~ 50k ops per gradient step.
            # BFGS takes maybe 50-100 steps.
            # 5M ops total. Very fast.
            return None # Let scipy compute numerically

        res = minimize(pen_objective, x0, method='L-BFGS-B', jac=pen_jac, options={'maxiter': 500})
        return res.x.reshape(N, 3)

    # Multiple restarts to escape local minima
    best_val = -np.inf
    best_params = None
    
    # Initial guesses
    guesses = []
    
    # 1. Random
    for seed in range(5):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.15)
        guesses.append(np.column_stack((x, y, r)))
        
    # 2. Grid-like
    # 5x5 grid
    x_grid, y_grid = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    r_flat = np.full(25, 0.18)
    # Add 26th
    x_flat = np.append(x_flat, 0.5)
    y_flat = np.append(y_flat, 0.5)
    r_flat = np.append(r_flat, 0.1)
    guesses.append(np.column_stack((x_flat, y_flat, r_flat)))
    
    # 3. Hexagonal packing approximation
    # Hex packing density ~ 0.9069
    # Area 1, 26 circles.
    # r ~ sqrt(0.9069 / (26 * pi)) ~ sqrt(0.0111) ~ 0.105
    # Let's try hex grid.
    # Rows: 1, 2, 1, 2... or similar.
    # 26 circles.
    # 1.5 rows of 16? No.
    # 5 rows: 5, 6, 5, 6, 4?
    # Let's just stick to random and grid.
    
    for g in guesses:
        if time.time() > deadline:
            break
        try:
            opt_params = optimize(g)
            # Check validity
            x = opt_params[:, 0]
            y = opt_params[:, 1]
            r = opt_params[:, 2]
            
            # Validity check
            valid = True
            if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6):
                valid = False
            if np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
                valid = False
            
            if valid:
                # Check overlaps
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = r[:, None] + r[None, :]
                # dist >= r_sum
                if np.all(dist >= r_sum - 1e-6):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_params = opt_params
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_params

# EVOLVE-BLOCK-END
