import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0 # seconds

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_params = None
    
    # Objective function
    def objective(params):
        # params shape: (3N,) -> x[0:N], y[N:2N], r[2N:3N]
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Sum of radii to maximize
        sum_r = np.sum(r)
        
        # Pairwise constraints
        # d_ij >= r_i + r_j
        # Violation v_ij = max(0, r_i + r_j - d_ij)
        # We want to minimize penalty of violations
        
        # Compute all pairwise distances
        # Using broadcasting for efficiency
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        # Violations
        v = np.maximum(0, r_sum - d)
        
        # Smooth penalty using log-sum-exp / softplus
        # S(v) = (1/beta) * log(1 + exp(beta * v))
        beta = 20.0
        # Avoid overflow in exp
        # v is usually small or 0. 
        # If v is large, exp(beta*v) can overflow.
        # Use np.logaddexp(0, beta * v) / beta
        penalty = np.sum(np.logaddexp(0, beta * v)) / beta
        
        return -sum_r + penalty

    def gradient(params):
        # Numerical gradient or analytical?
        # L-BFGS-B can use numerical if not provided, but analytical is faster.
        # However, the smooth penalty derivative is tricky.
        # Let's rely on numerical gradient initially for simplicity and robustness, 
        # or implement a simple finite difference if speed is an issue.
        # Given N=26, 3N=78 variables. Numerical gradient is 79 evaluations.
        # Each evaluation is O(N^2). 78*78*26*26 ~ 400k ops. Very fast.
        # So numerical gradient is fine.
        return None # Let L-BFGS-B handle it

    def is_valid(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Check bounds
        if np.any(x < 1e-6) or np.any(x > 1 - 1e-6): return False
        if np.any(y < 1e-6) or np.any(y > 1 - 1e-6): return False
        if np.any(r < 1e-6): return False
        
        # Check overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # Check if any distance < sum of radii + tolerance
        # Strictly disjoint interiors: d > r_i + r_j
        # Tolerance 1e-5
        if np.any(d < r_sum - 1e-5):
            return False
            
        return True

    def get_score(params):
        if not is_valid(params):
            return -np.inf
        r = params[2*N:3*N]
        return np.sum(r)

    def hex_lattice_init(perturb=0.0):
        # Generate a hexagonal lattice
        # Approximate size to fit 26 circles
        # Hex grid: 
        # Row 0: 5 circles
        # Row 1: 6 circles
        # Row 2: 5 circles
        # Row 3: 6 circles
        # Row 4: 4 circles
        # Total: 26
        
        points = []
        # Hex spacing
        # Let's just generate a dense hex grid and pick the first 26
        # Side length of hex unit ~ 0.2
        # Let's use a standard hex grid generation
        
        # Simpler: Grid with offset
        # 5x5 grid is 25. Add one in the middle?
        # Let's try a 5x5 grid with one extra in a gap.
        # Or just generate a 6x5 grid and take 26.
        
        # Let's use a hexagonal arrangement
        # x = i * dx + (j % 2) * dx/2
        # y = j * dy
        # dx = 2 * r, dy = sqrt(3) * r
        
        # We don't know r yet. Let's assume r ~ 0.1
        # 1/0.1 = 10 units. 
        # Let's just place 26 points in a hex-like pattern in [0.1, 0.9]
        
        # Generate points
        pts = []
        # 5 rows
        # 5, 6, 5, 6, 4
        counts = [5, 6, 5, 6, 4]
        y_coords = np.linspace(0.15, 0.85, 5)
        
        for j, count in enumerate(counts):
            x_coords = np.linspace(0.1, 0.9, count)
            # Offset odd rows
            if j % 2 == 1:
                x_coords = x_coords + 0.05 # Shift right
                # Clip to [0.1, 0.9] roughly
                x_coords = np.clip(x_coords, 0.1, 0.9)
            for x in x_coords:
                pts.append((x, y_coords[j]))
                
        # Take first 26
        pts = pts[:26]
        
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = np.full(26, 0.1) # Initial radius
        
        if perturb > 0:
            x += np.random.normal(0, perturb, 26)
            y += np.random.normal(0, perturb, 26)
            r += np.random.normal(0, perturb*0.1, 26)
            
        # Clip to bounds
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        r = np.clip(r, 0.01, 0.4)
        
        params = np.concatenate([x, y, r])
        return params

    def local_de_polish(params, generations=5, pop_size=10):
        # Simple local DE
        best_p = params.copy()
        best_f = objective(best_p)
        
        # Initialize population around best_p
        pop = [best_p.copy() for _ in range(pop_size)]
        for i in range(1, pop_size):
            pop[i] = best_p + np.random.normal(0, 0.01, 3*N)
            # Clip to valid bounds roughly
            pop[i][0:N] = np.clip(pop[i][0:N], 0, 1)
            pop[i][N:2*N] = np.clip(pop[i][N:2*N], 0, 1)
            pop[i][2*N:3*N] = np.clip(pop[i][2*N:3*N], 0, 0.5)
            
        for gen in range(generations):
            new_pop = []
            for i in range(pop_size):
                # Mutation
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                # Ensure they are not i
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    
                v = pop[r1] + 0.5 * (pop[r2] - pop[r3])
                
                # Crossover
                crossover_rate = 0.5
                trial = np.where(np.random.random(3*N) < crossover_rate, v, pop[i])
                
                # Bounds
                trial[0:N] = np.clip(trial[0:N], 0, 1)
                trial[N:2*N] = np.clip(trial[N:2*N], 0, 1)
                trial[2*N:3*N] = np.clip(trial[2*N:3*N], 0, 0.5)
                
                new_pop.append(trial)
            
            # Evaluate
            f_new = [objective(p) for p in new_pop]
            
            # Selection
            for i in range(pop_size):
                if f_new[i] < objective(pop[i]):
                    pop[i] = new_pop[i]
            
            # Update best
            f_best = objective(pop[0])
            if f_best < best_f:
                best_f = f_best
                best_p = pop[0].copy()
                
        return best_p

    # Main Loop
    while time.time() - start_time < DEADLINE:
        # Generate initial guess
        perturb = 0.05
        init_params = hex_lattice_init(perturb)
        
        # L-BFGS-B Optimization
        bounds = [
            (0, 1) for _ in range(N) # x
        ] + [
            (0, 1) for _ in range(N) # y
        ] + [
            (0.01, 0.4) for _ in range(N) # r
        ]
        
        try:
            res = minimize(
                objective, 
                init_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Polish with local DE
            polished_params = local_de_polish(res.x)
            
            # Check validity and score
            score = get_score(polished_params)
            if score > best_score:
                best_score = score
                best_params = polished_params.copy()
                
            # Also check the L-BFGS-B result directly in case DE messed it up
            score_lbfgs = get_score(res.x)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_params = res.x.copy()
                
        except Exception as e:
            pass

    # Return best result
    if best_params is None:
        # Fallback to trivial seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    return np.column_stack((x, y, r))
