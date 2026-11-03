"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def compute_radius(c):
        """Compute the valid radius for given centers."""
        # Pairwise distances
        n = len(c)
        min_pair = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_pair:
                    min_pair = d
        # Wall distances
        min_wall = np.inf
        for i in range(n):
            for k in range(2):
                d = 0.5 - abs(c[i, k])
                if d < min_wall:
                    min_wall = d
        return min(min_pair / 2.0, min_wall)
    
    def objective(x):
        """Objective: minimize -r (i.e., maximize r)."""
        c = x[:2*N].reshape(N, 2)
        r = x[2*N]
        # Compute actual feasible r
        min_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_pair:
                    min_pair = d
        min_wall = np.inf
        for i in range(N):
            for k in range(2):
                d = 0.5 - abs(c[i, k])
                if d < min_wall:
                    min_wall = d
        feasible_r = min(min_pair / 2.0, min_wall)
        # Penalize if r > feasible_r
        if r > feasible_r:
            return -(feasible_r) + 1000.0 * (r - feasible_r)
        return -r
    
    def constraints_eq(x):
        """Equality constraints: none."""
        return []
    
    def constraints_ineq(x):
        """Inequality constraints: r <= min_pair/2 and r <= min_wall for each coordinate."""
        c = x[:2*N].reshape(N, 2)
        r = x[2*N]
        cons = []
        # Pairwise: |c_i - c_j| >= 2r
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.linalg.norm(x[:2*N].reshape(N,2)[i] - x[:2*N].reshape(N,2)[j]) - 2*x[2*N]})
        # Wall: |c_ik| <= 0.5 - r for each coordinate
        for i in range(N):
            for k in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: 0.5 - abs(x[:2*N].reshape(N,2)[i, k]) - x[2*N]})
        return cons
    
    def make_constraints(c_init, r_init):
        """Create constraint functions that are faster using precomputed structure."""
        c = c_init.reshape(N, 2)
        r = r_init
        
        cons = []
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.linalg.norm(x[:2*N].reshape(N,2)[i] - x[:2*N].reshape(N,2)[j]) - 2*x[2*N]
                })
        # Wall constraints
        for i in range(N):
            for k in range(2):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, k=k: 0.5 - abs(x[:2*N].reshape(N,2)[i, k]) - x[2*N]
                })
        return cons
    
    # Generate initial configurations
    def grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    def random_init(r_guess):
        # Random points with minimum spacing
        pts = []
        attempts = 0
        while len(pts) < N and attempts < 10000:
            p = np.array([np.random.uniform(-0.5 + r_guess, 0.5 - r_guess),
                          np.random.uniform(-0.5 + r_guess, 0.5 - r_guess)])
            if all(np.linalg.norm(p - q) >= 2*r_guess*0.9 for q in pts):
                pts.append(p)
            attempts += 1
        while len(pts) < N:
            pts.append(np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)]))
        return np.array(pts)
    
    best_c = None
    best_r = 0.0
    
    # Try grid initialization first
    c_grid = grid_init()
    r_grid = compute_radius(c_grid)
    
    # Initial x for SLSQP
    x0 = np.concatenate([c_grid.flatten(), [r_grid * 0.99]])
    
    # Run SLSQP optimization
    try:
        cons = make_constraints(c_grid, r_grid)
        bounds = [(-0.5, 0.5)] * (2*N) + [(0.01, 0.25)]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < -best_r:
            c_opt = result.x[:2*N].reshape(N, 2)
            r_opt = compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_c is not None:
            return best_c
        return c_grid
    
    # Try multiple random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        r_guess = 0.06
        c_rand = random_init(r_guess)
        r_rand = compute_radius(c_rand)
        
        x0 = np.concatenate([c_rand.flatten(), [r_rand * 0.99]])
        
        try:
            cons = make_constraints(c_rand, r_rand)
            bounds = [(-0.5, 0.5)] * (2*N) + [(0.01, 0.25)]
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 150, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < -best_r:
                c_opt = result.x[:2*N].reshape(N, 2)
                r_opt = compute_radius(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
        except Exception:
            continue
    
    if best_c is not None:
        return best_c
    return c_grid
# EVOLVE-BLOCK-END
