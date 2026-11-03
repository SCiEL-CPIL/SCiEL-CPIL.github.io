import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    def evaluate_solution(xyz):
        """Compute the sum of radii, penalizing violations."""
        x = xyz[0:N]
        y = xyz[N:2*N]
        r = xyz[2*N:3*N]
        
        # Boundary violations
        boundary_violation = 0.0
        for i in range(N):
            if x[i] - r[i] < 0:
                boundary_violation += (x[i] - r[i])
            if x[i] + r[i] > 1:
                boundary_violation += (x[i] + r[i] - 1)
            if y[i] - r[i] < 0:
                boundary_violation += (y[i] - r[i])
            if y[i] + r[i] > 1:
                boundary_violation += (y[i] + r[i] - 1)
        
        # Pairwise overlap violations
        overlap_violation = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_violation += (min_dist - dist)
        
        total_violation = boundary_violation + overlap_violation
        sum_radii = np.sum(r)
        
        # Penalty function: we want to maximize sum_radii, minimize violation
        # Return negative (for minimization) with penalty
        penalty_weight = 100.0
        return -(sum_radii - penalty_weight * total_violation)
    
    def initial_guess(seed):
        """Generate an initial random configuration."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        # Start with small radii that might grow
        r = rng.uniform(0.02, 0.08, N)
        return np.concatenate([x, y, r])
    
    def bounds():
        """Define bounds for each variable."""
        b = []
        for i in range(N):
            b.append((0.0, 1.0))  # x_i
        for i in range(N):
            b.append((0.0, 1.0))  # y_i
        for i in range(N):
            b.append((0.001, 0.5))  # r_i (max radius is 0.5 for a single circle)
        return b
    
    bnds = bounds()
    
    best_solution = None
    best_score = -np.inf
    
    # Try multiple random restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        x0 = initial_guess(restart)
        
        try:
            result = minimize(
                evaluate_solution,
                x0,
                method='L-BFGS-B',
                bounds=bnds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success:
                xyz = result.x
                x = xyz[0:N]
                y = xyz[N:2*N]
                r = xyz[2*N:3*N]
                
                # Check if solution is valid (no violations)
                valid = True
                for i in range(N):
                    if x[i] - r[i] < -1e-6 or x[i] + r[i] > 1 + 1e-6:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-6 or y[i] + r[i] > 1 + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_solution = np.column_stack([x, y, r])
        except Exception:
            continue
    
    # If no valid solution found, fall back to a simple grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
    
    return best_solution
