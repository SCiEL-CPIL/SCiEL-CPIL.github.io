"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    
    Uses a multi-start local optimization approach with gradient-based methods.
    The objective is to maximize sum(r_i) subject to:
      - Circles are inside the unit square: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - Circles are non-overlapping: ||p_i - p_j||_2 >= r_i + r_j for all i != j
    
    We use a penalty-based approach with scipy.optimize.minimize (L-BFGS-B) 
    to handle the non-convex constraints via smooth penalties.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initial best result
    best_value = -1
    best_solution = None
    
    def get_initial_guesses():
        """Generate diverse initial configurations."""
        guesses = []
        
        # 1. Grid-based initialization
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
        r0 = 0.5 / side - 0.01
        x0 = np.array([c[0] for c in cells[:N]])
        y0 = np.array([c[1] for c in cells[:N]])
        r0_arr = np.full(N, r0)
        guesses.append(np.concatenate([x0, y0, r0_arr]))
        
        # 2. Random initialization with varying radii
        for trial in range(20):
            rng = np.random.RandomState(trial * 137 + 42)
            x0 = rng.uniform(0.1, 0.9, N)
            y0 = rng.uniform(0.1, 0.9, N)
            # Vary radii: some larger, some smaller
            base_r = rng.uniform(0.03, 0.08)
            r0_arr = base_r * rng.uniform(0.7, 1.3, N)
            # Clip to valid range
            r0_arr = np.clip(r0_arr, 0.01, 0.15)
            guesses.append(np.concatenate([x0, y0, r0_arr]))
        
        # 3. Hexagonal-like packing initialization
        for offset in [0.0, 0.1, -0.1]:
            x0 = np.zeros(N)
            y0 = np.zeros(N)
            r0_arr = np.zeros(N)
            cols = 6
            rows = 5
            for k in range(N):
                col = k % cols
                row = k // cols
                x0[k] = 0.1 + col * 0.15 + (0.075 if row % 2 == 1 else 0) + offset
                y0[k] = 0.1 + row * 0.17
                r0_arr[k] = 0.07
            x0 = np.clip(x0, 0.05, 0.95)
            y0 = np.clip(y0, 0.05, 0.95)
            guesses.append(np.concatenate([x0, y0, r0_arr]))
        
        return guesses
    
    def objective_and_constraints(params):
        """
        Compute objective (negative sum of radii) and constraint violations.
        params: [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: minimize -sum(r)
        obj = -np.sum(r)
        
        # Constraint violations (we want these to be <= 0)
        # Boundary constraints: x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
        # i.e., x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
        # We'll use a smooth penalty approach instead of hard constraints
        
        # Pairwise distance constraints: ||p_i - p_j|| >= r_i + r_j
        # i.e., (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        violation = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                # Smooth violation: max(0, min_dist - dist)
                dist = np.sqrt(dist_sq + 1e-12)
                viol = min_dist - dist
                if viol > 0:
                    violation += viol * viol
        
        # Boundary violations
        for i in range(N):
            for val in [x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i]]:
                if val < 0:
                    violation += val * val
        
        return obj, violation
    
    def penalized_objective(params, penalty_weight=100.0):
        """
        Penalized objective for unconstrained optimization.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Clip radii to reasonable range for evaluation
        r_clipped = np.clip(r, 0.001, 0.2)
        
        obj = -np.sum(r_clipped)
        
        # Penalty for violations
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r_clipped[i] + r_clipped[j]
                viol = min_dist - dist
                if viol > 0:
                    penalty += viol * viol
        
        for i in range(N):
            for val in [x[i] - r_clipped[i], 1 - x[i] - r_clipped[i], 
                         y[i] - r_clipped[i], 1 - y[i] - r_clipped[i]]:
                if val < 0:
                    penalty += val * val
        
        return obj + penalty_weight * penalty
    
    def penalized_gradient(params, penalty_weight=100.0):
        """
        Numerical gradient of penalized objective.
        """
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = penalized_objective(params, penalty_weight)
        for k in range(len(params)):
            params_p = params.copy()
            params_p[k] += eps
            f_p = penalized_objective(params_p, penalty_weight)
            grad[k] = (f_p - f0) / eps
        return grad
    
    def evaluate_solution(params):
        """
        Evaluate if solution is valid and return sum of radii.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Check validity
        for i in range(N):
            if r[i] < 1e-8:
                return -1
            if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                return -1
            if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                return -1
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-6:
                    return -1
        
        return np.sum(r)
    
    # Try multiple initializations
    guesses = get_initial_guesses()
    
    for g_idx, guess in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.001, 0.999))  # x
        for i in range(N):
            bounds.append((0.001, 0.999))  # y
        for i in range(N):
            bounds.append((0.001, 0.2))    # r
        
        try:
            result = minimize(
                penalized_objective,
                guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                val = evaluate_solution(result.x)
                if val > best_value:
                    best_value = val
                    best_solution = result.x.copy()
                    
        except Exception as e:
            pass
    
    # If we still don't have a good solution, use the grid-based one
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N]).flatten()
        best_value = evaluate_solution(best_solution)
        if best_value < 0:
            # Fallback to trivial
            best_solution = np.array([c for c in cells[:N]]).flatten()
    
    # Format output: array of shape (N, 3) with rows (x, y, r)
    x = best_solution[:N]
    y = best_solution[N:2*N]
    r = best_solution[2*N:3*N]
    
    result = np.column_stack([x, y, r])
    
    # Final validation and clipping
    for i in range(N):
        r[i] = max(1e-8, r[i])
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    return result
# EVOLVE-BLOCK-END
