"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def is_valid(circles, tol=1e-9):
        """Check if circles are inside unit square and non-overlapping."""
        for x, y, r in circles:
            if r < -tol:
                return False
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True
    
    def objective(params):
        """Negative sum of radii for minimization."""
        circles = params_to_circles(params)
        return -sum(c[2] for c in circles)
    
    def params_to_circles(params):
        """Convert flat parameter array to list of (x, y, r)."""
        circles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            circles.append((x, y, r))
        return circles
    
    def circles_to_params(circles):
        """Convert list of (x, y, r) to flat parameter array."""
        params = np.zeros(3 * N)
        for i, (x, y, r) in enumerate(circles):
            params[3*i] = x
            params[3*i + 1] = y
            params[3*i + 2] = r
        return params
    
    def constraints_for(circles):
        """Generate constraint functions for SLSQP."""
        cons = []
        for i in range(N):
            x, y, r = circles[i]
            # Circle inside unit square
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([1, 0, -1])})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([-1, 0, -1])})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([0, 1, -1])})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([0, -1, -1])})
            # Radius non-negative
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2], 'jac': lambda p, i=i: np.array([0, 0, 1])})
        
        # Non-overlapping constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2,
                    'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
                })
        return cons
    
    def _jac_overlap(p, i, j):
        """Jacobian for overlap constraint between circles i and j."""
        jac = np.zeros(3 * N)
        xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
        xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
        
        # d/dxi: 2(xi - xj)
        jac[3*i] = 2 * (xi - xj)
        # d/dxj: -2(xi - xj)
        jac[3*j] = -2 * (xi - xj)
        # d/dyi: 2(yi - yj)
        jac[3*i+1] = 2 * (yi - yj)
        # d/dyj: -2(yi - yj)
        jac[3*j+1] = -2 * (yi - yj)
        # d/dri: -2(ri + rj)
        jac[3*i+2] = -2 * (ri + rj)
        # d/drj: -2(ri + rj)
        jac[3*j+2] = -2 * (ri + rj)
        return jac
    
    def initial_grid():
        """Generate initial grid-based placement."""
        side = int(np.ceil(np.sqrt(N)))
        circles = []
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            circles.append((x, y, r))
        return circles
    
    def random_init():
        """Generate random initial placement."""
        circles = []
        for _ in range(N):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.01, 0.1)
            circles.append((x, y, r))
        return circles
    
    def local_search(circles, max_iters=50):
        """Local search by perturbing one circle at a time."""
        current = list(circles)
        current_sum = sum(c[2] for c in current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                x, y, r = current[i]
                # Try small perturbations
                for dx, dy, dr in [(0.01, 0, 0), (-0.01, 0, 0), (0, 0.01, 0), (0, -0.01, 0), 
                                    (0.005, 0.005, 0), (-0.005, -0.005, 0), 
                                    (0, 0, 0.005), (0, 0, -0.005)]:
                    new_x = x + dx
                    new_y = y + dy
                    new_r = max(0.001, r + dr)
                    # Check bounds
                    if new_x - new_r < -1e-9 or new_x + new_r > 1 + 1e-9:
                        continue
                    if new_y - new_r < -1e-9 or new_y + new_r > 1 + 1e-9:
                        continue
                    # Check overlaps
                    valid = True
                    for j in range(N):
                        if j == i:
                            continue
                        xj, yj, rj = current[j]
                        dist = np.sqrt((new_x - xj)**2 + (new_y - yj)**2)
                        if dist < new_r + rj - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_sum = current_sum - r + new_r
                    if new_sum > current_sum + 1e-12:
                        current[i] = (new_x, new_y, new_r)
                        current_sum = new_sum
                        improved = True
                        break
            if not improved:
                break
        return current
    
    # Start with grid initialization
    best_circles = initial_grid()
    best_sum = sum(c[2] for c in best_circles)
    
    # Try SLSQP optimization from grid
    try:
        params = circles_to_params(best_circles)
        cons = constraints_for(best_circles)
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        
        result = minimize(
            objective,
            params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if result.success and is_valid(circles_to_circles(result.x) if False else params_to_circles(result.x)):
            new_circles = params_to_circles(result.x)
            new_sum = sum(c[2] for c in new_circles)
            if new_sum > best_sum:
                best_circles = new_circles
                best_sum = new_sum
    except Exception as e:
        pass
    
    # Local search refinement
    if time.time() < deadline - 10.0:
        refined = local_search(best_circles, max_iters=100)
        refined_sum = sum(c[2] for c in refined)
        if refined_sum > best_sum and is_valid(refined):
            best_circles = refined
            best_sum = refined_sum
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 15.0:
            break
        
        circles = random_init()
        # Quick local search
        circles = local_search(circles, max_iters=30)
        circles_sum = sum(c[2] for c in circles)
        
        if circles_sum > best_sum:
            # Try SLSQP from this better starting point
            try:
                params = circles_to_params(circles)
                cons = constraints_for(circles)
                bounds = [(0, 1), (0, 1), (0, 0.5)] * N
                
                result = minimize(
                    objective,
                    params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 50, 'ftol': 1e-10}
                )
                
                if result.success:
                    new_circles = params_to_circles(result.x)
                    if is_valid(new_circles):
                        new_sum = sum(c[2] for c in new_circles)
                        if new_sum > best_sum:
                            best_circles = new_circles
                            best_sum = new_sum
            except Exception:
                pass
        else:
            # Still might be worth trying SLSQP if close
            if circles_sum > best_sum * 0.9:
                try:
                    params = circles_to_params(circles)
                    cons = constraints_for(circles)
                    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
                    
                    result = minimize(
                        objective,
                        params,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 50, 'ftol': 1e-10}
                    )
                    
                    if result.success:
                        new_circles = params_to_circles(result.x)
                        if is_valid(new_circles):
                            new_sum = sum(c[2] for c in new_circles)
                            if new_sum > best_sum:
                                best_circles = new_circles
                                best_sum = new_sum
                except Exception:
                    pass
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        refined = local_search(best_circles, max_iters=50)
        refined_sum = sum(c[2] for c in refined)
        if refined_sum > best_sum and is_valid(refined):
            best_circles = refined
    
    # Return as numpy array
    return np.array(best_circles)
# EVOLVE-BLOCK-END
