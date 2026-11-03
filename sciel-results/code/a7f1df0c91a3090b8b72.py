"""
Strategy: Multi-Start SLSQP for Maximum Equal-Radius Circle Packing
This approach reformulates the problem as a Non-Linear Programming (NLP) problem.
We assume all circles have the same radius r (which is often optimal for symmetric
containers like squares) and maximize r subject to non-overlap and boundary constraints.
We use scipy.optimize.minimize with the SLSQP method, which is effective for
constrained non-linear optimization. To avoid local minima, we perform multiple
random restarts with different initial configurations. The rectangle aspect ratio
is fixed to 1.0 (square) initially, as it is the most efficient shape for packing
circular objects under a perimeter constraint. If time permits, other aspect ratios
can be explored, but the square is the primary candidate for optimality.
"""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def _solve_aspect(w, h, num_restarts=10, time_limit=120.0):
    """
    Solve the packing problem for a given rectangle width w and height h.
    Returns the best valid solution (positions, radii) found.
    """
    start_time = time.time()
    best_r = -np.inf
    best_pos = None
    
    # Variables: x_1, y_1, ..., x_N, y_N, r
    # Total variables: 2*N + 1
    
    def objective(vars):
        # vars: [x1, y1, ..., xN, yN, r]
        r = vars[-1]
        # We want to maximize r, so minimize -r
        return -r

    def constraints(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[-1]
        
        cons = []
        
        # Wall constraints:
        # x_i - r >= 0  =>  x_i - r
        # w - x_i - r >= 0 => w - x_i - r
        # y_i - r >= 0  =>  y_i - r
        # h - y_i - r >= 0 => h - y_i - r
        
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[-1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[i] - v[-1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[-1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[N+i] - v[-1]})
            
        # Pairwise distance constraints:
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (2*r)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (2*v[-1])**2
                })
                
        return cons

    # Initial guesses
    # We try a few different initialization strategies
    for restart in range(num_restarts):
        if time.time() > time_limit - 5.0:
            break
            
        # Strategy 1: Grid + Noise
        # Strategy 2: Random
        if restart % 2 == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) * (w / side)
                y[i] = (row + 0.5) * (h / side)
            # Add small noise
            x += np.random.uniform(-0.05, 0.05, N)
            y += np.random.uniform(-0.05, 0.05, N)
            # Clip to valid range
            x = np.clip(x, 0.1, w - 0.1)
            y = np.clip(y, 0.1, h - 0.1)
            r0 = 0.05
        else:
            # Random initialization
            x = np.random.uniform(0.2, w - 0.2, N)
            y = np.random.uniform(0.2, h - 0.2, N)
            r0 = 0.05
            
        x0 = np.concatenate([x, y, [r0]])
        
        # Bounds for variables
        # x_i in [0, w], y_i in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
        bounds.append((0, min(w, h) / 2))
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                r_opt = res.x[-1]
                pos_opt = np.column_stack([res.x[:N], res.x[N:2*N]])
                radii_opt = np.full(N, r_opt)
                
                # Validate strictly
                if _is_valid(pos_opt, radii_opt, w, h, tol=1e-5):
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pos = pos_opt
                        best_radii = radii_opt
        except Exception:
            pass

    if best_pos is not None:
        return np.column_stack([best_pos, best_radii])
    return None

def _is_valid(pos, radii, w, h, tol=1e-5):
    """Check if the configuration is valid."""
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_value = -np.inf
    
    # Primary candidate: Square (w=1, h=1)
    # This is generally the optimal aspect ratio for packing circles
    w, h = 1.0, 1.0
    
    # Run multi-start SLSQP on the square
    # We allocate most of the time to the square
    time_alloc_square = 100.0
    res_square = _solve_aspect(w, h, num_restarts=20, time_limit=start_time + time_alloc_square)
    
    if res_square is not None:
        val = np.sum(res_square[:, 2])
        if val > best_value:
            best_value = val
            best_result = res_square
            
    # If time remains, try a few other aspect ratios
    if time.time() < deadline - 20.0:
        other_aspects = [1.2, 0.8, 1.5, 0.66]
        time_per_aspect = 10.0
        for aspect in other_aspects:
            if time.time() > deadline - 5.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 - w
            res = _solve_aspect(w, h, num_restarts=3, time_limit=time.time() + time_per_aspect)
            if res is not None:
                val = np.sum(res[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = res

    # Fallback if no valid solution found (should not happen with SLSQP and reasonable bounds)
    if best_result is None:
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])

    return best_result
