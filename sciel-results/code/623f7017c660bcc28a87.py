"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s
    
    # We want to maximize sum of radii, subject to:
    # 1. Circles are inside a rectangle of perimeter 4: 2*(w+h) <= 4 => w+h <= 2
    # 2. Circles have disjoint interiors: dist(i,j) >= r_i + r_j
    # 3. Circles touch walls: 0 <= x_i <= w, 0 <= y_i <= h, r_i <= min(x_i, w-x_i, y_i, h-y_i)
    
    # Strategy: 
    # - Use a square rectangle w=h=1 for simplicity (perimeter = 4). This is often optimal or near-optimal.
    # - Optimize positions (x_i, y_i) and radii (r_i) to maximize sum(r_i)
    # - Constraints:
    #   * r_i >= 0
    #   * x_i - r_i >= 0, w - x_i - r_i >= 0, y_i - r_i >= 0, h - y_i - r_i >= 0
    #   * (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Since we have a square w=h=1, the constraints become:
    #   r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    #   => r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    # Let's use SLSQP with variables: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    w = 1.0
    h = 1.0
    
    # Initial guess: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    init_x = np.array([((i % side) + 0.5) / side for i in range(N)])
    init_y = np.array([((i // side) + 0.5) / side for i in range(N)])
    init_r = np.full(N, 0.5 / side - 0.01)
    
    # Flatten to optimization variable
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
    
    def unflatten(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r
    
    def objective(v):
        x, y, r = unflatten(v)
        return -np.sum(r)
    
    def constraints(v):
        x, y, r = unflatten(v)
        cons = []
        
        # Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N + i]})  # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*N + i]})  # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N + i] - v[2*N + i]})  # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[N + i] - v[2*N + i]})  # 1 - y_i - r_i >= 0
        
        # Circle-circle constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2
                })
        
        # Non-negativity of radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N + i]})
        
        return cons
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(N):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((0.0, 0.5))  # r_i (max radius in unit square is 0.5)
    
    v0 = flatten(init_x, init_y, init_r)
    
    best_result = None
    best_value = -np.inf
    
    # Try multiple initializations
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb initial guess
        if attempt == 0:
            v_init = v0.copy()
        else:
            v_init = v0 + np.random.randn(3*N) * 0.05
            # Clip to bounds
            for i in range(N):
                v_init[i] = np.clip(v_init[i], 0.01, 0.99)
                v_init[N+i] = np.clip(v_init[N+i], 0.01, 0.99)
                v_init[2*N+i] = np.clip(v_init[2*N+i], 0.01, 0.4)
        
        try:
            result = minimize(
                objective, v_init, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_value:
                x, y, r = unflatten(result.x)
                # Verify feasibility
                feasible = True
                for i in range(N):
                    if x[i] - r[i] < -1e-6 or 1 - x[i] - r[i] < -1e-6 or \
                       y[i] - r[i] < -1e-6 or 1 - y[i] - r[i] < -1e-6 or r[i] < -1e-6:
                        feasible = False
                        break
                    for j in range(i + 1, N):
                        dist2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                        if dist2 < (r[i] + r[j])**2 - 1e-6:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible:
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
                        best_result = result.x.copy()
        except Exception:
            continue
    
    if best_result is not None:
        x, y, r = unflatten(best_result)
        # Create output: [x, y, r] for each circle, scaled to perimeter-4 rectangle
        # Our rectangle is w=1, h=1, perimeter = 4. Good.
        circles = np.column_stack([x, y, r])
        return circles
    else:
        # Fallback: return initial grid
        circles = np.column_stack([init_x, init_y, init_r])
        return circles

# EVOLVE-BLOCK-END
