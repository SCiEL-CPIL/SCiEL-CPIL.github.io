"""
Circle packing in unit square.
Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
"""

import numpy as np
from scipy.optimize import minimize
import time
import random

N = 26
START_TIME = time.time()
TIME_LIMIT = 120.0 # Seconds
SAFETY_MARGIN = 10.0 # Seconds to ensure we return before kill

def _get_time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _compute_overlap_and_boundary_violations(x, y, r):
    """
    Compute total violation magnitude.
    Returns (overlap_violation, boundary_violation)
    """
    # Boundary violations
    # x < r, x > 1-r, y < r, y > 1-r, r < 0
    b_viol = 0.0
    b_viol += np.sum(np.maximum(0, r - x)**2)
    b_viol += np.sum(np.maximum(0, r - (1 - x))**2)
    b_viol += np.sum(np.maximum(0, r - y)**2)
    b_viol += np.sum(np.maximum(0, r - (1 - y))**2)
    b_viol += np.sum(np.maximum(0, -r)**2)
    
    # Overlap violations
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d = np.sqrt(dx**2 + dy**2)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    viol = np.maximum(0, r_sum - d)
    np.fill_diagonal(viol, 0)
    
    o_viol = np.sum(viol**2)
    
    return o_viol, b_viol

def _is_valid(x, y, r, tol=1e-5):
    """
    Check if the configuration is valid (no overlaps, inside square).
    """
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d = np.sqrt(dx**2 + dy**2)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Check for overlaps
    if np.any(r_sum > d + tol):
        return False
        
    return True

def _polish_solution(x, y, r):
    """
    Run a local optimizer to refine the solution.
    Uses SLSQP with constraints for better precision.
    """
    params = np.column_stack((x, y, r))
    
    # Objective: Maximize sum of radii -> Minimize -sum(r)
    def objective(p):
        x_p = p[:N]
        y_p = p[N:2*N]
        r_p = p[2*N:3*N]
        return -np.sum(r_p)
    
    # Constraints:
    # 1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # 2. Overlap: dist(i,j) - r_i - r_j >= 0
    
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
    # 1 - x_i - r_i >= 0
    constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*N+i]})
    # y_i - r_i >= 0
    constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
    # 1 - y_i - r_i >= 0
    constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[N+i] - p[2*N+i]})
    
    # Overlap constraints for all pairs i < j
    for i in range(N):
        for j in range(i+1, N):
            def overlap_constraint(p, i=i, j=j):
                dx = p[i] - p[j]
                dy = p[N+i] - p[N+j]
                dist = np.sqrt(dx**2 + dy**2)
                return dist - p[2*N+i] - p[2*N+j]
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r (max radius in unit square for 1 circle is 0.5, but with 26 it's much smaller)
        
    try:
        res = minimize(
            objective, 
            params, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        if res.success or res.status == 0:
            new_params = res.x
            new_x = new_params[:N]
            new_y = new_params[N:2*N]
            new_r = new_params[2*N:3*N]
            # Ensure validity after polishing
            if _is_valid(new_x, new_y, new_r):
                return new_x, new_y, new_r
    except Exception:
        pass
        
    return x, y, r

def _simulate_annealing(x, y, r, t_start, t_end, steps, step_size):
    """
    Perform Simulated Annealing.
    """
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = -np.sum(r) # We minimize -sum(r)
    
    current_score = -np.sum(r)
    
    # Pre-compute indices for pairs to speed up overlap check
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
            
    for step in range(steps):
        # Check time
        if _get_time_remaining() < 5.0:
            break
            
        # Temperature decay (geometric)
        t = t_start * (t_end / t_start) ** (step / steps)
        
        # Generate neighbor
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        # Perturb one or two circles
        # Move a random circle
        idx = np.random.randint(N)
        new_x[idx] += np.random.normal(0, step_size)
        new_y[idx] += np.random.normal(0, step_size)
        
        # Occasionally adjust radius
        if np.random.random() < 0.3:
            new_r[idx] += np.random.normal(0, step_size * 0.5)
            # Clamp radius
            new_r[idx] = np.clip(new_r[idx], 0, 0.5)
            
        # Clamp positions
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        
        # Calculate new score (penalized)
        # For SA, we use a raw violation metric to define "energy"
        # Energy = -sum(r) + penalty * (violations)
        # We want to minimize Energy.
        
        o_viol, b_viol = _compute_overlap_and_boundary_violations(new_x, new_y, new_r)
        penalty_factor = 100.0 # High penalty to encourage validity
        new_energy = -np.sum(new_r) + penalty_factor * (o_viol + b_viol)
        
        delta_e = new_energy - current_score
        
        if delta_e < 0:
            # Better solution
            x, y, r = new_x, new_y, new_r
            current_score = new_energy
            if -np.sum(r) > best_score and _is_valid(x, y, r):
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                best_score = -np.sum(r)
        else:
            # Worse solution, accept with probability
            if t > 0:
                prob = np.exp(-delta_e / t)
                if np.random.random() < prob:
                    x, y, r = new_x, new_y, new_r
                    current_score = new_energy
                    # Update best only if valid
                    if _is_valid(x, y, r) and -np.sum(r) > best_score:
                        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                        best_score = -np.sum(r)

    return best_x, best_y, best_r

def _generate_initial_guesses():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                                  np.linspace(0.5/side, 1-0.5/side, side))
    grid_x = grid_x.flatten()[:N]
    grid_y = grid_y.flatten()[:N]
    r_init = np.full(N, 0.5/side * 0.9)
    inits.append((grid_x, grid_y, r_init))
    
    # 2. Hexagonal
    hex_pts = []
    row = 0
    spacing = 1.0 / side
    while len(hex_pts) < N:
        for col in range(side + 1):
            x = (col + (0.5 if row % 2 else 0)) * spacing
            y = row * spacing * 0.866
            if 0 <= x <= 1 and 0 <= y <= 1:
                hex_pts.append((x, y))
        row += 1
    if len(hex_pts) >= N:
        hex_pts = hex_pts[:N]
        hx = np.array([p[0] for p in hex_pts])
        hy = np.array([p[1] for p in hex_pts])
        hr = np.full(N, spacing * 0.4)
        inits.append((hx, hy, hr))
        
    # 3. Random
    for _ in range(5):
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        rr = np.full(N, 0.05)
        inits.append((rx, ry, rr))
        
    # 4. Corner/Edge biased (often helps for packing)
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    cx = np.array([c[0] for c in corners] + list(np.random.rand(N-4)))
    cy = np.array([c[1] for c in corners] + list(np.random.rand(N-4)))
    cr = np.full(N, 0.08)
    inits.append((cx, cy, cr))
    
    return inits

def construct():
    """
    Returns an array of shape (N, 3) with rows (x, y, radius).
    """
    best_solution = None
    best_score = -np.inf
    
    inits = _generate_initial_guesses()
    
    # Shuffle inits to vary order
    random.shuffle(inits)
    
    for init in inits:
        if _get_time_remaining() < 10.0:
            break
            
        x, y, r = init
        
        # 1. Initial Polish
        x, y, r = _polish_solution(x, y, r)
        
        # 2. Simulated Annealing
        # Parameters for SA
        t_start = 1.0
        t_end = 1e-4
        # Estimate steps based on time remaining
        # Each step is fast, but we have overhead. Let's run a fixed reasonable number.
        sa_steps = 2000 
        step_size = 0.1
        
        # If we have plenty of time, do more SA
        if _get_time_remaining() > 60.0:
            sa_steps = 5000
            step_size = 0.05
            
        x, y, r = _simulate_annealing(x, y, r, t_start, t_end, sa_steps, step_size)
        
        # 3. Final Polish
        x, y, r = _polish_solution(x, y, r)
        
        # Check validity and update best
        if _is_valid(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_solution = np.column_stack((x, y, r))
                
    # Fallback
    if best_solution is None:
        # Safe grid
        side = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                                      np.linspace(0.5/side, 1-0.5/side, side))
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        r_init = np.full(N, 0.5/side * 0.5) # Very safe radius
        best_solution = np.column_stack((grid_x, grid_y, r_init))
        
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print("Sum of radii:", np.sum(res[:, 2]))
    print("Valid:", _is_valid(res[:, 0], res[:, 1], res[:, 2]))

