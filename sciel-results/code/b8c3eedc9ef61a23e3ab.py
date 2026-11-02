"""
Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Multi-restart SLSQP optimization with explicit inequality constraints.
"""
import time
import numpy as np
from scipy.optimize import minimize
import random

N = 26
DEADLINE = 145.0  # Safety margin from 150s limit

def _validate_and_score(circles):
    """
    Validate circles and compute score.
    circles: array of shape (N, 3) [x, y, r]
    Returns: (is_valid, score)
    """
    n = len(circles)
    score = 0.0
    # Check boundaries and radii
    for i in range(n):
        x, y, r = circles[i]
        if r < -1e-9:
            return False, 0.0
        # Enforce small tolerance for boundaries
        if x < r - 1e-9 or x > 1 - r + 1e-9:
            return False, 0.0
        if y < r - 1e-9 or y > 1 - r + 1e-9:
            return False, 0.0
        score += max(0, r)
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            if dist_sq < sum_r**2 - 1e-9:
                return False, 0.0
    return True, score

def _initial_hex():
    """Generate a hexagonal lattice initialization."""
    # Try to fit 26 circles in a hex pattern
    # Approximate radius for 26 circles in unit square
    # Area per circle ~ 1/26. r ~ sqrt(1/(26*pi)) ~ 0.11
    # Hex spacing is 2*r * sqrt(3)/2 = r*sqrt(3)
    
    # We'll generate points on a hex grid and pick the first 26 that fit well
    # Or just create a structured grid that approximates hex
    points = []
    r_est = 0.1
    dx = 2 * r_est * 0.866
    dy = 2 * r_est
    
    y = r_est
    row = 0
    while len(points) < N and y < 1.0:
        x = r_est + (row % 2) * dx / 2
        while x < 1.0 and len(points) < N:
            # Check if inside
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y, r_est * 0.9])
            x += dx
        y += dy
        row += 1
    
    if len(points) < N:
        # Fallback to grid if hex didn't fill
        return _initial_grid()
        
    # Shuffle slightly to break symmetry
    random.shuffle(points)
    return np.array(points[:N])

def _initial_grid():
    """Generate a grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    points = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            points.append([x, y, r])
            idx += 1
    return np.array(points)

def _initial_corner_weighted():
    """Generate initialization with circles pushed towards corners."""
    points = []
    rng = np.random.default_rng(42)
    # Place some in corners
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for c in corners:
        points.append([c[0], c[1], 0.15])
    
    # Fill rest randomly but biased to edges
    while len(points) < N:
        x = rng.random()
        y = rng.random()
        # Bias towards edges
        if rng.random() < 0.5:
            x = 0.1 if rng.random() < 0.5 else 0.9
        else:
            y = 0.1 if rng.random() < 0.5 else 0.9
        r = 0.08 + 0.05 * rng.random()
        points.append([x, y, r])
    
    return np.array(points[:N])

def _slsqp_optimize(x0, time_limit):
    """
    Optimize circle packing using SLSQP.
    x: flattened array [x1, y1, r1, x2, y2, r2, ...]
    Objective: Maximize sum(r) -> Minimize -sum(r)
    Constraints:
    1. Boundaries: xi >= ri, xi <= 1-ri, yi >= ri, yi <= 1-ri
    2. Overlaps: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    """
    n = N
    
    def objective(x):
        # x is length 3*N
        # r is every 3rd element starting at index 2
        r = x[2::3]
        return -np.sum(r)
    
    def constraints_list():
        cons = []
        
        # Boundary constraints
        # For each circle i:
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        
        for i in range(n):
            xi_idx = 3*i
            yi_idx = 3*i + 1
            ri_idx = 3*i + 2
            
            # x_i >= r_i  => x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi_idx, ri=ri_idx: x[xi] - x[ri],
                'jac': lambda x, xi=xi_idx, ri=ri_idx: np.concatenate([
                    np.eye(3*n)[xi], -np.eye(3*n)[ri]
                ]) if False else (lambda x, xi=xi_idx, ri=ri_idx: np.concatenate([
                    np.eye(3*n)[xi], -np.eye(3*n)[ri]
                ])) # This lambda construction is inefficient, let's precompute or use simpler jac
            })
            # Actually, creating lambdas with closure in a loop is slow. 
            # Let's define a single jac function that returns the full gradient for all constraints? 
            # No, SLSQP expects separate fun/jac for each constraint or a dict.
            # Better approach: Use a single vectorized constraint function if possible, 
            # but SLSQP in scipy takes a list of dicts.
            # For N=26, 26*4 = 104 boundary constraints + 26*25/2 = 325 overlap constraints.
            # Total ~429 constraints. This might be slow for SLSQP.
            
        # Overlap constraints
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        
        # To improve performance, we will implement the jac explicitly for the whole vector?
        # No, SLSQP requires individual constraint definitions or we can use a penalty method 
        # if SLSQP is too slow. But the prompt asks for a structural change to explicit constraints.
        # Let's try to optimize the constraint definition.
        
        # Re-defining constraints efficiently:
        cons = []
        
        # Helper to get indices
        def get_idx(i, comp):
            return 3*i + comp
        
        # Boundary Constraints
        for i in range(n):
            # x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i] - x[3*i+2],
                'jac': lambda x, i=i: np.concatenate([
                    np.eye(3*n)[3*i], -np.eye(3*n)[3*i+2]
                ])
            })
            # 1 - x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2],
                'jac': lambda x, i=i: np.concatenate([
                    -np.eye(3*n)[3*i], -np.eye(3*n)[3*i+2]
                ])
            })
            # y_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2],
                'jac': lambda x, i=i: np.concatenate([
                    np.eye(3*n)[3*i+1], -np.eye(3*n)[3*i+2]
                ])
            })
            # 1 - y_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2],
                'jac': lambda x, i=i: np.concatenate([
                    -np.eye(3*n)[3*i+1], -np.eye(3*n)[3*i+2]
                ])
            })
            
        # Overlap Constraints
        for i in range(n):
            for j in range(i+1, n):
                # dist^2 - (ri+rj)^2 >= 0
                # f(x) = (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2
                # jac: 
                # d/dxi = 2(xi-xj)
                # d/dyi = 2(yi-yj)
                # d/dri = -2(ri+rj)
                # d/dxj = -2(xi-xj)
                # d/dyj = -2(yi-yj)
                # d/drj = -2(ri+rj)
                
                def fun_overlap(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                    
                def jac_overlap(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    grad = np.zeros(3*n)
                    grad[3*i] = 2*(xi-xj)
                    grad[3*i+1] = 2*(yi-yj)
                    grad[3*i+2] = -2*(ri+rj)
                    grad[3*j] = -2*(xi-xj)
                    grad[3*j+1] = -2*(yi-yj)
                    grad[3*j+2] = -2*(ri+rj)
                    return grad

                cons.append({
                    'type': 'ineq',
                    'fun': fun_overlap,
                    'jac': jac_overlap
                })
                
        return cons

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Start time
    t_start = time.time()
    
    # Run optimization
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints_list(),
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
    except Exception as e:
        # If SLSQP fails, return original
        return x0
        
    return res.x

def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    # Initializations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted()
    ]
    
    # Add some random inits
    rng = np.random.default_rng(123)
    for _ in range(3):
        x = rng.random((N, 2))
        r = 0.05 + 0.05 * rng.random(N)
        init = np.column_stack((x, r))
        inits.append(init)
        
    for init in inits:
        if time.time() > DEADLINE:
            break
            
        # Flatten init
        x0 = init.flatten()
        
        # Validate initial state? SLSQP will handle it, but if it's wildly invalid, it might fail.
        # Let's just run it.
        
        # Run SLSQP
        x_opt = _slsqp_optimize(x0, time.time())
        
        # Reshape
        circles = x_opt.reshape(N, 3)
        
        # Validate and Score
        valid, score = _validate_and_score(circles)
        
        # SLSQP might return slightly invalid points due to numerical precision.
        # If invalid, try to fix by shrinking radii slightly or rejecting.
        if not valid:
             # Try a simple repair: shrink all radii by 5%
             circles[:, 2] *= 0.95
             valid, score = _validate_and_score(circles)
             
        if valid and score > best_score:
            best_score = score
            best_circles = circles.copy()
            
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
