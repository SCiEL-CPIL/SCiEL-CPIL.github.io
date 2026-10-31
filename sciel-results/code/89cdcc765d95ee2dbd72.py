"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _initial_guess():
    """Generate an initial guess for 21 circles in a unit square (perimeter 4)."""
    # 5x5 grid with 21 circles (last 4 rows have 5 circles, last row has 1)
    # Actually, let's just use a 5x5 grid and take the first 21
    positions = []
    for i in range(5):
        for j in range(5):
            if len(positions) < 21:
                positions.append(((i + 0.5) / 5.0, (j + 0.5) / 5.0, 0.1))
    
    radii = np.array([0.1] * 21)
    centers = np.array([[p[0], p[1]] for p in positions])
    
    return centers, radii

def _compute_objective(params, n_circles):
    """
    params: array of shape (3*n_circles,)
            [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_n, y_n, r_n]
    Returns: negative sum of radii (to be minimized)
    """
    centers = params[0::3]
    ys = params[1::3]
    radii = params[2::3]
    
    # Objective: maximize sum of radii
    obj = -np.sum(radii)
    
    return obj

def _constraints(params, n_circles):
    """
    Constraints:
    1. All circles must be within [0,1] x [0,1]
    2. All circles must not overlap
    """
    centers_x = params[0::3]
    centers_y = params[1::3]
    radii = params[2::3]
    
    constraints = []
    
    # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n_circles):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})  # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})  # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})  # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})  # 1 - y_i - r_i >= 0
    
    # Non-overlapping constraints
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })
    
    return constraints

def _compute_best_result(centers, radii):
    """Compute the best result from centers and radii."""
    # Check validity
    valid = True
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                break
        if not valid:
            break
        if centers[i][0] - radii[i] < -1e-9 or 1 - centers[i][0] - radii[i] < -1e-9:
            valid = False
            break
        if centers[i][1] - radii[i] < -1e-9 or 1 - centers[i][1] - radii[i] < -1e-9:
            valid = False
            break
    
    if not valid:
        return None
    
    result = np.zeros((len(centers), 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    
    return result

def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) where each row is (x, y, r) for a circle
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    centers, radii = _initial_guess()
    
    best_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Try to improve with optimization
    n_circles = 21
    initial_params = np.zeros(3 * n_circles)
    for i in range(n_circles):
        initial_params[3*i] = centers[i][0]
        initial_params[3*i+1] = centers[i][1]
        initial_params[3*i+2] = radii[i]
    
    constraints = _constraints(initial_params, n_circles)
    
    try:
        result = minimize(
            _compute_objective,
            initial_params,
            args=(n_circles,),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if time.time() > deadline:
            pass  # Use best result so far
        
        else:
            if result.success:
                opt_centers = result.x[0::3]
                opt_radii = result.x[2::3]
                
                # Validate
                valid = True
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        dist = np.sqrt((opt_centers[i] - result.x[3*j])**2 + (result.x[3*i+1] - result.x[3*j+1])**2)
                        if dist < opt_radii[i] + opt_radii[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                    if result.x[3*i] - opt_radii[i] < -1e-9 or 1 - result.x[3*i] - opt_radii[i] < -1e-9:
                        valid = False
                        break
                    if result.x[3*i+1] - opt_radii[i] < -1e-9 or 1 - result.x[3*i+1] - opt_radii[i] < -1e-9:
                        valid = False
                        break
                
                if valid and np.sum(opt_radii) > best_sum:
                    best_sum = np.sum(opt_radii)
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()
    except Exception:
        pass
    
    # If optimization didn't improve, try a simpler grid-based approach with larger circles
    # Use a 5x4 grid (20 circles) + 1 extra, or adjust the grid
    
    # Better approach: Use a 5x5 grid but optimize the radii
    # For a 5x5 grid, each cell has side 0.2, so max radius per circle is 0.1
    # But we can do better by packing more efficiently
    
    # Let's try a different initial configuration: hexagonal packing
    if time.time() < deadline:
        try:
            # Hexagonal packing approximation
            hex_centers = []
            hex_radii = []
            
            # Row 0: 5 circles
            for i in range(5):
                hex_centers.append((0.1 + i * 0.2, 0.1))
                hex_radii.append(0.09)
            
            # Row 1: 4 circles (offset)
            for i in range(4):
                hex_centers.append((0.2 + i * 0.2, 0.1 + 0.1732))
                hex_radii.append(0.09)
            
            # Row 2: 5 circles
            for i in range(5):
                hex_centers.append((0.1 + i * 0.2, 0.2 + 0.3464))
                hex_radii.append(0.09)
            
            # Row 3: 4 circles (offset)
            for i in range(4):
                hex_centers.append((0.2 + i * 0.2, 0.3 + 0.5196))
                hex_radii.append(0.09)
            
            # Row 4: 3 circles
            for i in range(3):
                hex_centers.append((0.2 + i * 0.2, 0.4 + 0.6928))
                hex_radii.append(0.09)
            
            hex_centers = np.array(hex_centers[:21])
            hex_radii = np.array(hex_radii[:21])
            
            hex_sum = np.sum(hex_radii)
            if hex_sum > best_sum:
                best_sum = hex_sum
                best_centers = hex_centers.copy()
                best_radii = hex_radii.copy()
        except Exception:
            pass
    
    # Create final result
    result = np.zeros((21, 3))
    result[:, 0] = best_centers[:21]
    result[:, 1] = best_centers[:21]
    result[:, 2] = best_radii[:21]
    
    return result
# EVOLVE-BLOCK-END

