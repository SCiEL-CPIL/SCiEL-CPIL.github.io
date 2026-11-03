import numpy as np
from scipy.optimize import minimize
import itertools

N = 12
# Precompute normal vectors for the outer hexagon sides
# Aligned with inner hexagons (angle 0)
# Normals are at angles 0, 60, 120, 180, 240, 300
angles = np.deg2rad(np.arange(0, 360, 60))
normals = np.column_stack((np.cos(angles), np.sin(angles)))

def calc_side_length(positions):
    """
    Calculate the minimum side length S of the outer hexagon
    required to contain all inner hexagons at given positions.
    Assumes all inner hexagons are aligned with the outer one.
    """
    # positions shape (N, 2)
    # Projections of centers onto normals
    # proj shape (N, 6)
    proj = positions @ normals.T
    # Max projection for each normal
    max_proj = np.max(proj, axis=0)
    # The constraint is max_proj[k] + apothem_inner <= apothem_outer
    # apothem_inner = sqrt(3)/2
    # apothem_outer = S * sqrt(3)/2
    # max_proj[k] <= S * sqrt(3)/2 - sqrt(3)/2
    # S >= (max_proj[k] + sqrt(3)/2) / (sqrt(3)/2) = 1 + 2/sqrt(3) * max_proj[k]
    S_vals = 1.0 + (2.0 / np.sqrt(3.0)) * max_proj
    return np.max(S_vals)

def distance_constraints(positions, i, m):
    """
    Returns 0 if constraint is satisfied, negative otherwise.
    Constraint: dist(i, m) >= sqrt(3)
    """
    diff = positions[i] - positions[m]
    dist_sq = np.dot(diff, diff)
    return dist_sq - 3.0

def construct():
    # Initial layout: Hexagonal lattice
    # Generate a grid of points
    # Axial coords (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    points = []
    # Generate enough points to have 12+
    max_coord = 4
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            if abs(q + r) > max_coord:
                continue
            x = np.sqrt(3.0) * (q + r / 2.0)
            y = 1.5 * r
            points.append((x, y))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N points
    init_pos = np.array(points[:N])
    
    # We will try to optimize this layout
    # Objective function to minimize
    def objective(pos_flat):
        pos = pos_flat.reshape(N, 2)
        return calc_side_length(pos)
    
    # Constraints for SLSQP
    # Inequality constraints: g(x) >= 0
    constraints = []
    for i in range(N):
        for m in range(i + 1, N):
            def con_func(pos_flat, i=i, m=m):
                pos = pos_flat.reshape(N, 2)
                return distance_constraints(pos, i, m)
            constraints.append({'type': 'ineq', 'fun': con_func})
    
    # Run optimization
    best_S = np.inf
    best_pos = init_pos
    
    # Try multiple starts to avoid local minima
    # Start 1: The lattice points
    # Start 2: Lattice points with slight random noise
    # Start 3: Different subset of lattice points (e.g. shift the grid)
    
    starts = [init_pos]
    
    # Add perturbed versions
    for _ in range(5):
        noise = np.random.normal(0, 0.1, size=(N, 2))
        starts.append(init_pos + noise)
        
    # Try a different initial selection from the lattice
    # Maybe skip the center? Or take a different shape?
    # For now, just perturbations.
    
    for start in starts:
        try:
            res = minimize(objective, start.flatten(), method='SLSQP', 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
            if res.success:
                S = res.fun
                if S < best_S:
                    best_S = S
                    best_pos = res.x.reshape(N, 2)
        except Exception as e:
            pass
            
    # If optimization failed or didn't improve, use the initial lattice S
    S_initial = calc_side_length(init_pos)
    if best_S > S_initial:
        best_S = S_initial
        best_pos = init_pos

    # Construct inner hexagons list
    # (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = best_S
    angle = 0.0
    
    return inner, center, side, angle

