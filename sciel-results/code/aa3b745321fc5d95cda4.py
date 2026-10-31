import numpy as np
import time
from scipy.optimize import minimize

N = 26
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns an array of shape (N, 3) with rows (x, y, radius).
    """
    # 1. Initialization: Hexagonal lattice
    # We want to pack N=26 circles. 
    # A hexagonal lattice is denser than square.
    # Let's create a hex grid and pick the best N circles that fit.
    
    # Generate a hex grid of points
    # Side length s, height h = s * sqrt(3)
    # We'll generate a large grid and then select a subset that fits in [0,1]^2
    # Actually, simpler: just place them in a hex pattern scaled to fit.
    
    # Approximate number of circles in a hex packing of area 1
    # Area per circle ~ pi * r^2. Density ~ 0.9069.
    # Let's just try a specific arrangement.
    
    # Let's use a simple heuristic to generate initial positions and radii.
    # Start with a grid, then perturb.
    
    # Better: Use a known good configuration for N=26 or similar.
    # Since we don't have one, we generate a hex lattice.
    
    def hex_grid_init(n):
        # Create a hex lattice
        # Rows of circles
        # We want to fit n circles.
        # Let's try to fit them in a roughly square block of hex lattice.
        
        # Estimate side length of the block
        # n ~ k^2. k = sqrt(n) = 5.1. So 5x5 or 6x5.
        
        # Let's generate points in a hex lattice with spacing 1.0
        # Then scale them to fit in [0,1]x[0,1] with a small radius.
        
        points = []
        # Generate a 10x10 hex grid
        for i in range(10):
            for j in range(10):
                x = i + 0.5 * (j % 2)
                y = j * 0.866025  # sqrt(3)/2
                points.append((x, y))
        
        points = np.array(points)
        
        # Sort by distance from center to pick the most central ones?
        # Or just take the first N?
        # To make it symmetric, let's center the grid.
        center = points.mean(axis=0)
        points = points - center
        
        # Pick N points closest to the center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        pts = points[idx]
        
        # Scale to fit in [0,1]x[0,1]
        # Max coordinate
        max_coord = np.max(np.abs(pts))
        # We want the circles to fit. Let's assume radius r.
        # If we scale points by s, the bounding box is s * max_coord.
        # We need s * max_coord + r <= 0.5 (centered at 0.5)
        # Let's pick a small initial radius, say r=0.05
        r_init = 0.05
        s = (0.5 - r_init) / max_coord if max_coord > 0 else 1.0
        
        pts = pts * s + 0.5
        
        # Ensure they are within bounds
        # The scaling might push some out if the shape is irregular, but hex is compact.
        
        radii = np.full(n, r_init)
        
        return np.column_stack((pts[:, 0], pts[:, 1], radii))

    x0 = hex_grid_init(N)
    
    # 2. Define Constraints
    def constraints(x):
        # x is flattened: [x1, y1, r1, x2, y2, r2, ...]
        cons = []
        
        # Boundary constraints
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            
            # ri <= xi
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            # ri <= 1 - xi
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            # ri <= yi
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            # ri <= 1 - yi
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
            # ri > 0 (small positive)
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - 1e-6})
            
        # Disjointness constraints
        for i in range(N):
            for j in range(i+1, N):
                # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
                # xi - xj squared + yi - yj squared - (ri + rj)^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
                })
                
        return cons

    # 3. Objective Function
    def objective(x):
        # Maximize sum of radii -> Minimize negative sum
        radii = x[2::3]
        return -np.sum(radii)

    # 4. Optimization
    best_x = x0.flatten()
    best_val = objective(best_x)
    
    # SLSQP can be slow. Let's use it with reasonable maxiter.
    # We have ~150s. 
    # N=26 -> 78 variables.
    # Constraints: 26*5 + 26*25/2 = 130 + 325 = 455 constraints.
    
    # Let's try running SLSQP.
    # To handle time, we might need to limit iterations or use a multi-start approach if time permits.
    # Given the complexity, a single robust SLSQP run from a good start is safer.
    
    try:
        res = minimize(
            objective, 
            best_x, 
            method='SLSQP', 
            bounds=[(0, 1) for _ in range(2*N)] + [(1e-6, 0.5) for _ in range(N)],
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # 5. Radius Growth Pass (Polish)
    # Fix positions, optimize radii.
    # This is often effective to squeeze out more volume.
    
    def objective_radii_only(r):
        # r is array of N radii
        return -np.sum(r)
    
    def constraints_radii_only(r):
        cons = []
        x_pos = best_x[0::3]
        y_pos = best_x[1::3]
        
        for i in range(N):
            # Boundary
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: x_pos[i] - r[i]})
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: 1 - x_pos[i] - r[i]})
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: y_pos[i] - r[i]})
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: 1 - y_pos[i] - r[i]})
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: r[i] - 1e-6})
            
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_pos[i] - x_pos[j])**2 + (y_pos[i] - y_pos[j])**2
                cons.append({
                    'type': 'ineq',
                    'fun': lambda r, i=i, j=j, d2=dist_sq: d2 - (r[i] + r[j])**2
                })
                
        return cons

    current_radii = best_x[2::3].copy()
    
    try:
        res_r = minimize(
            objective_radii_only,
            current_radii,
            method='SLSQP',
            bounds=[(1e-6, 0.5) for _ in range(N)],
            constraints=constraints_radii_only,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if res_r.success or res_r.fun < -np.sum(current_radii):
            current_radii = res_r.x
            # Update best_x with new radii
            best_x[2::3] = current_radii
            best_val = objective(best_x)
            
    except Exception as e:
        print(f"Radius polish error: {e}")

    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        pass # Just return what we have

    # Final Validation and Cleanup
    # Ensure all constraints are satisfied strictly.
    # If a constraint is slightly violated, shrink radii.
    
    x_final = best_x.reshape(N, 3)
    
    # Simple repair loop
    for _ in range(10):
        valid = True
        # Check bounds
        for i in range(N):
            x, y, r = x_final[i]
            if r > x - 1e-9:
                x_final[i, 2] = max(1e-6, x - 1e-9)
                valid = False
            if r > 1 - x - 1e-9:
                x_final[i, 2] = min(x_final[i, 2], max(1e-6, 1 - x - 1e-9))
                valid = False
            if r > y - 1e-9:
                x_final[i, 2] = min(x_final[i, 2], max(1e-6, y - 1e-9))
                valid = False
            if r > 1 - y - 1e-9:
                x_final[i, 2] = min(x_final[i, 2], max(1e-6, 1 - y - 1e-9))
                valid = False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                dx = x_final[i, 0] - x_final[j, 0]
                dy = x_final[i, 1] - x_final[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = x_final[i, 2] + x_final[j, 2]
                if dist < sum_r - 1e-9:
                    # Shrink both radii proportionally or just shrink the larger one?
                    # Simple: shrink both by half the violation
                    viol = (sum_r - dist) / 2 + 1e-10
                    x_final[i, 2] = max(1e-6, x_final[i, 2] - viol)
                    x_final[j, 2] = max(1e-6, x_final[j, 2] - viol)
                    valid = False
                    
        if valid:
            break
            
    return x_final

# Keep the global N as per seed
N = 26
