import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 580  # Safety margin for 600s limit
    
    n = 12
    
    # Axes for separation check (normals to edges of unit hexagon with angle 0)
    # For a hexagon with angle 0, vertices are at 0, 60, 120, 180, 240, 300 degrees
    # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
    # We only need 3 unique directions (opposite normals give same separation condition)
    u1 = np.array([np.sqrt(3)/2, 0.5])      # 30 deg
    u2 = np.array([0, 1])                   # 90 deg
    u3 = np.array([-np.sqrt(3)/2, 0.5])     # 150 deg
    
    axes = np.array([u1, u2, u3])
    
    # The width of a unit hexagon along these axes is sqrt(3)
    # Two hexagons are separated if their projection intervals on ALL axes are disjoint
    # i.e., max projection distance > sqrt(3) on at least one axis
    min_dist = np.sqrt(3)
    
    # Objective function
    def objective(x):
        # x is flat array of 24 coords: [x0, y0, x1, y1, ...]
        centers = x.reshape((n, 2))
        
        # Check for intersections (Separating Axis Theorem)
        # For two convex polygons to be separated, there must exist an axis
        # where their projections are disjoint.
        # For two identical regular hexagons (same orientation), the critical axes
        # are the edge normals.
        # If max projection on any of the 3 axes > sqrt(3), they are separated.
        
        min_allowed_proj = min_dist + 1e-6
        
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                max_proj = np.max(projs)
                if max_proj <= min_allowed_proj:
                    # Return a large value to reject this configuration
                    # Use a penalty proportional to violation
                    violation = min_allowed_proj - max_proj
                    return 1e10 + violation
        
        # Containment: all inner hexagons must be inside the outer hexagon
        # Outer hexagon has side length s, centered at origin, angle 0
        # The constraint is: for each inner hexagon center c, and for each axis,
        # |c @ axis| + sqrt(3)/2 <= s * sqrt(3)/2
        # => s >= 1 + (2/sqrt(3)) * max(|c @ axis|)
        
        c_n1 = centers @ u1
        c_n2 = centers @ u2
        c_n3 = centers @ u3
        
        scale = 2.0 / np.sqrt(3)
        
        s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
        s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
        s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
        
        s = max(s1, s2, s3)
        
        return s

    # Generate diverse initial guesses
    def generate_initial_guess(seed_offset=0):
        rng = np.random.RandomState(42 + seed_offset)
        
        # Start with honeycomb lattice and add noise
        e1 = np.array([1.5, np.sqrt(3)/2])
        e2 = np.array([0, np.sqrt(3)])
        
        centers_init = []
        for i in range(3):
            for j in range(4):
                p = i * e1 + j * e2
                centers_init.append(p)
        
        centers_init = np.array(centers_init)
        centroid = np.mean(centers_init, axis=0)
        centers_init -= centroid
        
        # Add random perturbation
        noise = rng.normal(0, 0.3, size=(n, 2))
        centers_init += noise
        
        return centers_init.flatten()
    
    # Optimize with multiple restarts
    best_x = None
    best_f = np.inf
    
    # Try several restarts with different seeds
    max_restarts = 8
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout - 30:
            break
            
        x0 = generate_initial_guess(restart)
        
        # Nelder-Mead optimization
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        
        # Also try a gradient-based refinement if available
        # Use L-BFGS-B with a smoothed objective for better convergence
        if time.time() - start_time > timeout - 20:
            break
            
        def objective_smooth(x):
            centers = x.reshape((n, 2))
            
            # Smooth penalty for separation
            min_allowed_proj = min_dist + 1e-4
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    diff = centers[i] - centers[j]
                    projs = np.abs(diff @ axes.T)
                    max_proj = np.max(projs)
                    if max_proj < min_allowed_proj:
                        violation = min_allowed_proj - max_proj
                        penalty += 100.0 * violation**2
            
            c_n1 = centers @ u1
            c_n2 = centers @ u2
            c_n3 = centers @ u3
            
            scale = 2.0 / np.sqrt(3)
            
            s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
            s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
            s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
            
            s = max(s1, s2, s3)
            
            return s + penalty
        
        res2 = minimize(objective_smooth, x0, method='L-BFGS-B', 
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
        
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
    
    # Extract best result
    centers = best_x.reshape((n, 2))
    
    # Calculate final s
    c_n1 = centers @ u1
    c_n2 = centers @ u2
    c_n3 = centers @ u3
    
    scale = 2.0 / np.sqrt(3)
    
    s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
    s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
    s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
    s = max(s1, s2, s3)
    
    # Add a small safety margin to ensure strict containment
    s = s + 0.005
    
    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
