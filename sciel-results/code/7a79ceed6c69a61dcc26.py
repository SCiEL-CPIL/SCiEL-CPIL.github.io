import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 11
    
    # Helper functions
    
    def get_hex_vertices(cx, cy, side, angle_rad):
        """Get vertices of a regular hexagon."""
        angles = angle_rad + 60 * np.pi / 180 * np.arange(6)
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def compute_constraints(params):
        """
        Compute constraint violations.
        params: [x0, y0, a0, ..., x10, y10, a10, S]
        Returns a vector of values that must be >= 0 for a valid solution.
        """
        # Unpack parameters
        pos = params[:n*3].reshape(n, 3)
        S = params[n*3]
        
        x = pos[:, 0]
        y = pos[:, 1]
        angles = pos[:, 2] # in radians
        
        constraints = []
        
        # 1. Containment Constraints
        # Outer hexagon is centered at (0,0) with side S and angle 0.
        outer_verts = get_hex_vertices(0, 0, S, 0)
        
        for i in range(n):
            vi = get_hex_vertices(x[i], y[i], 1.0, angles[i])
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                edge = p2 - p1
                # For CCW polygon, inside is to the left.
                # cross(edge, p - p1) >= 0
                # We want cross(edge, p - p1) >= 0
                # So constraint is cross(edge, p - p1) >= 0
                for p in vi:
                    cross_val = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    constraints.append(cross_val)
        
        # 2. Separation Constraints
        # For each pair of inner hexagons, ensure they are disjoint.
        # Using SAT: For all axes (normals to edges of both hexagons),
        # the projection intervals must be disjoint.
        # Gap = max(min_i - max_j, min_j - max_i)
        # We require Gap >= 0 (strictly > 0 for robustness, but >= 0 is the boundary)
        
        for i in range(n):
            vi = get_hex_vertices(x[i], y[i], 1.0, angles[i])
            for j in range(i+1, n):
                vj = get_hex_vertices(x[j], y[j], 1.0, angles[j])
                
                # Get normals for hex i and hex j
                # Edge k of hex i: from vi[k] to vi[(k+1)%6]
                min_gap = np.inf
                for k in range(6):
                    # Axis from hex i
                    edge_i = vi[(k+1)%6] - vi[k]
                    normal_i = np.array([-edge_i[1], edge_i[0]])
                    norm_i = np.linalg.norm(normal_i)
                    if norm_i > 1e-9:
                        normal_i /= norm_i
                        proj_i = np.dot(vi, normal_i)
                        proj_j = np.dot(vj, normal_i)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        min_gap = min(min_gap, gap)
                    
                    # Axis from hex j
                    edge_j = vj[(k+1)%6] - vj[k]
                    normal_j = np.array([-edge_j[1], edge_j[0]])
                    norm_j = np.linalg.norm(normal_j)
                    if norm_j > 1e-9:
                        normal_j /= norm_j
                        proj_i = np.dot(vi, normal_j)
                        proj_j = np.dot(vj, normal_j)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        min_gap = min(min_gap, gap)
                
                # We require min_gap >= 0
                constraints.append(min_gap)
        
        return np.array(constraints)

    # Initial guess
    # Use a hexagonal lattice arrangement as a starting point
    # Spacing d = 2*sqrt(3)
    d = 2 * np.sqrt(3)
    
    # 1 center, 6 around, 4 more
    # Let's try a 3-5-3 grid in hex lattice coords
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,1), (-1,1)
    # (1,-1), (-1,-1)
    # (2,0), (-2,0) -> This is 11 points? 1+2+2+4+2 = 11. Yes.
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (1, -1), (-1, -1),
        (2, 0), (-2, 0)
    ]
    
    a1 = np.array([d, 0])
    a2 = np.array([d/2, d*np.sqrt(3)/2])
    
    init_pos = np.array([i*a1 + j*a2 for i, j in lattice_coords])
    init_angles = np.zeros(n)
    
    # Flatten: [x0, y0, a0, ..., x10, y10, a10, S]
    # Estimate initial S
    # The bounding box of this lattice is roughly width 4d, height 3d*sqrt(3)/2?
    # Let's just start with a large S and let it shrink
    init_S = 12.0
    
    x0 = np.zeros(n*3 + 1)
    for i in range(n):
        x0[i*3] = init_pos[i, 0]
        x0[i*3+1] = init_pos[i, 1]
        x0[i*3+2] = init_angles[i]
    x0[n*3] = init_S
    
    # Define objective function
    def objective(params):
        return params[n*3] # Minimize S
    
    # Define constraints for scipy
    # scipy constraints: fun(x) >= 0
    def constraint_fun(params):
        return compute_constraints(params)
    
    # Bounds for parameters
    # x, y should be within some reasonable range
    # angles should be in [0, 2*pi)
    # S should be > 0
    bounds = []
    for i in range(n):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 2*np.pi)) # angle
    bounds.append((np.sqrt(11), 20)) # S
    
    # Run optimization
    # SLSQP is good for this
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=[{'type': 'ineq', 'fun': constraint_fun}],
                   options={'maxiter': 500, 'ftol': 1e-9})
    
    # Extract results
    if res.success:
        final_params = res.x
    else:
        # Fallback to initial guess if optimization fails, though it should rarely happen
        final_params = x0
    
    final_pos = final_params[:n*3].reshape(n, 3)
    final_S = final_params[n*3]
    
    # Convert angles to degrees
    final_angles_deg = np.degrees(final_pos[:, 2])
    
    inner_hex_data = np.column_stack([final_pos[:, 0], final_pos[:, 1], final_angles_deg])
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, final_S
