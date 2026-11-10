import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    
    # Lattice vectors for honeycomb packing
    # Center-to-center distance is sqrt(3)
    u1 = np.array([np.sqrt(3), 0.0])
    u2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    # Generate candidate lattice points
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            c = i * u1 + j * u2
            points.append(c)
    points = np.array(points)
    
    # Center the grid
    points -= points.mean(axis=0)
    
    # Calculate distances from origin
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take the 12 closest points
    centers = points[sorted_idx[:12]]
    
    # Generate all vertices of the 12 unit hexagons
    vertex_angles = np.arange(6) * np.pi / 3.0
    unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
    
    all_vertices = []
    for c in centers:
        all_vertices.append(c + unit_vertices)
    all_vertices = np.vstack(all_vertices)
    
    # Function to compute min enclosing hexagon side length s and center
    def compute_min_enclosing_hexagon(vertices, num_angles=180):
        min_s = np.inf
        best_params = None
        best_angle = 0
        
        # Search for optimal orientation
        angles = np.linspace(0, np.pi/3, num_angles, endpoint=False)
        
        for theta in angles:
            normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
            normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
            
            projections = vertices @ normals.T
            M_k = np.max(projections, axis=0)
            m_k = np.min(projections, axis=0)
            
            # Widths
            widths = M_k - m_k
            max_width = np.max(widths)
            s_width = max_width / np.sqrt(3)
            
            if s_width >= min_s:
                continue
                
            # Optimize center and s for this angle
            # Objective: minimize s subject to constraints
            def objective(params):
                s, cx, cy = params
                if s < 0: return 1e10
                c = np.array([cx, cy])
                y = c @ normals.T
                lower = m_k - s * np.sqrt(3)/2
                upper = M_k + s * np.sqrt(3)/2
                viol_lower = np.max(np.maximum(0, lower - y))
                viol_upper = np.max(np.maximum(0, y - upper))
                return s + 100 * (viol_lower + viol_upper)
                
            # Initial guess: center is average of midpoints, s is s_width
            midpoints = (M_k + m_k) / 2.0
            # Solve for c such that c . n_k = midpoints?
            # Least squares
            c_init, *_ = np.linalg.lstsq(normals.T, midpoints, rcond=None)
            s0 = s_width
            
            res = minimize(objective, [s0, c_init[0], c_init[1]], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success and res.fun < min_s:
                min_s = res.x[0]
                best_params = res.x
                best_angle = theta
                
        if best_params is None:
             return min_s, np.array([0,0]), best_angle
             
        return best_params[0], np.array([best_params[1], best_params[2]]), best_angle

    s, c_final, best_angle = compute_min_enclosing_hexagon(all_vertices)
    
    # Check time
    if time.time() - start_time > 590:
        # Fallback
        s = 6.0
        c_final = np.array([0,0])
        best_angle = 0.0

    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([c_final[0], c_final[1], best_angle])
    
    return inner_hex_data, outer_hex_data, s
