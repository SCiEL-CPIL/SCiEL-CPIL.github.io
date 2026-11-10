import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses L-BFGS-B to minimize outer hexagon side length with smooth penalties.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin

    # Fixed angle for inner hexagons to simplify constraints
    inner_angle = 30.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angles = np.deg2rad(angle_deg) + np.deg2rad(60 * np.arange(6))
        return np.column_stack([cx + side * np.cos(angles), 
                                cy + side * np.sin(angles)])

    def objective(params):
        """
        params[0] = s
        params[1:25] = centers (12, 2)
        """
        if time.time() > deadline:
            return 1e10

        s = params[0]
        centers = params[1:25].reshape(12, 2)
        
        # s must be at least sqrt(12)
        if s < np.sqrt(12):
            return 1e10 + (np.sqrt(12) - s) ** 2

        # Outer hexagon vertices
        outer_verts = get_hex_vertices(0, 0, s, 30.0)
        
        # Penalty for boundary violation
        bound_penalty = 0.0
        for i in range(12):
            cx, cy = centers[i]
            inner_verts = get_hex_vertices(cx, cy, 1.0, inner_angle)
            for v in inner_verts:
                # Check against each edge of outer hex
                # Edge i: p1 to p2. Normal points inward.
                # For a regular hexagon centered at origin, inward normal for edge i
                # can be computed.
                # Simpler: use cross product with edge vectors.
                # If cross < 0, point is outside (assuming CCW vertices and inward check).
                # get_hex_vertices produces CCW vertices.
                # For edge p1->p2, vector p2-p1.
                # Point v is inside if cross(p2-p1, v-p1) >= 0.
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j + 1) % 6]
                    cross = np.cross(p2 - p1, v - p1)
                    if cross < 0:
                        bound_penalty += cross ** 2
                        
        # Penalty for overlap
        # For aligned hexagons, min center distance is sqrt(3)
        overlap_penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    overlap_penalty += (min_dist - d) ** 2

        # Objective: minimize s + weighted penalties
        # We want s to be small, but penalties must be zero.
        # If penalties are non-zero, we are in an invalid state.
        # We use a large weight for penalties to ensure they are resolved first.
        weight_bound = 100.0
        weight_overlap = 100.0
        
        return s + weight_bound * bound_penalty + weight_overlap * overlap_penalty

    # Initial configuration: Compact lattice cluster
    # Lattice vectors: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    # Points (i, j) in lattice coords -> x = sqrt(3)*(i + j/2), y = 1.5*j
    # We pick 12 points forming a compact shape
    lattice_points = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    centers_init = np.zeros((12, 2))
    for k, (i, j) in enumerate(lattice_points):
        centers_init[k, 0] = np.sqrt(3) * (i + j / 2.0)
        centers_init[k, 1] = 1.5 * j
        
    # Center the cluster
    center_of_mass = np.mean(centers_init, axis=0)
    centers_init -= center_of_mass
    
    # Estimate initial s
    # Max distance from center to any vertex + 1 (hex radius)
    max_dist = np.max(np.linalg.norm(centers_init, axis=1))
    # Hexagon circumradius is s. Inradius is s*sqrt(3)/2.
    # Distance from center to vertex of inner hex is 1.
    # So we need s such that outer hex contains all.
    # Rough estimate: s >= max_dist + 1
    s_init = max_dist + 1.5
    
    x0 = np.concatenate([[s_init], centers_init.flatten()])
    
    bounds = [(np.sqrt(12), None)] + [(-20, 20)] * 24
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    if time.time() > deadline:
        # Fallback to initial if timeout (should not happen with this fast objective)
        s_final = s_init
        centers_final = centers_init
    else:
        s_final = result.x[0]
        centers_final = result.x[1:].reshape(12, 2)
        
    # Ensure validity: if there are still small violations, expand s slightly
    # Check final validity
    outer_verts = get_hex_vertices(0, 0, s_final, 30.0)
    valid = True
    for i in range(12):
        cx, cy = centers_final[i]
        inner_verts = get_hex_vertices(cx, cy, 1.0, inner_angle)
        for v in inner_verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                cross = np.cross(p2 - p1, v - p1)
                if cross < -1e-8:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(centers_final[i] - centers_final[j])
            if d < np.sqrt(3) - 1e-8:
                valid = False
                break
        if not valid: break

    if not valid:
        # Expand s until valid
        for _ in range(100):
            s_final *= 1.01
            outer_verts = get_hex_vertices(0, 0, s_final, 30.0)
            valid = True
            for i in range(12):
                cx, cy = centers_final[i]
                inner_verts = get_hex_vertices(cx, cy, 1.0, inner_angle)
                for v in inner_verts:
                    for j in range(6):
                        p1 = outer_verts[j]
                        p2 = outer_verts[(j + 1) % 6]
                        cross = np.cross(p2 - p1, v - p1)
                        if cross < -1e-8:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid: break
            
    inner_hex_data = np.column_stack((centers_final, np.full(12, inner_angle)))
    outer_hex_data = np.array([0.0, 0.0, 30.0])
    
    return inner_hex_data, outer_hex_data, s_final
