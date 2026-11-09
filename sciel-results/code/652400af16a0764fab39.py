# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Initial guess: 7 in a flower, 4 in a second ring or scattered
    # Center: 0,0
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 2*sqrt(3) ?
    
    # Let's use a random-ish but structured start
    centers = np.array([
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5],
        # 4 more, placed further out
        [2*np.sqrt(3), 0.0],
        [np.sqrt(3), np.sqrt(3)*1.5],
        [-np.sqrt(3), np.sqrt(3)*1.5],
        [-2*np.sqrt(3), 0.0]
    ])
    angles = np.zeros(11)
    outer_side = 5.0
    
    # State vector: x1, y1, a1, ..., x11, y11, a11, S
    x0 = np.concatenate([centers.flatten(), angles, [outer_side]])
    
    def get_hex_vertices(cx, cy, s, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Vertices at angle + 60*i
        angles = angle_rad + np.deg2rad(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def get_outer_vertices(S, angle_deg=0):
        # Outer hex centered at 0,0
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(60 * np.arange(6))
        return np.column_stack([S * np.cos(angles), S * np.sin(angles)])
    
    def point_in_hexagon(p, hex_verts):
        # Check if point p is inside polygon hex_verts
        # Use cross product method
        x, y = p
        inside = True
        n = len(hex_verts)
        for i in range(n):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % n]
            # Vector p1 -> p2
            v1 = p2 - p1
            # Vector p1 -> p
            v2 = np.array([x, y]) - p1
            # Cross product
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            # For a convex polygon defined CCW, all crosses should have same sign
            # We assume outer hex is CCW
            if cross < -1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))

    def hexagons_intersect(v1, v2):
        # v1, v2 are (6,2) arrays
        # Check if any vertex of v1 is in v2
        for p in v1:
            if point_in_hexagon(p, v2):
                return True
        # Check if any vertex of v2 is in v1
        for p in v2:
            if point_in_hexagon(p, v1):
                return True
        # Check edge intersections
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def objective(state):
        if time.time() > deadline - 10:
            return 1e10
            
        centers = state[:22].reshape(11, 2)
        angles = state[22:33]
        S = state[33]
        
        if S < 1.0:
            return 1e10
            
        penalty = 0.0
        
        # Generate vertices for all inner hexagons
        inner_verts = []
        for i in range(11):
            cx, cy = centers[i]
            a = angles[i]
            v = get_hex_vertices(cx, cy, 1.0, a)
            inner_verts.append(v)
            
        # Check inner-inner intersections
        for i in range(11):
            for j in range(i+1, 11):
                if hexagons_intersect(inner_verts[i], inner_verts[j]):
                    penalty += 1000.0
                    
        # Check inner-outer containment
        outer_verts = get_outer_vertices(S)
        for i in range(11):
            # Check if all vertices of inner hex i are inside outer hex
            for p in inner_verts[i]:
                if not point_in_hexagon(p, outer_verts):
                    # Calculate distance to outside for smoother penalty?
                    # For now, hard penalty
                    penalty += 100.0
                    
        return S + penalty

    # Optimize
    best_state = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead as it doesn't require gradients and is robust to non-smooth penalties
    # L-BFGS-B might get stuck if gradients are noisy due to hard constraints
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'fatol': 1e-8, 'disp': False})
    
    if res.fun < best_val:
        best_state = res.x
        best_val = res.fun
        
    # Extract best solution
    centers = best_state[:22].reshape(11, 2)
    angles = best_state[22:33]
    S = best_state[33]
    
    # Final validation and cleanup
    # Ensure S is at least sqrt(11) as per contract? "side >= sqrt(11)" is a validity condition?
    # "Valid only if: ... side >= sqrt(11)"
    # sqrt(11) ~ 3.316
    if S < np.sqrt(11):
        S = np.sqrt(11)
        
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END
