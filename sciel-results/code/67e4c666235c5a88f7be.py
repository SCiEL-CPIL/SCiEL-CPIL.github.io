import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def get_centers_candidate_1():
    # Flower + 4 Cardinals
    d = np.sqrt(3)
    centers_axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    centers = []
    for q, r in centers_axial:
        x = d * (q + r/2)
        y = d * (r * np.sqrt(3) / 2)
        centers.append((x, y))
    return centers

def get_centers_candidate_2():
    # Flower + 4 Inter-cardinals (diagonals)
    d = np.sqrt(3)
    centers_axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,-1), (1,-1), (-1,1) # Note: (1,-1) and (-1,1) are already in the flower?
        # Flower: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # So (1,1) and (-1,-1) are NEW.
        # (1,-1) and (-1,1) are ALREADY IN THE FLOWER.
        # So this candidate adds (1,1), (-1,-1) and re-uses (1,-1), (-1,1)?
        # No, I need 4 NEW points.
        # The flower has 7.
        # I need 4 more.
        # The points (1,1), (-1,-1) are in Ring 2.
        # The points (1,-1), (-1,1) are in Ring 1.
        # So I can't add (1,-1) again.
        # I need 4 points from Ring 2.
        # Ring 2 points: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Candidate 2: (1,1), (-1,-1), (0,2), (0,-2)?
        # Let's try: (1,1), (-1,-1), (2,0), (-2,0)?
    ]
    # Let's just use the two main symmetric ones:
    # C1: (2,0), (-2,0), (0,2), (0,-2)
    # C2: (1,1), (-1,-1), (1,-1)?? No.
    # Let's try: (1,1), (-1,-1), (-1,2), (1,-2)?
    
    # Actually, let's just use C1. It's a good bet.
    pass

def construct():
    # Pre-compute vertices for the cluster at rotation 0
    centers = get_centers_candidate_1()
    # Center of mass
    com = np.mean(centers, axis=0)
    centers = [(c[0]-com[0], c[1]-com[1]) for c in centers]
    
    # Generate all vertices of the 11 hexagons
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * np.pi / 3
            x = cx + np.cos(angle)
            y = cy + np.sin(angle)
            vertices.append((x, y))
            
    vertices = np.array(vertices)
    
    def width_at_angle(theta):
        u = np.array([np.cos(theta), np.sin(theta)])
        projections = vertices @ u
        return np.max(projections) - np.min(projections)
        
    def objective(phi):
        # phi is the rotation of the inner cluster
        # Outer hexagon is fixed at 0
        # Widths in directions 0, pi/3, ...
        max_w = 0
        for k in range(6):
            angle = k * np.pi / 3 - phi
            w = width_at_angle(angle)
            if w > max_w:
                max_w = w
        return max_w / np.sqrt(3)
        
    # Search for optimal phi
    # Use a coarse grid first
    best_phi = 0
    best_val = float('inf')
    for i in range(100):
        phi = i * (np.pi/3) / 100
        val = objective(phi)
        if val < best_val:
            best_val = val
            best_phi = phi
            
    # Refine
    res = minimize_scalar(objective, bounds=(best_phi - np.pi/300, best_phi + np.pi/300), method='bounded')
    optimal_phi = res.x
    optimal_side = res.fun
    
    # Construct output
    inner = []
    for cx, cy in centers:
        # Rotate center by optimal_phi
        x_rot = cx * np.cos(optimal_phi) - cy * np.sin(optimal_phi)
        y_rot = cx * np.sin(optimal_phi) + cy * np.cos(optimal_phi)
        angle_deg = np.degrees(optimal_phi)
        inner.append([x_rot, y_rot, angle_deg])
        
    return inner, [0.0, 0.0], optimal_side, 0.0

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
