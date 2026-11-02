import numpy as np
import itertools
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 # seconds
    
    # Lattice generation
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    # Generate points for R=1 and R=2
    points_R1 = []
    points_R2 = []
    
    # R=1
    r1_coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in r1_coords:
        points_R1.append(axial_to_cartesian(q, r))
        
    # R=2
    # Points at distance 2
    # (2,0), (1,1), (0,2), (-1,3)?? No.
    # Hex distance: max(|q|, |r|, |q+r|)
    r2_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                r2_coords.append((q, r))
                
    for q, r in r2_coords:
        points_R2.append(axial_to_cartesian(q, r))
        
    best_S = float('inf')
    best_points = None
    best_translation = np.array([0.0, 0.0])
    
    # Iterate over combinations of 4 points from R2
    # Add to the fixed R1 points
    for combo in itertools.combinations(range(len(points_R2)), 4):
        if time.time() > deadline - 10:
            break
            
        selected_R2 = [points_R2[i] for i in combo]
        all_points = points_R1 + selected_R2
        
        # Calculate widths in 6 directions
        # Normals for flat-top hexagon (sides at 30, 90, ...)
        # Wait, my lattice is aligned with flat-top?
        # Vertices of unit hexagon at 0, 60...
        # Sides are perpendicular to 30, 90...
        # So normals are at 30, 90, 150, 210, 270, 330 degrees.
        angles = [np.deg2rad(a) for a in [30, 90, 150, 210, 270, 330]]
        normals = [np.array([np.cos(a), np.sin(a)]) for a in angles]
        
        max_width = 0.0
        for n in normals:
            projections = np.dot(all_points, n)
            width = np.max(projections) - np.min(projections)
            if width > max_width:
                max_width = width
                
        # S' = max_width / sqrt(3)
        S_prime = max_width / sqrt3
        S = S_prime + 1.0
        
        if S < best_S:
            best_S = S
            best_points = all_points
            # Calculate optimal translation
            # Center the projections
            # We want t such that (p - t) . n is centered
            # For each n, center is (max+min)/2
            # Solve for t using first two normals
            n0 = normals[0]
            n1 = normals[1]
            
            proj0 = np.dot(all_points, n0)
            center0 = (np.max(proj0) + np.min(proj0)) / 2.0
            
            proj1 = np.dot(all_points, n1)
            center1 = (np.max(proj1) + np.min(proj1)) / 2.0
            
            # t . n0 = center0
            # t . n1 = center1
            A = np.array([n0, n1]).T
            b = np.array([center0, center1])
            
            try:
                t = np.linalg.solve(A, b)
                best_translation = t
            except np.linalg.LinAlgError:
                best_translation = np.array([0.0, 0.0])
                
    if best_points is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0]]
        return inner, (0.0, 0.0), 1.0, 0.0

    # Translate points
    translated_points = [p - best_translation for p in best_points]
    
    # Format output
    inner = []
    for p in translated_points:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
