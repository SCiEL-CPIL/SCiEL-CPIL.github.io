import numpy as np

N = 11

def construct():
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal lattice positions using axial coordinates
    positions = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = sqrt3 * (q + r / 2)
            y = 1.5 * r
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # Select N closest to origin
    dists = np.linalg.norm(positions, axis=1)
    idx = np.argsort(dists)[:N]
    centers = positions[idx]
    
    # Center the cluster
    center_of_mass = centers.mean(axis=0)
    centers = centers - center_of_mass
    
    # Find minimal enclosing regular hexagon
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in np.linspace(0, 60, 601):
        angle = np.radians(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = centers @ R.T
        
        # 6 outward normals for flat-top hexagon
        normals = np.array([[np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)])
        
        projections = rotated @ normals.T
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        
        a_required = np.maximum(max_proj, -min_proj).max()
        side = 2 * a_required / sqrt3
        
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    inner = [[float(x), float(y), 0.0] for x, y in centers]
    
    return inner, (0.0, 0.0), float(best_side), float(best_angle)
