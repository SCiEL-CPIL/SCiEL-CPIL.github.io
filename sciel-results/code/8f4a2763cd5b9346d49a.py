import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # 1. Generate Hexagonal Lattice Points
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # We want 11 points closest to origin.
    
    points = []
    max_radius = 5 # Enough to cover 11 points
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            # Check if it's in the hex grid (always is, but we limit distance)
            # Distance squared in "axial" metric is q^2 + qr + r^2
            d2 = q*q + q*r + r*r
            if d2 <= max_radius*max_radius:
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                points.append((x, y, d2))
    
    # Sort by distance to origin
    points.sort(key=lambda p: p[2])
    
    # Take first N
    selected_points = [p[:2] for p in points[:N]]
    
    # 2. Generate Vertices of Inner Hexagons
    # Unit hexagon, angle 0
    # Vertices at angles 0, 60, ... 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    hex_verts = np.column_stack((cos_a, sin_a)) # 6x2
    
    all_vertices = []
    for cx, cy in selected_points:
        verts = hex_verts + np.array([cx, cy])
        all_vertices.extend(verts.tolist())
        
    all_vertices = np.array(all_vertices)
    
    # 3. Optimize Container
    # Container: Regular Hexagon, Side S, Center (0,0), Rotation alpha
    # We minimize S.
    # S is determined by the max projection of all_vertices onto the 3 principal axes of the container.
    # Axes are at alpha, alpha+60, alpha+120.
    # Inequality: |p . u| <= S * sqrt(3) / 2
    # So S >= (2/sqrt(3)) * max(|p . u|)
    
    best_S = float('inf')
    best_alpha = 0.0
    
    # Search alpha in [0, 60 degrees]
    # Step size 0.1 degrees
    for alpha_deg in np.arange(0, 60, 0.1):
        if time.time() > deadline:
            break
            
        alpha = np.deg2rad(alpha_deg)
        
        # Principal axes of the container (normals to the sides)
        # The sides are perpendicular to these axes.
        # For a regular hexagon, the normals to the sides are at angles:
        # alpha, alpha + 60, alpha + 120, alpha + 180, ...
        # We only need 3 unique directions due to symmetry of |dot|
        axes = []
        for k in range(3):
            theta = alpha + k * np.pi / 3.0
            axes.append(np.array([np.cos(theta), np.sin(theta)]))
        
        max_proj = 0.0
        for u in axes:
            projs = np.abs(all_vertices @ u)
            max_proj = max(max_proj, np.max(projs))
            
        S = (2.0 / np.sqrt(3.0)) * max_proj
        
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
            
    # Refine around best_alpha
    if time.time() < deadline:
        for alpha_deg in np.arange(best_alpha - 1.0, best_alpha + 1.0, 0.01):
            if alpha_deg < 0 or alpha_deg > 60: continue
            if time.time() > deadline:
                break
            alpha = np.deg2rad(alpha_deg)
            axes = []
            for k in range(3):
                theta = alpha + k * np.pi / 3.0
                axes.append(np.array([np.cos(theta), np.sin(theta)]))
            
            max_proj = 0.0
            for u in axes:
                projs = np.abs(all_vertices @ u)
                max_proj = max(max_proj, np.max(projs))
                
            S = (2.0 / np.sqrt(3.0)) * max_proj
            if S < best_S:
                best_S = S
                best_alpha = alpha_deg
                
    # Construct return values
    # inner: list of [x, y, angle_degrees]
    # All inner hexagons are aligned (angle 0)
    inner = [[p[0], p[1], 0.0] for p in selected_points]
    
    # center: (0.0, 0.0)
    # side: best_S
    # angle: best_alpha
    
    return inner, (0.0, 0.0), best_S, best_alpha

