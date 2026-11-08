import numpy as np

N = 11

def construct():
    # Generate hexagonal lattice points
    centers = []
    max_r = 2
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            # Check if within hexagonal distance max_r
            # Hex distance in axial coords (q, r) is (|q| + |r| + |q+r|) / 2
            if (abs(q) + abs(r) + abs(q + r)) // 2 <= max_r:
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2.0)
                centers.append((x, y))
                
    # Sort by distance to origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N
    centers = centers[:N]
    
    # Generate all vertices of the unit hexagons
    # Unit hexagon vertices relative to center:
    # (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    angles = np.arange(6) * np.pi / 3
    verts_rel = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    all_verts = []
    for cx, cy in centers:
        verts = verts_rel + np.array([cx, cy])
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Optimize outer hexagon angle
    # Outer hexagon defined by normals at alpha, alpha+60, alpha+120
    # Distance to side D = S * sqrt(3) / 2
    # Constraint: |v . n_i| <= D for all v, i
    # D(alpha) = max_v max_i |v . n_i|
    # We want to minimize D(alpha), which minimizes S.
    
    best_D = float('inf')
    best_alpha = 0.0
    
    # Search alpha in [0, pi/3]
    # Step size 0.01 rad (~0.5 deg) is sufficient for precision
    alphas = np.linspace(0, np.pi/3, 100)
    
    for alpha in alphas:
        normals = []
        for i in range(3):
            phi = alpha + i * np.pi / 3
            normals.append(np.array([np.cos(phi), np.sin(phi)]))
        normals = np.array(normals)
        
        # Calculate max projection for each vertex
        projections = np.abs(all_verts @ normals.T)
        max_proj = np.max(projections)
        
        if max_proj < best_D:
            best_D = max_proj
            best_alpha = alpha
            
    side = 2 * best_D / np.sqrt(3)
    
    # Convert best_alpha to degrees for consistency with inner angles
    best_alpha_deg = np.degrees(best_alpha)
    
    # Format inner hexagons
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), side, best_alpha_deg
