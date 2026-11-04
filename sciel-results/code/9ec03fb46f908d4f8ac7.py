"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place unit hexagons in a hexagonal lattice arrangement
    # Pointy-top hexagons, side length 1
    # Hex grid coordinates: axial (q, r)
    # Convert to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # For N=11, use a compact hexagonal arrangement
    # Center at (0,0), then 6 neighbors, then 4 more
    # Axial coordinates for a hexagonal cluster of 11:
    # Ring 0: (0,0) -> 1
    # Ring 1: 6 positions -> total 7
    # Ring 2: need 4 more from the 12 positions in ring 2
    
    # Let's try a different arrangement: two rows
    # Actually, let's use a systematic hexagonal packing
    
    # Generate hex grid positions
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # For N=11, use a compact arrangement
    # Try: center + 6 ring + 4 from next ring
    # Ring 2 positions in axial coords:
    ring2_positions = []
    for d in range(1, 3):
        for q in range(-d, d + 1):
            for r in range(-d, d + 1):
                if abs(q + r) == d and abs(q) <= d and abs(r) <= d:
                    # Only take positions where max(|q|,|r|,|q+r|) == d
                    if max(abs(q), abs(r), abs(q + r)) == d:
                        ring2_positions.append((q, r))
    
    # Take first 4 from ring 2 (they'll be symmetric-ish)
    # Actually, let's just place 11 in a tight cluster
    # Use a spiral or sorted by distance
    
    # Generate all hex positions up to radius 2
    all_positions = []
    for r in range(-2, 3):
        for q in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                all_positions.append((q, r))
    
    # Sort by distance from center, take first 11
    all_positions.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
    selected = all_positions[:N]
    
    inner = []
    for q, r in selected:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])  # No rotation needed for hex tiling
    
    # Compute minimal enclosing hexagon
    # For pointy-top hexagons, the enclosing hexagon should also be pointy-top
    # The side length of the enclosing hexagon is determined by the max projection
    # in the 6 directions of the hexagon
    
    # Get centers
    centers = np.array([[p[0], p[1]] for p in inner])
    
    # For a regular hexagon with center (cx, cy) and side s, pointy-top:
    # The vertices are at angles 30, 90, 150, 210, 270, 330 degrees
    # A point (x,y) is inside if its projection onto each of the 6 edge normals <= s
    # Edge normals for pointy-top hex: angles 0, 60, 120, 180, 240, 300
    # Actually, for a hexagon with flat sides, the distance from center to each side is s * cos(30) = s*sqrt(3)/2
    # Wait, let me reconsider.
    
    # For a regular hexagon with side length s, the distance from center to each side (apothem) is s * sqrt(3)/2
    # For pointy-top, the "width" (distance between opposite sides) is 2 * apothem = s * sqrt(3)
    # The "height" (distance between opposite vertices) is 2s
    
    # To find the minimum s such that all centers are inside the hexagon (and the hexagons themselves fit):
    # Each inner hexagon has "radius" s_inner=1, so its center must be at least apothem_inner = sqrt(3)/2 from the boundary
    # So the enclosing hexagon's apothem must be >= max projection of centers + sqrt(3)/2
    
    # For pointy-top hexagon centered at origin with side S:
    # The 6 sides are at distance S*sqrt(3)/2 from center
    # Normal directions: 90, 150, 210, 270, 330, 30 degrees (pointing outward)
    # Actually, for pointy-top, vertices at 90, 150, 210, 270, 330, 30
    # Sides are perpendicular to directions: 0, 60, 120, 180, 240, 300
    # Wait, I need to be careful.
    
    # Let me use a different approach: for a regular hexagon, a point is inside iff
    # for all 6 edge normal directions n_i, dot(p - center, n_i) <= apothem
    # where n_i are unit vectors pointing outward perpendicular to each edge
    
    # For pointy-top hexagon with side S:
    # Vertices at angles 30+60k degrees for k=0..5 (if we start from top)
    # Actually standard: vertices at 90, 150, 210, 270, 330, 30
    # Edge midpoints (apothem directions) at 0, 60, 120, 180, 240, 300
    # Apothem = S * sqrt(3)/2
    
    # So for each direction angle a in [0, 60, 120, 180, 240, 300]:
    # The constraint is: max over points of (x*cos(a) + y*sin(a)) <= S*sqrt(3)/2
    # And min >= -S*sqrt(3)/2
    # So S >= 2 * max(|projection|) / sqrt(3) for each direction
    
    # But we also need the inner hexagons to fit, so add their apothem sqrt(3)/2
    
    # Let's compute for the current arrangement
    cx, cy = 0.0, 0.0  # Assume center at origin for now
    points = centers - np.array([cx, cy])
    
    S_min = 0.0
    for angle_deg in range(0, 360, 60):
        angle = np.radians(angle_deg)
        nx, ny = np.cos(angle), np.sin(angle)
        projections = points[:, 0] * nx + points[:, 1] * ny
        max_proj = np.max(np.abs(projections))
        # Need S such that S*sqrt(3)/2 >= max_proj + sqrt(3)/2 (inner hex apothem)
        S_candidate = (max_proj + np.sqrt(3) / 2) * 2 / np.sqrt(3)
        S_min = max(S_min, S_candidate)
    
    # Try to optimize the center position
    def objective(params):
        cx, cy = params
        pts = centers - np.array([cx, cy])
        S = 0.0
        for angle_deg in range(0, 360, 60):
            angle = np.radians(angle_deg)
            nx, ny = np.cos(angle), np.sin(angle)
            projections = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = np.max(np.abs(projections))
            S_candidate = (max_proj + np.sqrt(3) / 2) * 2 / np.sqrt(3)
            S = max(S, S_candidate)
        return S
    
    result = minimize(objective, [0.0, 0.0], method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    best_cx, best_cy = result.x
    best_S = result.fun
    
    return inner, (best_cx, best_cy), best_S, 0.0
# EVOLVE-BLOCK-END
