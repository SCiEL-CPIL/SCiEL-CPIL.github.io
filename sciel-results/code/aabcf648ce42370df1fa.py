"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Approach: Model as circle packing. Each unit hexagon has circumradius 1.
    The outer hexagon of side length L has circumradius L.
    We need to place N circles of radius 1 inside a circle of radius L (outer circumradius),
    but we also need to ensure the actual hexagons fit inside the outer hexagon.
    
    For a regular hexagon of side length s, the distance from center to any point in the hexagon
    is at most s (circumradius). The inradius is s * sqrt(3)/2.
    
    A unit hexagon (side 1) has circumradius 1 and inradius sqrt(3)/2.
    
    If we place a unit hexagon with center at distance d from the outer hexagon center,
    and the unit hexagon is oriented the same way as the outer hexagon, then the maximum
    distance from the outer center to any vertex of the inner hexagon is d + 1.
    For this to be within the outer hexagon, we need d + 1 <= L (outer side length = circumradius).
    
    However, if the inner hexagon is rotated, the worst-case distance might be different.
    For simplicity and a good heuristic, let's assume all inner hexagons are axis-aligned (angle=0)
    and use the constraint that the center of each inner hexagon must be within distance L-1 
    from the outer center. But this is only valid if the inner hexagon's vertices don't stick out.
    
    Actually, for two concentric, coaxial regular hexagons (same orientation), if the inner one has
    side 1 and the outer has side L, the inner fits iff L >= 1. The distance from center to a vertex
    of the inner is 1, and the outer hexagon extends to distance L from center. So we need L >= 1 + d
    where d is the offset of the inner center from the outer center? No.
    
    Let me think more carefully. The outer hexagon is the set of points (x,y) such that for all 6 
    directions, the projection is within some bound. For a regular hexagon of side L centered at origin
    with a flat top (one side parallel to x-axis), the vertices are at angles 0, 60, 120, 180, 240, 300
    degrees at distance L from center.
    
    A point (x,y) is inside the outer hexagon iff for each of the 6 normals, the dot product with the
    point is <= L * sqrt(3)/2 (the inradius times... wait).
    
    Actually, for a regular hexagon of side length s, the inradius (apothem) is s*sqrt(3)/2.
    The hexagon can be defined as the intersection of 6 half-planes. For a hexagon centered at origin
    with one vertex pointing right (pointy top), the vertices are at (s, 0), (s/2, s*sqrt(3)/2), etc.
    The 6 supporting lines are at distance s*sqrt(3)/2 from the center in directions 30, 90, 150, 210, 270, 330 degrees.
    
    So a point p is inside iff for all k in 0..5: p . n_k <= s*sqrt(3)/2 where n_k are unit vectors
    at angles 30 + 60k degrees.
    
    For an inner unit hexagon with center c and angle theta, its vertices are at c + (cos(theta + 60k), sin(theta + 60k)) for k=0..5.
    All these vertices must be inside the outer hexagon.
    
    This is complex. Let's use the circle approximation first: place N circles of radius 1 inside a circle
    of radius R, minimizing R. Then set L = R (since outer hexagon circumradius = side length). But we need
    to verify the hexagons actually fit. If they don't, we can increase L slightly.
    
    For N=11, a known good packing of 11 circles of radius 1 in a minimum enclosing circle has R ≈ 3.304...
    Let's use a force-directed or simple optimization to find good positions.
    """
    # Initialize positions in a hexagonal pattern
    # For N circles, a hexagonal arrangement
    # For N=11: center + 6 in first ring + 4 in second ring? Or 1+6+4?
    # Actually, 1+6+12=19, so 11 is 1+6+4.
    
    # Let's use a simple grid/hex grid initialization
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # First ring: 6 circles
    for k in range(6):
        angle = k * np.pi / 3
        centers.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    # Second ring: need 4 more
    # Place them at angles between the first ring
    for k in range(4):
        angle = (k + 0.5) * np.pi / 3
        centers.append([4.0 * np.cos(angle), 4.0 * np.sin(angle)])
    
    centers = np.array(centers)
    
    def objective(params):
        # params: 2N values for centers, then 1 for outer radius L
        cx = params[:N]
        cy = params[N:2*N]
        L = params[2*N]
        
        # Penalty for circles overlapping
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        # Penalty for circles outside outer hexagon
        # Approximate: distance from center to outer hexagon boundary
        # For simplicity, use circumradius: dist <= L
        for i in range(N):
            dist = np.sqrt(cx[i]**2 + cy[i]**2)
            if dist > L - 1.0:
                penalty += (dist - (L - 1.0)) ** 2
        
        return L + penalty
    
    # Initial guess
    initial_params = np.concatenate([centers.flatten(), [4.0]])
    
    # Optimize
    result = minimize(objective, initial_params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    cx = best_params[:N]
    cy = best_params[N:2*N]
    L = best_params[2*N]
    
    # Ensure L is at least 1
    L = max(L, 1.0)
    
    # Construct inner hexagons (all with angle 0)
    inner = []
    for i in range(N):
        inner.append([float(cx[i]), float(cy[i]), 0.0])
    
    return inner, (0.0, 0.0), float(L), 0.0
# EVOLVE-BLOCK-END
