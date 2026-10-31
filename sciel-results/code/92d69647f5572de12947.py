import numpy as np
import time
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, angle_deg=0.0, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
        angle = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle + k * np.pi / 3.0
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    def is_inside_hexagon(pts, center, side, angle_deg=0.0):
        """Check if all points are inside or on the boundary of the outer hexagon."""
        angle = np.radians(angle_deg)
        # The 6 side normals
        inside = True
        for k in range(6):
            alpha = angle + k * np.pi / 3.0
            n = np.array([np.cos(alpha), np.sin(alpha)])
            # The side is at distance side*sqrt(3)/2 from center in direction n
            dist = side * sqrt3 / 2.0
            # Project points onto normal
            projections = (pts - np.array(center)) @ n
            if np.any(projections > dist + 1e-9):
                inside = False
                break
        return inside
    
    def compute_min_side(pts_centers, max_angle_steps=24):
        """Find minimal outer hexagon side length containing unit hexagons at given centers."""
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 0.0
        
        # Try different outer hexagon centers and angles
        # Start with centroid
        centroid = np.mean(pts_centers, axis=0)
        
        for angle_deg in range(0, 60, 60 // max_angle_steps):
            angle = np.radians(angle_deg)
            
            # For a given angle, find the minimal side length
            # The side length must satisfy: for each of 6 normals, 
            # max over all inner hex vertices of (v - center)·n <= side*sqrt(3)/2
            
            # We need to find the optimal center too. For fixed angle, 
            # the optimal center minimizes the maximum extent.
            
            # For each normal direction, the constraint is:
            # center·n + max_v(v·n) <= side*sqrt(3)/2 + center·n ... 
            # Actually: (v - center)·n <= side*sqrt(3)/2 for all v
            # So: center·n >= v·n - side*sqrt(3)/2 for all v
            # And: center·n <= v·n + side*sqrt(3)/2 for all v
            
            # This is a linear programming problem. For simplicity, 
            # let's fix center at centroid and compute required side.
            
            center = centroid
            required_sides = []
            for k in range(6):
                alpha = angle + k * np.pi / 3.0
                n = np.array([np.cos(alpha), np.sin(alpha)])
                
                # For each inner hexagon, compute max projection of its vertices
                max_proj = -float('inf')
                for cx, cy in pts_centers:
                    verts = hex_vertices(cx, cy, 0.0, 1.0)
                    projs = (verts - np.array(center)) @ n
                    max_proj = max(max_proj, np.max(projs))
                
                required_sides.append(max_proj * 2.0 / sqrt3)
            
            side = max(required_sides)
            
            if side < best_side:
                best_side = side
                best_center = (center[0], center[1])
                best_angle = angle_deg
        
        # Refine
        for angle_deg in np.linspace(best_angle - 2, best_angle + 2, 9):
            angle = np.radians(angle_deg)
            center = centroid
            required_sides = []
            for k in range(6):
                alpha = angle + k * np.pi / 3.0
                n = np.array([np.cos(alpha), np.sin(alpha)])
                
                max_proj = -float('inf')
                for cx, cy in pts_centers:
                    verts = hex_vertices(cx, cy, 0.0, 1.0)
                    projs = (verts - np.array(center)) @ n
                    max_proj = max(max_proj, np.max(projs))
                
                required_sides.append(max_proj * 2.0 / sqrt3)
            
            side = max(required_sides)
            
            if side < best_side:
                best_side = side
                best_center = (center[0], center[1])
                best_angle = angle_deg
        
        return best_side, best_center, best_angle
    
    # Generate patterns
    base7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    extra_candidates = [(2,0), (2,-1), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (-2,-1), (-1,-2), (0,-3), (1,-3), (2,-3), (3,-2), (3,-1), (3,0)]
    
    patterns = []
    for combo in combinations(extra_candidates, 4):
        pattern = base7 + list(combo)
        if len(set(pattern)) == 11:
            patterns.append(pattern)
    
    best_side = float('inf')
    best_arrangement = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        pts = []
        for q, r in pattern:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            pts.append((x, y))
        pts = np.array(pts)
        
        side, center, angle = compute_min_side(pts)
        
        if side < best_side:
            best_side = side
            best_arrangement = pattern
            best_outer_center = center
            best_outer_angle = angle
    
    if best_arrangement is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for q, r in best_arrangement:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        inner.append([x, y, 0.0])
    
    return inner, best_outer_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
