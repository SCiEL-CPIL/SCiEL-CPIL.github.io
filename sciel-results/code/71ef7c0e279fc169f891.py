import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Try different hexagonal cluster radii and find the one with minimal outer hexagon side
    best = None
    best_side = float('inf')
    
    for r in range(0, 20):
        # Generate hexagonal cluster of radius r
        # Center at origin, hexagons on lattice with spacing sqrt(3)
        hexagons = []
        if r == 0:
            hexagons.append((0.0, 0.0))
        else:
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    k = -i - j
                    if max(abs(i), abs(j), abs(k)) <= r:
                        # Convert axial to pixel: x = sqrt(3)*(i + j/2), y = 1.5*j
                        x = np.sqrt(3) * (i + j / 2.0)
                        y = 1.5 * j
                        hexagons.append((x, y))
        
        if not hexagons:
            continue
        
        # Compute all vertices of all hexagons
        all_vertices = []
        for cx, cy in hexagons:
            for angle_deg in range(0, 360, 60):
                angle = np.radians(angle_deg)
                vx = cx + 0.5 * np.cos(angle)
                vy = cy + 0.5 * np.sin(angle)
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        # Find minimum outer hexagon side length that contains all vertices
        # Try different rotations of the outer hexagon
        min_side_for_r = float('inf')
        for outer_rot in range(0, 30):  # Try 30 rotations (0 to 150 degrees)
            rot_rad = np.radians(outer_rot)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            
            # Rotate vertices
            vx = all_vertices[:, 0] * cos_r - all_vertices[:, 1] * sin_r
            vy = all_vertices[:, 0] * sin_r + all_vertices[:, 1] * cos_r
            
            # For a regular hexagon with side s centered at origin, rotated by 0,
            # the constraint is that for each vertex, the max of the 6 half-plane distances <= s*sqrt(3)/2
            # The 6 normals are at angles 0, 60, 120, 180, 240, 300 degrees
            # Distance from center to each side = s * sqrt(3) / 2
            # For a point (px, py), the signed distance to the side with normal at angle theta is:
            # px*cos(theta) + py*sin(theta)
            # We need max over all 6 normals of (px*cos + py*sin) <= s*sqrt(3)/2
            # and min >= -s*sqrt(3)/2
            
            max_proj = 0
            for theta in range(0, 360, 60):
                th_rad = np.radians(theta)
                proj = vx * np.cos(th_rad) + vy * np.sin(th_rad)
                max_proj = max(max_proj, np.max(proj), -np.min(proj))
            
            side_needed = max_proj / (np.sqrt(3) / 2)
            if side_needed < min_side_for_r:
                min_side_for_r = side_needed
        
        # Count hexagons
        n_hex = len(hexagons)
        if n_hex < N:
            continue
            
        if min_side_for_r < best_side:
            best_side = min_side_for_r
            best = (r, hexagons)
    
    if best is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    r, hexagons = best
    inner = [[x, y, 0.0] for x, y in hexagons]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
