"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate centers for a side-3 hexagonal cluster (18 positions)
    # Using axial coordinates (q, r) for flat-top hexagons
    # x = 1.5 * q, y = sqrt(3) * (r + q/2)
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # We have 18 centers; pick 12 that form the most compact cluster
    # The 12 closest to center (0,0)
    all_centers.sort(key=lambda c: c[0]**2 + c[1]**2)
    centers = all_centers[:12]
    
    # Compute the minimal enclosing regular hexagon for these centers
    # A regular hexagon of side s, rotated by angle theta, centered at (cx, cy)
    # contains point (x, y) if for all 6 directions, the projection is within bounds
    
    def hexagon_contains(x, y, cx, cy, s, theta):
        """Check if point (x,y) is inside regular hexagon centered at (cx,cy) with side s, rotation theta."""
        dx = x - cx
        dy = y - cy
        # Rotate point by -theta to align hexagon with axes
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        
        # For a regular hexagon of side s aligned with axes (flat-top),
        # the vertices are at angles 0, 60, 120, 180, 240, 300 degrees
        # The hexagon contains (rx, ry) if max over 6 directions of projection <= s
        # Actually, for a regular hexagon with "radius" (center to vertex) = s,
        # it contains points where max(|rx|, |ry|, |rx*0.5 - ry*sqrt(3)/2|, 
        #                              |rx*0.5 + ry*sqrt(3)/2|, |-rx|, |-ry|) <= s
        # But this is for a specific orientation. Let me use the half-width approach.
        
        # For a regular hexagon with side length s, the apothem (center to edge midpoint) = s * sqrt(3)/2
        # The hexagon in standard orientation (flat-top) has:
        #   |x| <= s and |y| <= s*sqrt(3)/2 for the bounding box, but that's not exact.
        
        # Better: a regular hexagon of side s can be described as the intersection of 6 half-planes.
        # For a hexagon with vertices at angles theta + k*60 degrees, distance s from center:
        # The half-plane constraint for edge k: 
        #   (x-cx)*cos(theta + k*pi/3) + (y-cy)*sin(theta + k*pi/3) <= s * sqrt(3)/2
        
        for k in range(6):
            angle = theta + k * math.pi / 3
            proj = dx * math.cos(angle) + dy * math.sin(angle)
            if proj > s * sqrt3 / 2 + 1e-10:
                return False
        return True
    
    def required_side(cx, cy, theta, points):
        """Find minimum side length s such that all points are inside hexagon."""
        s_max = 0.0
        for (px, py) in points:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = theta + k * math.pi / 3
                proj = dx * math.cos(angle) + dy * math.sin(angle)
                if proj > s_max:
                    s_max = proj
        return s_max / (sqrt3 / 2)  # s * sqrt(3)/2 = max_proj, so s = max_proj / (sqrt(3)/2)
    
    # Optimize over center and rotation
    best_s = float('inf')
    best_cx, best_cy, best_theta = 0.0, 0.0, 0.0
    
    # First, try center at centroid of the 12 hexagons
    cx0 = sum(c[0] for c in centers) / N
    cy0 = sum(c[1] for c in centers) / N
    
    # Try various rotations
    for theta in [0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi]:
        s = required_side(cx0, cy0, theta, centers)
        if s < best_s:
            best_s = s
            best_cx, best_cy, best_theta = cx0, cy0, theta
    
    # Fine-tune center position
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        for dx in [-0.1, 0.0, 0.1]:
            for dy in [-0.1, 0.0, 0.1]:
                s = required_side(best_cx + dx, best_cy + dy, best_theta, centers)
                if s < best_s - 1e-9:
                    best_s = s
                    best_cx += dx
                    best_cy += dy
                    improved = True
        if not improved:
            break
    
    # Fine-tune rotation
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for dtheta in [-0.05, 0.0, 0.05]:
            s = required_side(best_cx, best_cy, best_theta + dtheta, centers)
            if s < best_s - 1e-9:
                best_s = s
                best_theta += dtheta
                improved = True
        if not improved:
            break
    
    # Add small epsilon to ensure strict disjointness and containment
    epsilon = 1e-6
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    return inner, (best_cx, best_cy), best_s + epsilon, best_theta
# EVOLVE-BLOCK-END

