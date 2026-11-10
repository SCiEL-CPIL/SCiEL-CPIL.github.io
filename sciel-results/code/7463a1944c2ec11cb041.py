"""Packing 11 unit hexagons inside a regular hexagon using cluster-based GA."""
import numpy as np
from scipy.spatial import ConvexHull
import random
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def point_in_hexagon(pt, center, side, angle_deg):
    """Check if point is inside a regular hexagon."""
    # Transform point to hexagon's local coordinate system
    angle_rad = math.radians(angle_deg)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    # Rotate by -angle
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at (side*cos(60*i), side*sin(60*i))
    # The hexagon is defined by |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx + ly/sqrt(3)| <= side
    # Actually for a regular hexagon with flat top (vertex at 0 deg), the inequalities are:
    # |x| <= R, |y| <= R*sqrt(3)/2, |x*sqrt(3)/2 - y/2| <= R where R is circumradius = side
    # Wait, let me use the standard: regular hexagon with side s has inradius s*sqrt(3)/2 and circumradius s.
    # For a hexagon with a vertex pointing right (angle 0), the vertices are at angles 0, 60, 120, 180, 240, 300.
    # The half-width in x direction is s, in y direction is s*sqrt(3)/2.
    # The constraint is: |x| <= s, |y| <= s*sqrt(3)/2, and |x + y/sqrt(3)| <= s and |x - y/sqrt(3)| <= s
    
    R = side
    inr = R * math.sqrt(3) / 2
    
    # Check three pairs of parallel sides
    if abs(lx) > R + 1e-9:
        return False
    if abs(ly) > inr + 1e-9:
        return False
    # The other two pairs:
    # Side 1: x + y/sqrt(3) = R  => x + y/sqrt(3) <= R
    # Side 2: x + y/sqrt(3) = -R => x + y/sqrt(3) >= -R
    val1 = lx + ly / math.sqrt(3)
    if abs(val1) > R + 1e-9:
        return False
    # Actually for a hexagon with vertices at 0,60,120,180,240,300 degrees:
    # The six half-planes are:
    # x <= R, x >= -R
    # -x/sqrt(3) + y <= R, -x/sqrt(3) + y >= -R  (wait, let me recompute)
    # Vertex at 0 deg: (R, 0). Adjacent vertices at 60 deg: (R/2, R*sqrt(3)/2) and 300 deg: (R/2, -R*sqrt(3)/2)
    # Edge from (R,0) to (R/2, R*sqrt(3)/2): direction is (-R/2, R*sqrt(3)/2), normal is (R*sqrt(3)/2, R/2) normalized
    # This is getting complicated. Let me use a simpler check: the hexagon is the intersection of 6 half-planes.
    
    # For a regular hexagon centered at origin with circumradius R and first vertex at angle 0:
    # The supporting lines are at distance inradius = R*sqrt(3)/2 from center, with normals at angles 30, 90, 150, 210, 270, 330 degrees.
    # So for each normal direction n_k = (cos(30+60k), sin(30+60k)), we need dot(p, n_k) <= R*sqrt(3)/2
    
    for k in range(6):
        theta_n = math.radians(30 + 60 * k)
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        if lx * nx + ly * ny > inr + 1e-9:
            return False
    return True

def hexagon_distance(h1_center, h1_angle, h2_center, h2_angle, side=1.0):
    """Compute minimum distance between two unit hexagons. Returns 0 if they overlap."""
    # Get vertices of both hexagons
    v1 = hexagon_vertices(h1_center[0], h1_center[1], side, h1_angle)
    v2 = hexagon_vertices(h2_center[0], h2_center[1], side, h2_angle)
    
    # Check if any vertex of h1 is inside h2 or vice versa
    for pt in v1:
        if point_in_hexagon(pt, h2_center, side, h2_angle):
            return -1.0  # Overlapping
    for pt in v2:
        if point_in_hexagon(pt, h1_center, side, h1_angle):
            return -1.0  # Overlapping
    
    # Compute minimum distance between edges
    min_dist = float('inf')
    for i in range(6):
        for j in range(6):
            # Edge i of h1
            p1 = v1[i]
            p2 = v1[(i+1) % 6]
            # Edge j of h2
            p3 = v2[j]
            p4 = v2[(j+1) % 6]
            
            # Segment-segment distance
            d = seg_seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    
    return min_dist

def seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    # Check all endpoint-to-segment distances
    d1 = point_seg_dist(p1, p3, p4)
    d2 = point_seg_dist(p2, p3, p4)
    d3 = point_seg_dist(p3, p1, p2)
    d4 = point_seg_dist(p4, p1, p2)
    return min(d1, d2, d3, d4)

def point_seg_dist(pt, a, b):
    """Distance from point to line segment ab."""
    ab = b - a
    t = np.dot(pt - a, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(pt - closest)

def max_inradius(center, side, angle_deg, inner_hexes):
    """Compute the minimum distance from center to the farthest vertex of all inner hexagons.
    This gives a lower bound on the required outer hexagon side length (circumradius)."""
    max_r = 0.0
    for (cx, cy, ang) in inner_hexes:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            r = math.sqrt((v[0] - center[0])**2 + (v[1] - center[1])**2)
            if r > max_r:
                max_r = r
    return max_r

def is_valid_packing(inner_hexes, tol=0.001):
    """Check if the packing is valid: no overlaps."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            d = hexagon_distance(inner_hexes[i][:2], inner_hexes[i][2], 
                                  inner_hexes[j][:2], inner_hexes[j][2])
            if d < -tol:  # Overlapping
                return False
    return True

def fitness(inner_hexes, center=(0.0, 0.0)):
    """Compute fitness: outer hexagon side length. Returns infinity if invalid."""
    if not is_valid_packing(inner_hexes):
        return float('inf')
    # The outer hexagon must contain all inner hexagons
    # For a regular hexagon with circumradius R, the maximum distance from center to any point inside is R
    # So we need R >= max distance from center to any vertex of inner hexagons
    R = max_inradius(center, 1.0, 0.0, inner_hexes)
    return R

def construct():
    """Construct a packing of 11 unit hexagons inside a regular hexagon."""
    # Use a structured initial guess based on known packings
    # For 11 hexagons, a good arrangement is a central hexagon surrounded by 6, 
    # then 4 more placed around
    
    # Start with a known good configuration
    # Central hex at origin
    # 6 hexagons around it at distance 2*sqrt(3)/2 = sqrt(3) ≈ 1.732 in directions 0,60,120,180,240,300
    # But for hexagons, the center-to-center distance for touching is 2*inradius = sqrt(3)
    
    sqrt3 = math.sqrt(3)
    
    # Initial configuration: hexagonal packing
    positions = []
    angles = []
    
    # Central hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 surrounding hexagons
    for k in range(6):
        angle = k * 60
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        positions.append((x, y))
        angles.append(0.0)
    
    # 4 more hexagons: place them at the "gaps" between the outer ring
    # The gaps are at angles 30, 90, 150, 210, 270, 330 degrees at distance 2*sqrt3
    # But we only need 4 more. Let's place them at 0, 120, 240 degrees at distance 2*sqrt3
    # Actually, let's try a different arrangement
    
    # Better: use a more compact arrangement
    # Reset and try a different pattern
    
    # Let's use a 3-4-3-1 pattern or similar
    # For now, let's optimize the initial hexagonal packing
    
    inner = []
    for (x, y), a in zip(positions, angles):
        inner.append([x, y, a])
    
    # Now try to improve with local search
    # Simple local search: try small perturbations
    
    best_inner = [list(h) for h in inner]
    best_fitness = fitness(best_inner)
    
    # Local search with random perturbations
    random.seed(42)
    for iteration in range(2000):
        # Pick a random hexagon and perturb it
        idx = random.randint(0, N-1)
        new_inner = [list(h) for h in best_inner]
        
        # Perturb position
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 10)  # degrees
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        f = fitness(new_inner)
        if f < best_fitness:
            best_inner = new_inner
            best_fitness = f
    
    # Get the outer hexagon parameters
    center = (0.0, 0.0)
    side = best_fitness
    angle = 0.0
    
    return best_inner, center, side, angle

# EVOLVE-BLOCK-START
def construct():
    """Construct a packing of 11 unit hexagons inside a regular hexagon."""
    sqrt3 = math.sqrt(3)
    
    # Initial configuration: hexagonal packing with 1 center + 6 around + 4 more
    positions = []
    angles = []
    
    # Central hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 surrounding hexagons at distance sqrt3
    for k in range(6):
        angle = k * 60
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        positions.append((x, y))
        angles.append(0.0)
    
    # 4 more hexagons: place at positions that fill gaps
    # Place them at angles 30, 150, 270, 30 (opposite) at distance 2*sqrt3
    # Actually, let's place them at the vertices of a larger hexagon
    for k in range(4):
        angle = 30 + k * 90  # 30, 120, 210, 300
        rad = math.radians(angle)
        x = 2 * sqrt3 * math.cos(rad)
        y = 2 * sqrt3 * math.sin(rad)
        positions.append((x, y))
        angles.append(0.0)
    
    inner = []
    for (x, y), a in zip(positions, angles):
        inner.append([x, y, a])
    
    # Local search optimization
    random.seed(42)
    best_inner = [list(h) for h in inner]
    best_fitness = fitness(best_inner)
    
    for iteration in range(3000):
        idx = random.randint(0, N-1)
        new_inner = [list(h) for h in best_inner]
        
        dx = random.gauss(0, 0.15)
        dy = random.gauss(0, 0.15)
        da = random.gauss(0, 15)
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        f = fitness(new_inner)
        if f < best_fitness:
            best_inner = new_inner
            best_fitness = f
    
    center = (0.0, 0.0)
    side = best_fitness
    angle = 0.0
    
    return best_inner, center, side, angle
# EVOLVE-BLOCK-END
