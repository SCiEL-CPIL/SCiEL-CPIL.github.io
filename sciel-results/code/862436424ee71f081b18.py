import numpy as np
import time
from itertools import combinations

N = 12
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    elif e < 1e-12:
        s = -np.dot(d1, r) / a
        s = max(0.0, min(1.0, s))
        return np.linalg.norm(s * d1 - r)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    if denom < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, (b * t - c) / a))
    else:
        s = (b * f - c * e) / denom
        if s < 0.0 or s > 1.0:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            t = (b * s + f) / e
            t = max(0.0, min(1.0, t))
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    
    return np.linalg.norm(s * d1 - r - t * d2)

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (including boundary)."""
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_dist(p1, p2, q1, q2) < eps:
                return True
    return False

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            # For a convex polygon, a point is inside if it's on the same side of all edges
            # We check if any vertex of inner is outside outer
            # Simple method: check if any edge of inner intersects outer boundary, or if center is outside
            # More robust: check if any vertex of inner is outside outer
            # A point P is inside convex polygon V if for all edges (A,B), cross(B-A, P-A) has consistent sign
            # Assuming outer is CCW.
            
            # Let's just check if any inner vertex is outside outer.
            # If an edge intersects, it's also invalid, but vertex check catches most.
            # Actually, if a vertex is outside, it's invalid. If all vertices are inside, 
            # for convex polygons, the whole polygon is inside? 
            # No, if one vertex is inside and another is outside, they cross.
            # If ALL vertices of inner are inside outer, then inner is inside outer (convexity).
            
            is_inside = True
            for p in inner_verts[i]:
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                    # Determine orientation of outer
                    # We can just check if p is to the left of all edges (for CCW)
                    # But we don't know orientation. Let's compute orientation of outer.
                    # Outer hexagon vertices from _hex_vertices are CCW if angle increases.
                    # cos/sin with increasing angle is CCW in standard math coords.
                    # So cross_ab should be >= -eps for all edges if p is inside.
                    if cross_ab < -eps:
                        inside = False
                        break
                if not inside:
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # 1. Generate candidate lattice positions
    # Triangular lattice basis vectors
    # v1 = (1, 0)
    # v2 = (0.5, sqrt(3)/2)
    # We want positions roughly within a radius of 3-4 from origin to keep it compact.
    # Let's generate a grid of indices i, j such that distance from origin is < 4.
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.0 + j * 0.5
            y = j * (np.sqrt(3) / 2.0)
            if x*x + y*y <= 4.0**2: # Radius 4
                candidates.append((x, y))
    
    # We need to choose 12 positions from candidates.
    # To reduce combinations, we can fix the first few or use a heuristic.
    # A better strategy: Try specific "shapes" or use a greedy approach?
    # Given the time limit, a full combinatorial search of C(100, 12) is impossible.
    # We need a smarter search.
    
    # Strategy: 
    # 1. Fix the global rotation angle (try 0 to 30 degrees in small steps, or just 0).
    #    Actually, the lattice is symmetric. Rotating the lattice by 30 degrees maps it to itself.
    #    So we can just fix the lattice orientation and rotate the OUTER hexagon?
    #    The task allows rotating inner hexagons. 
    #    If we place them on a triangular lattice, their natural orientation is 0 or 30 degrees.
    #    Let's try two global inner orientations: 0 and 30.
    
    # 2. To find a valid set of 12 positions, we can use a backtracking search or a randomized restart.
    #    Given 120s, we can do a limited random search.
    
    # Let's try a simpler approach first: 
    # The "hexagonal number" arrangement. 
    # A central hexagon surrounded by 6, then 12... 
    # 1 + 6 + 12 = 19. We only need 12.
    # So we can take the center and the first ring (6) and 5 from the second ring.
    
    # Let's generate the specific set of lattice points corresponding to a "compact" cluster.
    # Points in ring 0: (0,0)
    # Points in ring 1: 6 points at distance 1
    # Points in ring 2: 12 points at distance 2 (in lattice units)
    
    # Let's construct the set of 12 points manually for a few "shapes".
    # Shape 1: Center + 6 neighbors + 5 more from next ring.
    # This is getting complex to enumerate all shapes.
    
    # Alternative: Use a local search.
    # Start with a random set of 12 points from candidates.
    # Check validity. If invalid, swap one point with another to resolve conflict.
    
    # However, the most robust "jump" from the seed (which is just a line) is to find ANY valid compact packing.
    # The seed has side 40. Any valid packing will have side < 10.
    # So score will jump from 1.0 (40/40? No, seed/value. If seed value is 40, and we find 5, score is 40/5 = 8).
    # Wait, the prompt says: "score is the outer side length... search score is seed/value".
    # Seed program returns side 40.0. So seed value is 40.0.
    # If I find a packing with side 5.0, my score is 40.0 / 5.0 = 8.0.
    
    # Let's try to find a valid packing.
    # I will use a simple randomized local search.
    
    np.random.seed(42)
    
    # Precompute lattice points
    lattice_points = np.array(candidates)
    
    # Initial guess: pick 12 random points
    # To make it more likely to be valid, pick points that are close to each other?
    # No, disjointness requires them to be far enough.
    # Lattice points at distance 1 are tangent (distance between centers is 1, side 1 hexagons have width sqrt(3) ~ 1.732? 
    # No, side 1 hexagon has distance between opposite sides sqrt(3). Distance between centers of tangent hexagons is 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732.
    # In triangular lattice, nearest neighbor distance is 1.
    # So hexagons at lattice distance 1 OVERLAP.
    # We need lattice distance >= 2?
    # Distance between centers at lattice distance 2 is 2.
    # 2 > sqrt(3) ~ 1.732. So they are disjoint.
    # So we need to pick 12 points from the triangular lattice such that no two are at distance 1.
    # This is an independent set problem on the triangular lattice graph.
    
    # Let's filter candidates to ensure no two are too close.
    # We can do this by greedy selection or backtracking.
    
    # Let's try to construct a specific valid configuration.
    # Consider a "hexagonal" shape of side length 2 (in lattice units).
    # It has 1 + 6 + 12 = 19 points.
    # We need to remove 7 points such that no two remaining are at distance 1.
    # Actually, if we take all points at even lattice distance from center?
    # Lattice distance d(i,j) = max(|i|, |j|, |i+j|) ? No, that's for hex distance.
    # Triangular lattice distance is Euclidean.
    
    # Let's just use a simple heuristic:
    # 1. Sort all candidate points by distance from origin.
    # 2. Greedily pick points if they are at least 1.8 units away from all already picked points.
    # 3. If we get 12, we have a valid configuration (locally).
    # 4. Then compute the bounding hexagon.
    
    sorted_indices = np.argsort(np.sum(lattice_points**2, axis=1))
    
    picked = []
    for idx in sorted_indices:
        if len(picked) >= 12:
            break
        p = lattice_points[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < 1.8: # sqrt(3) is 1.732, add margin
                valid = False
                break
        if valid:
            picked.append(p)
            
    if len(picked) == 12:
        inner_hexes = [[p[0], p[1], 0.0] for p in picked]
        
        # Now find minimal bounding hexagon.
        # The bounding hexagon should be centered at the centroid of the picked points.
        # Its orientation can be 0 or 30 degrees.
        # We need to find the minimal side length S such that all points are inside.
        
        centroid = np.mean(picked, axis=0)
        
        best_side = float('inf')
        best_outer_angle = 0.0
        
        for outer_angle in [0.0, 30.0]:
            # For a regular hexagon with side S, center C, angle A:
            # The distance from center to each side is S * sqrt(3) / 2.
            # The projection of a point P onto the normal of each side must be <= S * sqrt(3) / 2.
            # Normals are at angles A + 0, 60, 120, 180, 240, 300.
            
            max_proj = 0.0
            for p in picked:
                v = p - centroid
                for k in range(6):
                    angle = np.deg2rad(outer_angle + k * 60)
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    proj = np.dot(v, normal)
                    max_proj = max(max_proj, abs(proj))
            
            # max_proj is the maximum distance from center to a point projected onto normals.
            # This distance must be <= apothem = S * sqrt(3) / 2.
            # So S >= max_proj * 2 / sqrt(3).
            side = max_proj * 2.0 / np.sqrt(3.0)
            
            # Verify validity
            is_valid, _ = _check_validity(inner_hexes, side, centroid, outer_angle)
            if is_valid and side < best_side:
                best_side = side
                best_outer_angle = outer_angle
                
        if best_side < float('inf'):
            result = (inner_hexes, [centroid[0], centroid[1]], best_side, best_outer_angle)
            # The score is seed/value = 40 / best_side.
            # But the function returns the values, and the evaluator calculates the score.
            # The task says: "score is the outer side length". 
            # Wait, "The evaluator measures the raw value (lower is better). The search score is seed/value".
            # So if I return side=5, the evaluator sees value=5.
            # The search score reported to me will be 40/5 = 8.
            # I just need to return the valid packing.
            
            return result
            
    # Fallback: Return the seed configuration if search fails (should not happen)
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
