import numpy as np
import time
from itertools import combinations

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _min_dist_polygons(v1, v2):
    """
    Calculate the minimum distance between two convex polygons.
    If they overlap, return 0.
    """
    n1, n2 = len(v1), len(v2)
    # Check if any vertex of v1 is inside v2 or vice versa, or if edges intersect
    # For convex polygons, if no edges intersect and no vertices are inside, they are disjoint.
    
    # Helper: point in convex polygon
    def _point_in_poly(p, poly):
        n = len(poly)
        # Check orientation of p with respect to each edge
        # All cross products should have the same sign
        signs = []
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross != 0:
                signs.append(1 if cross > 0 else -1)
        if not signs:
            return True # Degenerate
        return all(s == signs[0] for s in signs)

    # Check vertex containment
    for p in v1:
        if _point_in_poly(p, v2):
            return 0.0
    for p in v2:
        if _point_in_poly(p, v1):
            return 0.0

    # Check edge intersections
    def _seg_seg_intersect(p1, p2, p3, p4):
        def _orient(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val > 0: return 1
            if val < 0: return -1
            return 0
        o1 = _orient(p1, p2, p3)
        o2 = _orient(p1, p2, p4)
        o3 = _orient(p3, p4, p1)
        o4 = _orient(p3, p4, p2)
        if o1 != o2 and o3 != o4:
            return True
        return False

    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_intersect(p1, p2, q1, q2):
                return 0.0

    # If disjoint, calculate min distance between edges
    min_d = float('inf')
    def _seg_seg_dist(p1, p2, p3, p4):
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

    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
    return min_d

def _outer_side_for_config(centers, outer_angle_deg):
    """
    Calculate the minimum outer side length required to contain all inner hexagons
    with the given centers and outer angle.
    Inner hexagons are fixed at angle 0.
    """
    # Vertices of a unit hexagon at angle 0
    base_verts = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    
    # Normals of the outer hexagon
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + np.pi/3 * k
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for c in centers:
        verts = c + base_verts
        # Project all vertices onto all normals
        # proj = verts @ normals.T
        # max_proj = max(max_proj, proj.max())
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
            
    # Apothem = S * sqrt(3) / 2
    # max_proj <= Apothem
    # S >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def _check_validity(centers, outer_side, outer_center, outer_angle_deg, eps=1e-7):
    """
    Check if the configuration is valid:
    1. Inner hexagons are disjoint.
    2. Inner hexagons are inside the outer hexagon.
    """
    # 1. Disjointness
    base_verts = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    inner_verts = [c + base_verts for c in centers]
    
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_polygons(inner_verts[i], inner_verts[j])
            if d < eps:
                return False

    # 2. Containment
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle_deg)
    
    # Check if all inner vertices are inside outer hexagon
    # Using half-plane test
    # Outer hexagon is convex.
    # For each edge of outer hexagon, check if points are on the correct side.
    
    # Get outward normals of outer hexagon
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + np.pi/3 * k
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    apothem = outer_side * np.sqrt(3) / 2
    
    for v in inner_verts:
        # v is array of 6 points
        # dist = v @ normals.T
        # if any dist > apothem + eps, it's outside
        dists = v @ normals.T
        if np.any(dists > apothem + eps):
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Generate a set of candidate configurations on a triangular lattice.
    # 2. For each configuration, find the optimal outer angle using ternary search.
    # 3. Verify validity and record the best side length.
    
    best_side = float('inf')
    best_centers = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate lattice points
    # Triangular lattice spacing 1.0 is too tight for unit hexagons (touching).
    # We need a lattice that allows for valid packings.
    # Let's use a grid of points and filter them.
    
    # Range: -5 to 5
    # Step: 0.1 for fine search? That's 101x101 = 10k points. Picking 12 from 10k is huge.
    # Better: Generate specific "motifs" or use a coarser lattice and then refine?
    # Actually, for N=12, a compact cluster is likely.
    # Let's use a triangular lattice with spacing 1.0 but allow offsets?
    # No, let's use a square grid of candidate centers with step 0.25.
    # And use a greedy or random selection?
    
    # A more robust approach for small N:
    # 1. Define a set of "base" configurations (e.g., 2x6 rectangle, 3x4 rectangle, hexagonal cluster).
    # 2. Jitter them?
    # 3. Or, use a lattice of side 1.0 (touching) and just pick valid subsets?
    # If we pick 12 points from a triangular lattice with spacing 1.0, they will be touching (distance 1.0).
    # But unit hexagons on a triangular lattice with spacing 1.0 touch at vertices?
    # Center distance 1.0.
    # Hexagon radius (vertex) = 1.0.
    # If two centers are distance 1.0 apart, their vertices touch.
    # Is that "disjoint"? The task says "disjoint". Usually, touching boundaries might be allowed or not.
    # "Disjoint" usually means interior disjoint.
    # Let's assume touching is allowed (distance >= min_dist).
    # Min dist for unit hexagons to be disjoint (interiors) is 0.
    # But if they overlap, it's invalid.
    # If they touch, min_dist = 0.
    # My _check_validity uses eps=1e-7. So d < 1e-7 is invalid.
    # So touching is valid.
    
    # Let's generate points on a triangular lattice.
    # a = 1.0
    # Points: (i + 0.5*j, j*sqrt(3)/2)
    
    pts = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    
    # We want to pick 12 points.
    # To ensure compactness, we should pick points close to the centroid.
    # Let's sort by distance from origin and take the first 12?
    # This gives a hexagonal cluster of 12 points (center + 6 + 5).
    # This is a very strong candidate.
    
    # Let's try several "seeds" of 12 points from this lattice.
    # Seed 1: Closest 12 to origin.
    # Seed 2: Closest 12 to (1, 0).
    # Seed 3: Random samples from the lattice, ensuring min distance >= 0.99 (to allow for lattice jitter if needed, but we are on lattice).
    
    candidates = []
    
    # Candidate 1: Central cluster
    d = np.sum(pts**2, axis=1)
    idx = np.argsort(d)[:12]
    candidates.append(pts[idx])
    
    # Candidate 2: Shifted cluster
    d = np.sum((pts - [1.0, 0.0])**2, axis=1)
    idx = np.argsort(d)[:12]
    candidates.append(pts[idx])
    
    # Candidate 3: Random 12 with min distance constraint
    np.random.seed(42)
    for _ in range(20):
        # Sample 20 random points, then pick 12 greedily
        sample_idx = np.random.choice(len(pts), 20, replace=False)
        sample_pts = pts[sample_idx]
        # Sort by distance to origin to prefer compact
        d = np.sum(sample_pts**2, axis=1)
        sample_idx_sorted = np.argsort(d)
        sample_pts = sample_pts[sample_idx_sorted]
        
        picked = []
        for p in sample_pts:
            valid = True
            for q in picked:
                if np.linalg.norm(p - q) < 0.99: # Must be distinct lattice points (dist >= 1.0)
                    valid = False
                    break
            if valid:
                picked.append(p)
                if len(picked) == 12:
                    break
        if len(picked) == 12:
            candidates.append(np.array(picked))
            
    # Candidate 4: Rectangular grids
    # 2x6
    rect_pts = []
    for i in range(6):
        for j in range(2):
            # Align with lattice?
            # Just use integer coords for simplicity, then project?
            # Better to stick to lattice points.
            pass
            
    # Let's just use the lattice candidates.
    
    # For each candidate, find optimal outer angle.
    # The function f(angle) = outer_side(angle) is convex?
    # It's a max of linear functions, so it's convex.
    # We can use ternary search or golden section search on [0, 60).
    
    def find_best_angle(centers):
        # Ternary search on [0, 60)
        a, b = 0.0, 60.0
        for _ in range(50):
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            f1 = _outer_side_for_config(centers, m1)
            f2 = _outer_side_for_config(centers, m2)
            if f1 < f2:
                b = m2
            else:
                a = m1
        mid = (a + b) / 2.0
        return mid, _outer_side_for_config(centers, mid)
    
    for centers in candidates:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Center the configuration to minimize side length?
        # The side length calculation assumes outer center is (0,0).
        # We can translate the inner centers to minimize the bounding box.
        # For a fixed orientation, the optimal translation is to center the "mass" or bounding box.
        # Actually, the side length is determined by max projection.
        # Shifting all centers by v shifts all projections by dot(v, n).
        # To minimize max_k (max_i dot(c_i, n_k) + dot(v, n_k)),
        # we want to center the range of projections.
        
        # Let's just try a few translations: (0,0), and centroid.
        centroid = np.mean(centers, axis=0)
        
        for trans in [np.array([0.0, 0.0]), -centroid]:
            shifted_centers = centers + trans
            
            angle, side = find_best_angle(shifted_centers)
            
            # Verify
            if side < best_side:
                # Check validity
                # Outer center is (0,0)
                # Outer angle is angle
                # Outer side is side
                if _check_validity(shifted_centers, side, (0.0, 0.0), angle):
                    best_side = side
                    best_centers = shifted_centers
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = angle
                    
    if best_centers is not None:
        inner = [[best_centers[i][0], best_centers[i][1], 0.0] for i in range(N)]
        return (inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Run construct to ensure it works
# construct()
