import numpy as np
import time
import math

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
    Optimized for speed using vectorized numpy operations where possible.
    """
    n1, n2 = len(v1), len(v2)
    
    # Helper: point in convex polygon (vectorized for single point vs poly)
    def _point_in_poly(p, poly):
        # poly: (6, 2), p: (2,)
        # cross products
        p1 = poly
        p2 = np.roll(poly, -1, axis=0)
        edges = p2 - p1
        # Cross product z-component
        # (p2 - p1) x (p - p1)
        # (ex * ey - ey * ex) -> wait, 2D cross is ax*by - ay*bx
        # Let e = p2-p1, v = p-p1
        # cross = e[0]*v[1] - e[1]*v[0]
        v = p - p1
        crosses = edges[:, 0] * v[:, 1] - edges[:, 1] * v[:, 0]
        # If all crosses have same sign (or zero), point is inside
        # For a convex polygon defined counter-clockwise, all crosses should be >= 0 (or <= 0)
        # Since we don't know winding, check if all are non-negative OR all non-positive
        if np.all(crosses >= -1e-10) or np.all(crosses <= 1e-10):
            return True
        return False

    # 1. Check vertex containment
    # Check v1 vertices against v2
    for p in v1:
        if _point_in_poly(p, v2):
            return 0.0
    # Check v2 vertices against v1
    for p in v2:
        if _point_in_poly(p, v1):
            return 0.0

    # 2. Check edge intersections
    def _seg_seg_intersect(p1, p2, p3, p4):
        # Vectorized orientation test
        def _orient(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val > 1e-10: return 1
            if val < -1e-10: return -1
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

    # 3. If disjoint, calculate min distance between edges
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

    # Iterate over all edge pairs
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
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + np.pi/3 * k
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    apothem = outer_side * np.sqrt(3) / 2
    
    for v in inner_verts:
        dists = v @ normals.T
        if np.any(dists > apothem + eps):
            return False
            
    return True

def _relaxation_step(centers, outer_side, outer_angle_deg, step_size=0.1):
    """
    Perform a single force-directed relaxation step.
    Returns new centers and a flag indicating if any overlaps were resolved.
    """
    base_verts = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    inner_verts = [c + base_verts for c in centers]
    
    forces = np.zeros_like(centers)
    moved = False
    
    # 1. Repulsive forces between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_polygons(inner_verts[i], inner_verts[j])
            if d < 1e-7:
                # Overlap detected. Push apart based on center distance.
                # A more sophisticated approach would use contact points, but center push is a good heuristic.
                delta = centers[j] - centers[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-7:
                    # Random nudge
                    angle = np.random.uniform(0, 2*np.pi)
                    delta = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1.0
                else:
                    delta = delta / dist
                # Push force proportional to overlap (approximated by how much dist < min_dist)
                # For hexagons, min center dist for no overlap is ~1.0 (actually depends on angle, but 1.0 is safe for lattice-like)
                # Actually, for unit hexagons, min center dist is 1.0 if aligned, but can be less if rotated.
                # Let's just apply a strong repulsive force if d==0.
                force_mag = 0.5 
                forces[i] -= delta * force_mag
                forces[j] += delta * force_mag
                moved = True
            elif d < 0.2:
                # Soft repulsion to prevent future overlaps
                delta = centers[j] - centers[i]
                dist = np.linalg.norm(delta)
                if dist > 1e-7:
                    delta = delta / dist
                    force_mag = (0.2 - d) * 0.5
                    forces[i] -= delta * force_mag
                    forces[j] += delta * force_mag
                    moved = True

    # 2. Attractive forces towards the center of the outer hexagon (to keep them compact)
    # The outer center is assumed to be (0,0) for this calculation step.
    # We want to minimize the bounding box, so pulling to origin helps.
    for i in range(N):
        dist_to_center = np.linalg.norm(centers[i])
        if dist_to_center > 1.0:
            # Pull back
            force_mag = (dist_to_center - 1.0) * 0.1
            dir_to_center = -centers[i] / dist_to_center
            forces[i] += dir_to_center * force_mag
            moved = True

    # Update centers
    new_centers = centers + forces * step_size
    
    # 3. Constrain to outer hexagon
    # If any hexagon is outside, push it back in.
    # This is a complex projection. For simplicity, if the bounding box grows, we might just reject the step.
    # But for relaxation, we can check if the new configuration fits in the current outer_side.
    # If not, we might need to shrink the step or revert.
    
    # Simple check: does the new configuration fit in the current outer_side?
    # If not, we might have pushed something out.
    # We can compute the required side for the new centers.
    new_required_side = _outer_side_for_config(new_centers, outer_angle_deg)
    if new_required_side > outer_side * 1.01: # 1% tolerance
        # Revert or shrink step
        # For now, just revert the step that caused expansion if it was too big
        # A better approach: project back. But let's just return the original if it got worse.
        return centers, False

    return new_centers, True

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_centers = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Strategy:
    # 1. Initialize centers in a compact cluster (e.g., hexagonal lattice).
    # 2. Run force-directed relaxation to resolve overlaps and tighten packing.
    # 3. Optimize global rotation angle.
    # 4. Iterate with different initializations or step sizes.
    
    def find_best_angle(centers):
        # Golden section search on [0, 60)
        phi = (1 + np.sqrt(5)) / 2
        invphi = 1 / phi
        invphim = 1 - invphi
        a, b = 0.0, 60.0
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = _outer_side_for_config(centers, c)
        fd = _outer_side_for_config(centers, d)
        
        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - invphi * (b - a)
                fc = _outer_side_for_config(centers, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + invphi * (b - a)
                fd = _outer_side_for_config(centers, d)
        
        mid = (a + b) / 2.0
        return mid, _outer_side_for_config(centers, mid)
    
    # Initial configurations
    # 1. Hexagonal cluster
    # 2. Random jittered hexagonal cluster
    # 3. Linear array (fallback)
    
    initializations = []
    
    # Hex cluster: center + 6 neighbors + 5 more
    # Lattice basis: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate lattice points
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    
    # Sort by distance to origin
    d = np.sum(lattice_pts**2, axis=1)
    idx = np.argsort(d)
    
    # Take first 12
    init1 = lattice_pts[idx[:12]].copy()
    initializations.append(init1)
    
    # Jittered version
    np.random.seed(42)
    init2 = init1 + np.random.normal(0, 0.1, init1.shape)
    initializations.append(init2)
    
    # Another jitter
    init3 = init1 + np.random.normal(0, 0.2, init1.shape)
    initializations.append(init3)
    
    # Try relaxation for each initialization
    for init_centers in initializations:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        centers = init_centers.copy()
        
        # Relaxation loop
        # Start with a large "virtual" outer side to allow movement, then tighten?
        # Or start with the side required by the initial config and try to shrink it?
        # The task is to minimize outer side.
        # So, we can simulate the process of shrinking the outer hexagon.
        # If the inner hexagons don't fit, they are pushed apart.
        
        # Let's define a target side length and try to fit.
        # Start with a large side, decrease it.
        
        current_side = _outer_side_for_config(centers, 0.0) * 1.5
        min_side_found = current_side
        
        # Coarse search for side length
        # We want to find the minimum side such that a valid configuration exists.
        
        # Let's do a binary search on the side length?
        # Or just run relaxation with a fixed side and see if it stabilizes?
        
        # Better: Run relaxation with a "pressure" that increases (side decreases).
        
        # Let's try a sequence of side lengths.
        # Estimate lower bound: 12 hexagons. Area ~ 12 * 2.6 = 31.2.
        # Outer hex area = 2.6 * S^2. S ~ sqrt(12) ~ 3.5.
        # So S is likely around 3.5 - 4.5.
        
        # Try sides from 5.0 down to 3.0
        for target_side in np.arange(5.0, 3.0, -0.1):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            # Try to relax into this side
            success = False
            for step in range(100):
                new_centers, moved = _relaxation_step(centers, target_side, 0.0, step_size=0.05)
                if not moved:
                    # Check if it fits
                    req_side = _outer_side_for_config(centers, 0.0)
                    if req_side <= target_side * 1.01:
                        success = True
                        break
                    else:
                        # It doesn't fit, so we need to push harder or increase side
                        # For now, break and try larger side
                        break
                centers = new_centers
                
            if success:
                # Found a valid configuration for this side
                # Now optimize angle
                angle, side = find_best_angle(centers)
                if side < best_side:
                    # Verify
                    if _check_validity(centers, side, (0.0, 0.0), angle):
                        best_side = side
                        best_centers = centers
                        best_outer_angle = angle
                        
                # Try to shrink further?
                # The loop continues to smaller target_side, so we will try.
                # But we should keep the best centers found so far as the starting point for smaller sides?
                # Yes, let's update the starting point for the next (smaller) side.
                # Actually, the `centers` variable is already updated.
                
            # If it failed, we might want to restart with a larger side or different init?
            # For now, just continue.
            
        # End of side loop
        
        # End of init loop
        
    if best_centers is not None:
        inner = [[best_centers[i][0], best_centers[i][1], 0.0] for i in range(N)]
        return (inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback: Return the initial lattice config with its computed side
    centers = initializations[0]
    angle, side = find_best_angle(centers)
    if _check_validity(centers, side, (0.0, 0.0), angle):
        inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        return (inner, best_outer_center, side, angle)
        
    # Absolute fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Run construct to ensure it works
# construct()
