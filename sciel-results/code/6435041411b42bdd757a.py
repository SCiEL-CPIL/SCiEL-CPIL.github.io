"""Hexagonal packing: 12 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 130.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside regular hexagon."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon frame
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    # Regular hexagon with flat-top: |x| <= side*sqrt(3)/2, |y| <= side, and combined
    # Actually for pointy-top vs flat-top... let's use the standard:
    # For a regular hexagon with vertices at angles 0,60,120,...:
    # The hexagon is the intersection of 6 half-planes
    for i in range(6):
        theta = a + i * np.pi / 3.0
        nx = -np.sin(theta)
        ny = np.cos(theta)
        if nx * dx + ny * dy > side * np.sqrt(3) / 2 + 1e-10:
            return False
    return True

def hexagon_overlap_penalty(h1, h2):
    """Smooth penalty for overlap between two hexagons.
    h1, h2: (cx, cy, angle_deg) tuples.
    Uses minimum distance between edges as a smooth measure.
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Get vertices
    v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
    v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
    
    # Check if any vertex of one is inside the other
    # Use a smooth approximation: min distance from vertices of one to edges of other
    min_dist = float('inf')
    
    # Distance from each vertex of hex1 to each edge of hex2
    for i in range(6):
        p = v1[i]
        for j in range(6):
            e1 = v2[j]
            e2 = v2[(j+1) % 6]
            # Distance from point p to segment e1-e2
            d = e2 - e1
            if np.dot(d, d) < 1e-15:
                dist = np.linalg.norm(p - e1)
            else:
                t = np.dot(p - e1, d) / np.dot(d, d)
                t = max(0, min(1, t))
                closest = e1 + t * d
                dist = np.linalg.norm(p - closest)
            min_dist = min(min_dist, dist)
    
    for i in range(6):
        p = v2[i]
        for j in range(6):
            e1 = v1[j]
            e2 = v1[(j+1) % 6]
            d = e2 - e1
            if np.dot(d, d) < 1e-15:
                dist = np.linalg.norm(p - e1)
            else:
                t = np.dot(p - e1, d) / np.dot(d, d)
                t = max(0, min(1, t))
                closest = e1 + t * d
                dist = np.linalg.norm(p - closest)
            min_dist = min(min_dist, dist)
    
    # If min_dist < 0 (shouldn't happen with this method), they overlap
    # The penalty: if hexagons are disjoint, min_dist > 0
    # We want to penalize when they're too close
    # Actually, let's use a different approach: check if centers are close enough
    
    # Better: use the fact that two unit hexagons overlap if distance between centers < some threshold
    # For same orientation: overlap if dist < sqrt(3) (flat-to-flat distance)
    # For different orientations it's more complex
    
    # Let's use a simpler but effective approach:
    # Compute the minimum distance between the two convex polygons
    # If negative, they overlap
    
    # Actually, let me use a cleaner approach with the support function / GJK-like idea
    # But for simplicity and speed, let's use vertex-in-polygon test with smooth penalty
    
    # Count how many vertices of each hex are inside the other
    count1 = 0
    for i in range(6):
        if point_in_hexagon(v1[i][0], v1[i][1], cx2, cy2, 1.0, a2):
            count1 += 1
    
    count2 = 0
    for i in range(6):
        if point_in_hexagon(v2[i][0], v2[i][1], cx1, cy1, 1.0, a1):
            count2 += 1
    
    # If any vertex is inside, there's overlap
    # Smooth penalty: use softplus of the "penetration depth"
    # Approximate penetration depth as -min_dist (when negative)
    
    # For now, use a simple binary penalty with smoothing
    overlap = max(0, count1 + count2)
    return overlap * 10.0 + max(0, -min_dist) * 5.0 if min_dist < 0 else 0.0


def compute_objective(params, outer_side, outer_center, outer_angle):
    """Compute objective: outer_side + penalty for overlaps and containment violations."""
    # params: 3*N = [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
    ocx, ocy = outer_center
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    penalty = 0.0
    
    # Check containment: all vertices of each inner hex must be inside outer hex
    for i in range(N):
        x, y, a = inner[i]
        verts = hexagon_vertices(x, y, 1.0, a)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], ocx, ocy, outer_side, outer_angle):
                # Compute distance to boundary (approximate)
                # Penalty for being outside
                penalty += 100.0
    
    # Check pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            penalty += hexagon_overlap_penalty(inner[i], inner[j])
    
    return outer_side + penalty


def initial_guess():
    """Generate a reasonable initial configuration."""
    # Place hexagons in a hexagonal pattern
    # Center + 6 around + 5 more
    params = []
    
    # Try a pattern: 1 center, 6 in ring, 5 in outer ring
    # Or: 2 rows of 6, or 3 rows of 4
    
    # Let's try: place them in a roughly circular/hexagonal arrangement
    # with some rotation to fit better
    
    # Simple approach: grid-like arrangement
    # Unit hexagons with flat-top have width sqrt(3), height 2
    # Spacing: dx = sqrt(3), dy = 1.5 for honeycomb
    
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append((k * dx - 1.5 * dx, 0.0, 0.0))
    # Row 1: 4 hexagons, offset
    for k in range(4):
        positions.append((k * dx - dx, dy, 0.0))
    # Row 2: 4 hexagons
    for k in range(4):
        positions.append((k * dx - 1.5 * dx, 2 * dy, 0.0))
    
    for x, y, a in positions:
        params.extend([x, y, a])
    
    return np.array(params)


def construct():
    start_time = time.time()
    
    # Initial guess
    params0 = initial_guess()
    
    # Estimate initial outer side
    # Bounding box of initial config
    xs = params0[0::3]
    ys = params0[1::3]
    max_x = np.max(xs) + 1.0
    min_x = np.min(xs) - 1.0
    max_y = np.max(ys) + 1.0
    min_y = np.min(ys) - 1.0
    width = max_x - min_x
    height = max_y - min_y
    # For a hexagon to contain this, side needs to be at least max(width, height) / sqrt(3) roughly
    outer_side0 = max(width, height) / np.sqrt(3) * 1.2 + 1.0
    
    outer_center0 = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    outer_angle0 = 0.0
    
    best_params = params0.copy()
    best_score = compute_objective(best_params, outer_side0, outer_center0, outer_angle0)
    best_outer_side = outer_side0
    best_center = outer_center0
    best_outer_angle = outer_angle0
    
    # Optimize: minimize outer_side subject to constraints
    # We'll jointly optimize inner positions and outer side
    # Full parameter vector: [inner_params (3N), outer_side, outer_cx, outer_cy, outer_angle]
    
    def full_objective(full_params):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return 1e10
        
        inner_p = full_params[:3*N]
        oside = full_params[3*N]
        ocx = full_params[3*N+1]
        ocy = full_params[3*N+2]
        oangle = full_params[3*N+3]
        
        # Ensure outer side is positive
        if oside < 0.5:
            return 1e10
        
        return compute_objective(inner_p, oside, (ocx, ocy), oangle)
    
    # Start from initial guess
    full_params0 = np.concatenate([params0, [outer_side0, outer_center0[0], outer_center0[1], outer_angle0]])
    
    # Try multiple optimization strategies
    best_result = full_params0.copy()
    best_obj = full_objective(full_params0)
    
    # Nelder-Mead (doesn't need gradients, robust)
    try:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            res = minimize(
                full_objective,
                full_params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
    except Exception:
        pass
    
    # Try L-BFGS-B with finite differences
    try:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            # Bounds: outer side >= 1, positions in reasonable range
            bounds = []
            for i in range(3*N):
                if i % 3 == 2:  # angle
                    bounds.append((-180, 180))
                else:
                    bounds.append((-20, 20))
            bounds.append((1.0, 30.0))   # outer side
            bounds.append((-20, 20))      # ocx
            bounds.append((-20, 20))      # ocy
            bounds.append((-180, 180))    # oangle
            
            res = minimize(
                full_objective,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-8,
                    'gtol': 1e-6
                }
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
    except Exception:
        pass
    
    # Try Powell method
    try:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            res = minimize(
                full_objective,
                best_result,
                method='Powell',
                options={
                    'maxiter': 200,
                    'ftol': 1e-8,
                    'xtol': 1e-6
                }
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
    except Exception:
        pass
    
    # Extract best solution
    inner_params = best_result[:3*N]
    outer_side = best_result[3*N]
    outer_cx = best_result[3*N+1]
    outer_cy = best_result[3*N+2]
    outer_angle = best_result[3*N+3]
    
    # Verify and clean up
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2] % 360
        if a > 180:
            a -= 360
        inner.append([float(x), float(y), float(a)])
    
    # Final verification: check if all hexagons are inside outer
    # If not, scale up outer side slightly
    all_inside = True
    for i in range(N):
        x, y, a = inner[i]
        verts = hexagon_vertices(x, y, 1.0, a)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                all_inside = False
                break
        if not all_inside:
            break
    
    if not all_inside:
        # Increase outer side until everything fits
        for _ in range(100):
            outer_side *= 1.01
            all_inside = True
            for i in range(N):
                x, y, a = inner[i]
                verts = hexagon_vertices(x, y, 1.0, a)
                for v in verts:
                    if not point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                        all_inside = False
                        break
                if not all_inside:
                    break
            if all_inside:
                break
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)
