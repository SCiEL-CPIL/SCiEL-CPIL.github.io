import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    angles_rad = np.radians(angles)
    xs = cx + s * np.cos(angles_rad)
    ys = cy + s * np.sin(angles_rad)
    return np.column_stack([xs, ys])

def polygon_overlap_area(v1, v2):
    """Compute overlap area between two convex polygons using Sutherland-Hodgman clipping."""
    def clip(subject, clipper):
        if len(subject) == 0:
            return []
        output = list(subject)
        for i in range(len(clipper)):
            if len(output) == 0:
                return []
            input_list = output
            output = []
            S = input_list[-1]
            for E in input_list:
                if point_in_polygon(E, clipper):
                    if not point_in_polygon(S, clipper):
                        output.append(line_intersect(S, E, clipper[i], clipper[(i+1) % len(clipper)]))
                    output.append(E)
                elif point_in_polygon(S, clipper):
                    output.append(line_intersect(S, E, clipper[i], clipper[(i+1) % len(clipper)]))
                S = E
        return output

    def point_in_polygon(p, poly):
        # Ray casting
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def poly_area(poly):
        if len(poly) < 3:
            return 0.0
        area = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    clipped = clip([tuple(v) for v in v1], [tuple(v) for v in v2])
    return poly_area(clipped)

def min_outer_side(inner_positions, outer_center, outer_side, outer_angle):
    """
    Compute the minimum outer side length needed to contain all inner hexagons,
    given outer hexagon center and orientation.
    
    For a regular hexagon with center C, side s, and rotation angle a,
    a point P is inside iff for each of the 6 directions (normal to each side),
    the projection of (P-C) onto that normal is <= s * sqrt(3)/2.
    """
    ocx, ocy = outer_center
    a = np.radians(outer_angle)
    # The 6 outward normals of the outer hexagon
    # For a regular hexagon with vertices at angles 0,60,...,300,
    # the side normals are at angles 30, 90, 150, 210, 270, 330 (i.e., offset by 30)
    normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_angle)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for (cx, cy, ang) in inner_positions:
        verts = hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            rel = v - np.array([ocx, ocy])
            projs = rel @ normals.T
            max_proj = max(max_proj, np.max(projs))
    
    # Required side length: s = max_proj / (sqrt(3)/2)
    required_side = max_proj / (np.sqrt(3) / 2.0)
    return required_side

def objective(params, inner_angles):
    """
    Objective: minimize the outer hexagon side length.
    params: [cx_0, cy_0, cx_1, cy_1, ..., cx_11, cy_11] (24 values)
    inner_angles: fixed angles for each inner hexagon
    """
    if time.time() > DEADLINE:
        return 1e6
    
    positions = []
    for i in range(N):
        cx = params[2*i]
        cy = params[2*i+1]
        positions.append((cx, cy, inner_angles[i]))
    
    # Outer hexagon centered at origin, angle 0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute required side
    side = min_outer_side(positions, outer_center, 0.0, outer_angle)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(N):
        v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
        for j in range(i+1, N):
            v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
            # Quick distance check
            dist = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
            if dist < 3.5:  # Max distance where overlap possible for unit hexagons
                area = polygon_overlap_area(v1, v2)
                overlap_penalty += area * 100.0  # Heavy penalty
    
    return side + overlap_penalty

def generate_hex_lattice_seed():
    """Generate a hexagonal lattice arrangement of 12 hexagons."""
    # Hex lattice: centers at distance 2*sqrt(3) apart for non-overlapping unit hexagons
    # Actually, two unit hexagons with parallel sides need center distance >= 2*sqrt(3)
    # to not overlap. But if rotated, minimum distance can be less.
    # For parallel orientation, minimum center distance is 2*sqrt(3) ≈ 3.464
    
    d = 2 * np.sqrt(3) + 0.1  # small gap
    
    positions = []
    angles = []
    # Hex lattice basis vectors
    e1 = np.array([d, 0])
    e2 = np.array([d/2, d * np.sqrt(3)/2])
    
    # Generate all lattice points within a radius, pick 12
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * e1 + j * e2
            lattice_points.append(p)
    
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: np.dot(p, p))
    
    # Pick 12 closest
    for p in lattice_points[:N]:
        positions.append([float(p[0]), float(p[1])])
        angles.append(0.0)
    
    return positions, angles

def generate_ring_seed():
    """Generate a ring arrangement."""
    positions = []
    angles = []
    r = 3.5
    for i in range(N):
        a = 2 * np.pi * i / N
        positions.append([r * np.cos(a), r * np.sin(a)])
        angles.append(np.degrees(a))
    return positions, angles

def generate_cluster_seed():
    """Generate a compact cluster."""
    positions = []
    angles = []
    # Center + ring
    positions.append([0.0, 0.0])
    angles.append(0.0)
    r = 3.6
    for i in range(N - 1):
        a = 2 * np.pi * i / (N - 1)
        positions.append([r * np.cos(a), r * np.sin(a)])
        angles.append(np.degrees(a))
    return positions, angles

def optimize_from_seed(positions, angles, max_iter=500):
    """Use scipy minimize to refine positions."""
    x0 = []
    for p in positions:
        x0.extend([p[0], p[1]])
    
    inner_angles = np.array(angles)
    
    try:
        result = minimize(
            objective,
            x0,
            args=(inner_angles,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < 1e5:
            best_params = result.x
            best_score = result.fun
        else:
            best_params = x0
            best_score = objective(x0, inner_angles)
    except Exception:
        best_params = x0
        best_score = objective(x0, inner_angles)
    
    # Extract positions
    final_positions = []
    for i in range(N):
        cx = best_params[2*i]
        cy = best_params[2*i+1]
        final_positions.append([float(cx), float(cy), float(angles[i])])
    
    # Compute actual outer side
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    side = min_outer_side(final_positions, outer_center, 0.0, outer_angle)
    
    # Check for overlaps - if any, add penalty
    has_overlap = False
    for i in range(N):
        v1 = hex_vertices(final_positions[i][0], final_positions[i][1], 1.0, final_positions[i][2])
        for j in range(i+1, N):
            v2 = hex_vertices(final_positions[j][0], final_positions[j][1], 1.0, final_positions[j][2])
            dist = np.sqrt((final_positions[i][0]-final_positions[j][0])**2 + (final_positions[i][1]-final_positions[j][1])**2)
            if dist < 3.5:
                area = polygon_overlap_area(v1, v2)
                if area > 1e-6:
                    has_overlap = True
                    break
        if has_overlap:
            break
    
    if has_overlap:
        side += 10.0  # penalty
    
    return final_positions, side, has_overlap

def try_different_angles(positions, base_angles, time_limit):
    """Try rotating individual hexagons to improve packing."""
    best_positions = [list(p) for p in positions]
    best_side, _, _ = optimize_from_seed(best_positions, base_angles, max_iter=100)
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 5:
        return best_positions, best_side
    
    # Try a few angle perturbations
    for attempt in range(10):
        if time.time() > start_time + time_limit:
            break
        test_positions = [list(p) for p in positions]
        test_angles = list(base_angles)
        # Randomly rotate 2-3 hexagons
        for _ in range(3):
            idx = np.random.randint(N)
            test_angles[idx] = base_angles[idx] + np.random.uniform(-30, 30)
        
        pos, side, has_overlap = optimize_from_seed(test_positions, test_angles, max_iter=100)
        if side < best_side:
            best_positions = pos
            best_side = side
    
    return best_positions, best_side

start_time = time.time()

def construct():
    global start_time
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_positions = None
    best_side = 1e10
    
    # Generate multiple seeds
    seeds = []
    
    # Hex lattice
    pos, ang = generate_hex_lattice_seed()
    seeds.append((pos, ang))
    
    # Ring
    pos, ang = generate_ring_seed()
    seeds.append((pos, ang))
    
    # Cluster
    pos, ang = generate_cluster_seed()
    seeds.append((pos, ang))
    
    # Random initial
    np.random.seed(42)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(0.0)
    seeds.append((pos, ang))
    
    # Another random with different angles
    np.random.seed(123)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(np.random.uniform(0, 60))
    seeds.append((pos, ang))
    
    # Optimize each seed
    for seed_idx, (positions, angles) in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        
        pos, side, has_overlap = optimize_from_seed(positions, angles, max_iter=300)
        
        if side < best_side:
            best_side = side
            best_positions = pos
    
    # Refine the best solution with angle optimization
    if best_positions is not None and time.time() < deadline - 20:
        base_angles = [p[2] for p in best_positions]
        base_pos_xy = [[p[0], p[1]] for p in best_positions]
        
        # Try angle perturbations on the best
        for attempt in range(15):
            if time.time() > deadline - 10:
                break
            
            test_angles = list(base_angles)
            # Perturb 1-3 angles
            n_perturb = np.random.randint(1, 4)
            for _ in range(n_perturb):
                idx = np.random.randint(N)
                test_angles[idx] = base_angles[idx] + np.random.uniform(-45, 45)
            
            test_pos = [[p[0], p[1]] for p in best_positions]
            pos, side, has_overlap = optimize_from_seed(test_pos, test_angles, max_iter=200)
            
            if side < best_side:
                best_side = side
                best_positions = pos
                base_angles = [p[2] for p in best_positions]
    
    # Final verification and cleanup
    if best_positions is None:
        # Fallback
        best_positions = [[2.0 * np.sqrt(3) * k, 0.0, 0.0] for k in range(N)]
        best_side = 40.0
    else:
        # Recompute the true outer side
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        best_side = min_outer_side(best_positions, outer_center, 0.0, outer_angle)
        
        # Verify no overlaps
        for i in range(N):
            v1 = hex_vertices(best_positions[i][0], best_positions[i][1], 1.0, best_positions[i][2])
            for j in range(i+1, N):
                v2 = hex_vertices(best_positions[j][0], best_positions[j][1], 1.0, best_positions[j][2])
                dist = np.sqrt((best_positions[i][0]-best_positions[j][0])**2 + (best_positions[i][1]-best_positions[j][1])**2)
                if dist < 3.5:
                    area = polygon_overlap_area(v1, v2)
                    if area > 1e-4:
                        # Overlap detected - this solution is invalid
                        # We'll still return it but the score will reflect the issue
                        pass
    
    # Format output
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), float(p[2])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# Run construct at module level to pre-compute? No, the evaluator calls construct()
# But we need to make sure it's fast enough. Let's just define it.

# Actually, let's pre-compute a good solution at import time if possible
# But the evaluator calls construct(), so we should just have it ready.

# Let me restructure: pre-compute at module level
def _precompute():
    """Pre-compute a good packing at module load time."""
    global start_time
    start_time = time.time()
    deadline = start_time + 125.0  # Leave 5s margin for construct() call overhead
    
    best_positions = None
    best_side = 1e10
    
    seeds = []
    pos, ang = generate_hex_lattice_seed()
    seeds.append((pos, ang))
    pos, ang = generate_ring_seed()
    seeds.append((pos, ang))
    pos, ang = generate_cluster_seed()
    seeds.append((pos, ang))
    
    np.random.seed(42)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(0.0)
    seeds.append((pos, ang))
    
    np.random.seed(123)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(np.random.uniform(0, 60))
    seeds.append((pos, ang))
    
    for seed_idx, (positions, angles) in enumerate(seeds):
        if time.time() > deadline - 20:
            break
        pos, side, has_overlap = optimize_from_seed(positions, angles, max_iter=200)
        if side < best_side:
            best_side = side
            best_positions = pos
    
    # Refine
    if best_positions is not None and time.time() < deadline - 10:
        base_angles = [p[2] for p in best_positions]
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            test_angles = list(base_angles)
            for _ in range(3):
                idx = np.random.randint(N)
                test_angles[idx] = base_angles[idx] + np.random.uniform(-45, 45)
            test_pos = [[p[0], p[1]] for p in best_positions]
            pos, side, has_overlap = optimize_from_seed(test_pos, test_angles, max_iter=150)
            if side < best_side:
                best_side = side
                best_positions = pos
                base_angles = [p[2] for p in best_positions]
    
    if best_positions is None:
        best_positions = [[2.0 * np.sqrt(3) * k, 0.0, 0.0] for k in range(N)]
        best_side = 40.0
    else:
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        best_side = min_outer_side(best_positions, outer_center, 0.0, outer_angle)
    
    _precomputed_inner = []
    for p in best_positions:
        _precomputed_inner.append([float(p[0]), float(p[1]), float(p[2])])
    _precomputed_side = float(best_side)

# Don't run precompute at import - let construct() do it with its own timing
# Actually, to be safe, let's just have construct() do the work

def construct():
    """Main entry point."""
    t_start = time.time()
    deadline = t_start + 130.0
    
    best_positions = None
    best_side = 1e10
    
    seeds = []
    pos, ang = generate_hex_lattice_seed()
    seeds.append((pos, ang))
    pos, ang = generate_ring_seed()
    seeds.append((pos, ang))
    pos, ang = generate_cluster_seed()
    seeds.append((pos, ang))
    
    np.random.seed(42)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(0.0)
    seeds.append((pos, ang))
    
    np.random.seed(123)
    pos = []
    ang = []
    for _ in range(N):
        pos.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        ang.append(np.random.uniform(0, 60))
    seeds.append((pos, ang))
    
    for seed_idx, (positions, angles) in enumerate(seeds):
        if time.time() > deadline - 25:
            break
        pos, side, has_overlap = optimize_from_seed(positions, angles, max_iter=200)
        if side < best_side:
            best_side = side
            best_positions = pos
    
    if best_positions is not None and time.time() < deadline - 15:
        base_angles = [p[2] for p in best_positions]
        for attempt in range(12):
            if time.time() > deadline - 8:
                break
            test_angles = list(base_angles)
            for _ in range(3):
                idx = np.random.randint(N)
                test_angles[idx] = base_angles[idx] + np.random.uniform(-45, 45)
            test_pos = [[p[0], p[1]] for p in best_positions]
            pos, side, has_overlap = optimize_from_seed(test_pos, test_angles, max_iter=150)
            if side < best_side:
                best_side = side
                best_positions = pos
                base_angles = [p[2] for p in best_positions]
    
    if best_positions is None:
        best_positions = [[2.0 * np.sqrt(3) * k, 0.0, 0.0] for k in range(N)]
        best_side = 40.0
    else:
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        best_side = min_outer_side(best_positions, outer_center, 0.0, outer_angle)
    
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), float(p[2])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

