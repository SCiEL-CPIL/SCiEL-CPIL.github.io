"""Packing unit hexagons inside a regular hexagon - discrete topology search."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 138.0  # safety margin before 150s kill


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.01):
    """Smooth approximation of required side length using log-sum-exp."""
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    projections = []

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = lx * nx + ly * ny
                projections.append(proj)
                projections.append(-proj)

    if not projections:
        return 0.0

    projections = np.array(projections, dtype=np.float64)
    amax = np.max(projections)
    if not np.isfinite(amax):
        return 0.0
    max_abs_proj_smooth = amax + math.log(np.sum(np.exp((projections - amax) / temp))) * temp

    return 2.0 * max_abs_proj_smooth / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def axial_to_xy(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian for hex grid."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y


def generate_polyhexes(n_cells):
    """Generate all polyhexes of n_cells using BFS on axial grid."""
    if n_cells == 1:
        return [{(0, 0)}]

    # Start from all polyhexes of n_cells-1
    # For simplicity, generate from the known small set
    # We'll use a growth approach
    
    all_polyhexes = set()
    
    # Base cases
    if n_cells == 1:
        return [{(0, 0)}]
    if n_cells == 2:
        return [{(0, 0), (1, 0)}]
    if n_cells == 3:
        # Two shapes: straight and bent
        return [{(0, 0), (1, 0), (2, 0)},
                {(0, 0), (1, 0), (0, 1)}]
    if n_cells == 4:
        # Four shapes
        return [{(0, 0), (1, 0), (2, 0), (3, 0)},
                {(0, 0), (1, 0), (2, 0), (1, 1)},
                {(0, 0), (1, 0), (0, 1), (1, 1)},
                {(0, 0), (1, 0), (0, 1), (-1, 1)}]
    if n_cells == 5:
        # Seven shapes
        return [{(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},
                {(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)},
                {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
                {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)},
                {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (0, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0)}]
    if n_cells == 6:
        # Twelve shapes
        return [{(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)},
                {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1)},
                {(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1)},
                {(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)},
                {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1)},
                {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
                {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (0, 2), (1, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (-1, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, 2)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (-2, 1)},
                {(0, 0), (1, 0), (0, 1), (-1, 1), (0, 2), (-1, 2)}]
    
    # For larger n, use growth from n-1
    prev = generate_polyhexes(n_cells - 1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    for ph in prev:
        ph_set = set(ph)
        # Find all boundary cells
        for cell in ph_set:
            for dq, dr in neighbors:
                new_cell = (cell[0] + dq, cell[1] + dr)
                if new_cell not in ph_set:
                    new_ph = ph_set | {new_cell}
                    # Normalize: shift so min q, then min r is at origin
                    min_q = min(c[0] for c in new_ph)
                    min_r = min(c[1] for c in new_ph)
                    normalized = frozenset((c[0] - min_q, c[1] - min_r) for c in new_ph)
                    all_polyhexes.add(normalized)
    
    return [set(ph) for ph in all_polyhexes]


def get_discrete_topologies():
    """
    Generate candidate discrete topologies for 11 hexagons.
    Uses known good configurations from literature and generates variants.
    """
    topologies = []
    
    # The standard hexagonal cluster (radius 2 minus some corners)
    # This is the most compact arrangement
    base_cluster = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    topologies.append(base_cluster)
    
    # Variant: different corner configurations
    # Remove different corner hexagons and add elsewhere
    variants = [
        # Remove (2,0), add (0,-2)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (0, 2), (-1, 2), (0, -2)],
        # Remove (0,2), add (2,-1)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (1, 1), (-1, 2), (2, -1)],
        # Remove (-1,2), add (0,-2)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (1, 1), (0, 2), (0, -2)],
        # More spread out
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (2, -1), (1, 1), (0, 2)],
        # Linear-ish arrangement
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
         (1, 1), (2, 1), (3, 1), (0, 1), (-1, 1)],
        # Two rows
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 0)],
    ]
    topologies.extend(variants)
    
    # Try to generate more using BFS growth from 10-cell polyhexes
    try:
        if time.time() < DEADLINE - 60:
            # Generate 10-cell polyhexes and grow to 11
            polyhexes_10 = generate_polyhexes(10)
            neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            generated_11 = set()
            
            for ph in polyhexes_10[:200]:  # Limit for time
                ph_set = set(ph)
                for cell in ph_set:
                    for dq, dr in neighbors:
                        new_cell = (cell[0] + dq, cell[1] + dr)
                        if new_cell not in ph_set:
                            new_ph = ph_set | {new_cell}
                            # Normalize
                            min_q = min(c[0] for c in new_ph)
                            min_r = min(c[1] for c in new_ph)
                            normalized = tuple(sorted((c[0] - min_q, c[1] - min_r) for c in new_ph))
                            if normalized not in generated_11:
                                generated_11.add(normalized)
                                topologies.append([list(c) for c in normalized])
                                if len(topologies) > 100:
                                    break
                    if len(topologies) > 100:
                        break
                if len(topologies) > 100:
                    break
    except Exception:
        pass
    
    return topologies[:50]  # Limit to 50 topologies


def optimize_topology(topology, time_budget=15.0):
    """
    Optimize a given discrete topology by adjusting positions, rotations, and container.
    Returns (inner, center, side, angle) or None if invalid.
    """
    start = time.time()
    n = len(topology)
    if n != N:
        return None
    
    # Initial positions from axial coordinates
    scale = 1.0 + 1e-6
    initial_positions = []
    for q, r in topology:
        x, y = axial_to_xy(q, r, scale)
        initial_positions.append([x, y, 0.0])
    
    # Compute centroid for initial outer center
    cx = sum(p[0] for p in initial_positions) / n
    cy = sum(p[1] for p in initial_positions) / n
    
    # Parameters: 3*n for inner (x, y, angle) + 3 for outer (cx, cy, angle)
    params = []
    for x, y, a in initial_positions:
        params.extend([x, y, a])
    params.extend([cx, cy, 0.0])
    
    def objective(p):
        inner = []
        for i in range(n):
            inner.append([p[i*3], p[i*3+1], p[i*3+2]])
        ocx = p[n*3]
        ocy = p[n*3+1]
        oang = p[n*3+2]
        
        side = compute_required_side_smooth(inner, (ocx, ocy), oang, temp=0.01)
        
        # Overlap penalty
        penalty = 0.0
        min_dist = SQRT3 - 0.02
        for i in range(n):
            for j in range(i+1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        return side + 300.0 * penalty
    
    # Phase 1: Simulated Annealing
    T0 = 0.5
    T_end = 0.001
    total_steps = 8000
    
    np.random.seed(hash(tuple(topology)) % 10000)
    
    current = params[:]
    current_obj = objective(current)
    best_params = current[:]
    best_obj = current_obj
    best_valid = False
    best_inner = None
    best_center = None
    best_angle = None
    best_side = float('inf')
    
    # Check initial validity
    inner_init = []
    for i in range(n):
        inner_init.append([current[i*3], current[i*3+1], current[i*3+2]])
    if check_all_disjoint(inner_init):
        side_init = compute_required_side(inner_init, (current[n*3], current[n*3+1]), current[n*3+2])
        if check_inside_outer(inner_init, (current[n*3], current[n*3+1]), side_init, current[n*3+2]):
            best_valid = True
            best_inner = inner_init
            best_center = (current[n*3], current[n*3+1])
            best_angle = current[n*3+2]
            best_side = side_init
            best_obj = side_init
    
    for step in range(total_steps):
        if time.time() - start > time_budget:
            break
        
        T = T0 * (T_end / T0) ** (step / total_steps)
        new_p = current[:]
        move_type = np.random.randint(0, 4)
        
        if move_type < 2:
            idx = np.random.randint(0, n)
            base = idx * 3
            new_p[base] += np.random.uniform(-0.1, 0.1)
            new_p[base+1] += np.random.uniform(-0.1, 0.1)
        elif move_type == 2:
            idx = np.random.randint(0, n)
            new_p[idx*3+2] += np.random.uniform(-10, 10)
        else:
            if np.random.random() < 0.7:
                new_p[n*3] += np.random.uniform(-0.1, 0.1)
                new_p[n*3+1] += np.random.uniform(-0.1, 0.1)
            else:
                new_p[n*3+2] += np.random.uniform(-10, 10)
        
        new_obj = objective(new_p)
        
        if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
            current = new_p
            current_obj = new_obj
            
            if step % 500 == 0:
                inner_l = []
                for i in range(n):
                    inner_l.append([current[i*3], current[i*3+1], current[i*3+2]])
                if check_all_disjoint(inner_l):
                    side_l = compute_required_side(inner_l, (current[n*3], current[n*3+1]), current[n*3+2])
                    if check_inside_outer(inner_l, (current[n*3], current[n*3+1]), side_l, current[n*3+2]):
                        if side_l < best_side:
                            best_valid = True
                            best_inner = inner_l
                            best_center = (current[n*3], current[n*3+1])
                            best_angle = current[n*3+2]
                            best_side = side_l
                            best_obj = side_l
                            best_params = current[:]
    
    # Phase 2: Nelder-Mead polish
    if time.time() - start < time_budget - 3:
        try:
            def polish_obj(p):
                inner_l = []
                for i in range(n):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                side = compute_required_side_smooth(inner_l, (p[n*3], p[n*3+1]), p[n*3+2], temp=0.005)
                pen = 0.0
                min_d = SQRT3 - 0.01
                for i in range(n):
                    for j in range(i+1, n):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 1000.0 * pen
            
            res = minimize(polish_obj, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            
            inner_l = []
            for i in range(n):
                inner_l.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            cx_p = res.x[n*3]
            cy_p = res.x[n*3+1]
            ang_p = res.x[n*3+2]
            
            if check_all_disjoint(inner_l):
                side_p = compute_required_side(inner_l, (cx_p, cy_p), ang_p)
                if check_inside_outer(inner_l, (cx_p, cy_p), side_p, ang_p):
                    if side_p < best_side:
                        best_inner = inner_l
                        best_center = (cx_p, cy_p)
                        best_angle = ang_p
                        best_side = side_p
        except Exception:
            pass
    
    if best_valid:
        return best_inner, best_center, best_side, best_angle
    return None


def construct():
    start_time = time.time()
    
    # Generate candidate topologies
    topologies = get_discrete_topologies()
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sort topologies by a heuristic (compactness)
    def topology_compactness(topo):
        xs = [t[0] for t in topo]
        ys = [t[1] for t in topo]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))
    
    topologies.sort(key=topology_compactness)
    
    # Optimize each topology with decreasing time budget
    time_per_topology = max(5.0, (DEADLINE - 20 - (time.time() - start_time)) / max(1, len(topologies)))
    
    for topo in topologies:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Convert to list of tuples for consistency
        topo_list = [tuple(t) for t in topo]
        
        result = optimize_topology(topo_list, time_budget=time_per_topology)
        
        if result is not None:
            inner, center, side, angle = result
            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = center
                best_angle = angle
    
    # Fallback if no valid solution found
    if best_inner is None:
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale_fallback)
            fallback_centers.append((x, y))
        
        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_center, best_side, best_angle):
        # Force a valid fallback
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        scale_fallback = 1.1
        fallback_centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale_fallback)
            fallback_centers.append((x, y))
        
        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    return best_inner, best_center, best_side, best_angle
