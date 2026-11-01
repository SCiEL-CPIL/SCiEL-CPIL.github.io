import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))

    def get_hex_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon."""
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append((vx, vy))
        return verts

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def sat_overlap_check(v1, v2, eps=1e-9):
        """
        Check if two convex polygons (defined by their vertices) overlap using SAT.
        v1, v2: lists of (x, y) tuples.
        Returns True if they overlap (including touching).
        """
        axes = []
        
        # Get axes from edges of v1
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))
        
        # Get axes from edges of v2
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))
        
        for axis in axes:
            min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
            max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
            min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
            max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
            
            # Check for gap
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

    def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
        """Check if all inner hexagons are inside the outer hexagon."""
        ox, oy = outer_center
        outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
        
        for (ix, iy, iangle) in inner:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            for (vx, vy) in verts:
                inside = True
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    ex = p2[0] - p1[0]
                    ey = p2[1] - p1[1]
                    px = vx - p1[0]
                    py = vy - p1[1]
                    cross = ex * py - ey * px
                    if cross < -eps:
                        inside = False
                        break
                if not inside:
                    return False
        return True

    def compute_required_side(inner, outer_center, outer_angle_deg):
        """
        Compute the minimum outer side length to contain all inner hexagons.
        """
        if not inner:
            return 0.0
        
        ox, oy = outer_center
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
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
                
                for nx, ny in normals:
                    proj = lx * nx + ly * ny
                    if proj > max_projection:
                        max_projection = proj
        
        return 2.0 * max_projection / SQRT3

    def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
        """
        Smooth approximation using log-sum-exp for the max projection.
        """
        if not inner:
            return 0.0
        
        ox, oy = outer_center
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
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
                
                for nx, ny in normals:
                    proj = lx * nx + ly * ny
                    projections.append(proj)
        
        if not projections:
            return 0.0
        
        max_proj = max(projections)
        if max_proj < -1000:
            return 0.0
        
        scaled = [beta * (p - max_proj) for p in projections]
        sum_exp = sum(math.exp(s) for s in scaled)
        smooth_max = max_proj + math.log(sum_exp) / beta
        
        return 2.0 * smooth_max / SQRT3

    def objective_smooth(params, outer_center, outer_angle_deg):
        """
        Objective function for scipy.optimize.minimize.
        params: flat array of [x1, y1, a1, x2, y2, a2, ...]
        """
        inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner.append((x, y, a))
        
        # Smooth size
        S = compute_required_side_smooth(inner, outer_center, outer_angle_deg)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            for j in range(i + 1, N):
                vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                # Approximate overlap depth for penalty
                # Using SAT to check overlap, but we need a continuous measure.
                # Simple penalty: if overlapping, add a large constant.
                # To make it smooth, we can use the minimum distance between polygons if they don't overlap,
                # but SAT gives boolean. 
                # For now, a hard penalty is sufficient for L-BFGS to push them apart if they start close.
                if sat_overlap_check(vi, vj, 0.0):
                    penalty += 100.0
        
        return S + penalty

    def local_search(inner, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
        cur_valid = check_all_disjoint(cur_inner) and check_inside(cur_inner, (ocx, ocy), cur_S, oangle)
        
        if not cur_valid:
            # Try to fix invalid config
            for _ in range(50):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.5)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.5)
                cur_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
                cur_valid = check_all_disjoint(cur_inner) and check_inside(cur_inner, (ocx, ocy), cur_S, oangle)
                if cur_valid:
                    break
            if not cur_valid:
                return
        
        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
        
        for it in range(2000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.15)
            cur_inner[idx][1] += rng.gauss(0, 0.15)
            if rng.random() < 0.03:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            new_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
            new_valid = check_all_disjoint(cur_inner) and check_inside(cur_inner, (ocx, ocy), new_S, oangle)
            
            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
            elif new_valid:
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    # Strategy 1: Gradient-based optimization from multiple starts
    initial_configs = []
    
    # Start 1: Hexagonal lattice
    sp = SQRT3
    hex_config = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    for i in [0, 2, 4]:
        theta = math.radians(60 * i + 30)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    for i in [1, 3, 5]:
        theta = math.radians(60 * i)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    while len(hex_config) < N:
        hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
    initial_configs.append(hex_config[:N])

    # Start 2: Random perturbation of lattice
    rng = random.Random(42)
    rand_config = []
    for i in range(N):
        rand_config.append([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0, 180)])
    initial_configs.append(rand_config)

    # Start 3: Linear array (fallback)
    line_config = []
    for i in range(N):
        line_config.append([(i - N // 2) * sp, 0.0, 0.0])
    initial_configs.append(line_config)

    for cfg in initial_configs:
        if time.time() > deadline:
            break
        
        # Flatten for scipy
        params0 = []
        for x, y, a in cfg:
            params0.extend([x, y, a])
        
        # Run L-BFGS-B
        try:
            res = minimize(objective_smooth, params0, args=((0.0, 0.0), 0.0), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            # Reconstruct inner
            new_inner = []
            for i in range(N):
                new_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            
            # Check validity and refine with local search
            local_search(new_inner, 0.0, 0.0, 0.0, hash(str(res.x)) % 10000)
            
        except Exception:
            pass

    # Strategy 2: Local search on best found so far
    if best_valid and best_inner:
        # Run local search again on the best found to squeeze out more
        local_search(best_inner, best_center[0], best_center[1], best_angle, 12345)

    # Fallback if nothing valid found
    if not best_valid or best_inner is None:
        sp = SQRT3
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_required_side(fallback_inner, (0.0, 0.0), 0.0)
        for _ in range(100):
            if check_all_disjoint(fallback_inner) and check_inside(fallback_inner, (0.0, 0.0), S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_required_side(fallback_inner, (0.0, 0.0), 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
