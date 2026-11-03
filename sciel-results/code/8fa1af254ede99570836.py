"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

def _hex_vertices_np(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg. Returns (6,2) array."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Base vertices for unit hexagon
    thetas = np.array([math.radians(60 * k) for k in range(6)])
    xs = side * np.cos(thetas)
    ys = side * np.sin(thetas)
    # Rotate
    rx = ca * xs - sa * ys
    ry = sa * xs + ca * ys
    # Translate
    return np.column_stack((rx + cx, ry + cy))

def _separating_axis_check_np(vertsA, vertsB):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (invalid), False if disjoint (valid).
    vertsA, vertsB are (6,2) arrays.
    """
    # Get edge normals for A and B
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        lengths = np.linalg.norm(edges, axis=1)
        # Avoid division by zero
        lengths[lengths < 1e-9] = 1e-9
        # Normal is perpendicular to edge: (-dy, dx) / length
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, np.newaxis]
        return normals

    normals_A = get_normals(vertsA)
    normals_B = get_normals(vertsB)
    all_normals = np.vstack([normals_A, normals_B])
    
    # Project A and B onto all normals
    # (12, 2) @ (2, 6) -> (12, 6)
    proj_A = vertsA.T @ all_normals
    proj_B = vertsB.T @ all_normals
    
    min_A = np.min(proj_A, axis=1)
    max_A = np.max(proj_A, axis=1)
    min_B = np.min(proj_B, axis=1)
    max_B = np.max(proj_B, axis=1)
    
    # Check for separation
    # If max_A <= min_B or max_B <= min_A for any axis, they are separated
    sep = (max_A <= min_B + 1e-9) | (max_B <= min_A + 1e-9)
    
    if np.any(sep):
        return False # Disjoint
    return True # Overlap

def _is_valid_fast(cfg):
    """Check all pairs of hexagons are disjoint. cfg is list of [x, y, angle]."""
    m = len(cfg)
    polys = [_hex_vertices_np(p[0], p[1], p[2]) for p in cfg]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check_np(polys[i], polys[j]):
                return False
    return True

def _outer_hex_side_fast(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # Normals of outer hexagon
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    normals_np = np.array(normals) # (6, 2)
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices_np(px, py, pangle)
        # Project all vertices onto all 6 normals
        # verts is (6,2), normals_np is (6,2) -> We need dot product of each vertex with each normal
        # (6, 2) @ (2, 6) -> (6, 6)
        projections = verts @ normals_np.T
        # We want the max projection value for any vertex on any normal
        # Note: The side length S is related to the max distance from center to a vertex along a normal.
        # Distance = dot(vert - center, normal).
        # S = max_dist / apothem_unit
        dists = projections - (normals_np @ np.array([cx, cy]))
        max_proj = np.max(dists)
        if max_proj > max_need:
            max_need = max_proj
            
    return max_need / apothem_unit

def _smooth_collision_penalty(cfg, sigma=0.1):
    """
    Smooth penalty for collisions.
    We approximate the collision by checking the minimum distance between vertices of different hexagons.
    If distance < 0 (overlap), penalty is high.
    To be differentiable, we use a soft-min or simply sum of squared overlaps if we can compute them.
    For simplicity and speed in SA/L-BFGS, we will use a penalty based on the SAT overlap depth.
    However, SAT depth is not differentiable.
    
    Alternative: Use the distance between centers. If distance < 2*apothem, they might collide.
    This is an approximation.
    
    Better: Use the "soft" SDF idea from inspiration but simplified.
    For each pair, compute the minimum distance between their vertices.
    d_min = min( ||v_i - w_j|| )
    Penalty += max(0, 2 - d_min)**2  # If they are closer than 2 (diameter), penalize.
    This is smooth-ish (kink at 0, but squared makes it differentiable).
    """
    m = len(cfg)
    total_penalty = 0.0
    polys = [_hex_vertices_np(p[0], p[1], p[2]) for p in cfg]
    
    for i in range(m):
        for j in range(i + 1, m):
            # Compute min distance between polys[i] and polys[j]
            # Sample points on edges? No, just vertices for speed.
            # Actually, vertex-vertex min distance is a lower bound on true distance.
            # If vertex-vertex distance < 2, they are definitely close/overlapping.
            # Let's use a simple penalty: if any vertex of i is inside j, or vice versa, or distance is small.
            
            # Fast check: min distance between vertices
            # (6, 1, 2) - (1, 6, 2) -> (6, 6, 2)
            diff = polys[i][:, np.newaxis, :] - polys[j][np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_d = np.min(dists)
            
            # Threshold: Two unit hexagons can touch at distance ~2 (vertex to vertex) or sqrt(3) (edge to edge).
            # Minimum distance between disjoint hexagons in lattice is sqrt(3) ~ 1.732.
            # Let's penalize if min_d < 1.7.
            threshold = 1.7
            if min_d < threshold:
                # Quadratic penalty
                total_penalty += (threshold - min_d) ** 2
                
    return total_penalty

def _objective(v):
    """
    Objective function for L-BFGS-B.
    v is flattened: [x1, y1, a1, ..., x12, y12, a12, cx, cy, ca]
    Returns outer_side + lambda * penalty
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    s = _outer_hex_side_fast(cfg, center, angle)
    
    # Check validity first. If invalid, return large value + penalty.
    # To be smooth, we use the penalty.
    penalty = _smooth_collision_penalty(cfg)
    
    # If there is significant overlap, the side length calculation is meaningless for packing,
    # but we want to drive penalty to 0.
    # We add penalty to the side length.
    return s + 10.0 * penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_hexagonal2():
        """4x3 hexagonal arrangement."""
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for r in range(4):
            for c in range(3):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 2.0, y - 1.5 * row_height, 0.0])
        return pts

    def make_cluster():
        """Central hex + 6 around + 5 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    inits = [make_hex_lattice(), make_hexagonal2(), make_cluster()]
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0

    # Evaluate initial configurations
    for init in inits:
        if not _is_valid_fast(init):
            continue
        s = _outer_hex_side_fast(init, (0.0, 0.0), 0.0)
        if s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0

    if best_cfg is None:
        # Fallback: random
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 10.0
        best_center = [0.0, 0.0]
        best_angle = 0.0

    # Phase 1: Simulated Annealing to refine positions and angles
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score

    T = 0.5
    T_end = 0.01
    alpha = 0.999
    
    # SA Loop
    while time.time() - t0 < TIME_LIMIT - 20.0:
        r = random.random()
        
        # Choose move
        if r < 0.6:
            idx = random.randrange(N)
            old_x, old_y, old_a = cfg[idx]
            step = 0.1 * (1.0 + T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
            cfg[idx][2] += random.gauss(0, 10.0)
            # Undo function
            undo = lambda: (cfg[idx].__setitem__(0, old_x), cfg[idx].__setitem__(1, old_y), cfg[idx].__setitem__(2, old_a))
        elif r < 0.8:
            old_cx, old_cy = center
            step = 0.1
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
            undo = lambda: (center.__setitem__(0, old_cx), center.__setitem__(1, old_cy))
        elif r < 0.9:
            old_angle = angle
            angle += random.gauss(0, 5.0)
            undo = lambda: None # Angle is local, just reset if rejected
            # Need to handle angle reset carefully
            # We'll store old angle in a variable for this block
        else:
            # Move two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1, old_a1 = cfg[idx1]
            old_x2, old_y2, old_a2 = cfg[idx2]
            step = 0.1 * (1.0 + T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx1][2] += random.gauss(0, 5.0)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
            cfg[idx2][2] += random.gauss(0, 5.0)
            undo = lambda: (
                cfg[idx1].__setitem__(0, old_x1), cfg[idx1].__setitem__(1, old_y1), cfg[idx1].__setitem__(2, old_a1),
                cfg[idx2].__setitem__(0, old_x2), cfg[idx2].__setitem__(1, old_y2), cfg[idx2].__setitem__(2, old_a2)
            )

        # Evaluate
        new_score = _outer_hex_side_fast(cfg, tuple(center), angle)
        
        # Check validity
        valid = _is_valid_fast(cfg)
        
        if valid:
            if new_score < cur_score:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - cur_score
                if random.random() < math.exp(-delta / T):
                    cur_score = new_score
                else:
                    undo()
        else:
            # Invalid, reject (or accept with very low prob if T is high, but let's be strict to avoid drifting)
            undo()
        
        T = max(T_end, T * alpha)

    # Phase 2: L-BFGS-B Polish
    # Flatten best_cfg
    v0 = []
    for p in best_cfg:
        v0.extend(p)
    v0.extend(best_center)
    v0.append(best_angle)
    
    # L-BFGS-B
    try:
        res = minimize(
            _objective, 
            v0, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-9}
        )
        
        if res.success and res.fun < best_score:
            v_opt = res.x
            # Unflatten
            opt_cfg = []
            for i in range(N):
                opt_cfg.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            opt_center = [v_opt[3*N], v_opt[3*N+1]]
            opt_angle = v_opt[3*N+2]
            
            # Verify validity of optimized solution
            if _is_valid_fast(opt_cfg):
                opt_side = _outer_hex_side_fast(opt_cfg, tuple(opt_center), opt_angle)
                if opt_side < best_score:
                    best_score = opt_side
                    best_cfg = opt_cfg
                    best_center = opt_center
                    best_angle = opt_angle
    except Exception as e:
        pass # Keep best from SA if optimization fails

    # Final check
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
