"""Packing 12 unit hexagons in a regular hexagon via Grid Search."""
import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin under the 150 s kill
EPS = 1e-6

# Unit hexagon (circumradius 1) vertices for rotation 0
_BASE_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])


def _rot(pts, c, s):
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _min_outer_side(pos, ang):
    """Minimal side length of an outer regular hexagon (any center/rotation)
    containing the unit hexagons at pos (n,2) with angles ang (n,)."""
    verts = []
    for k in range(N):
        th = ang[k]
        verts.append(_rot(_BASE_VERTS, math.cos(th), math.sin(th)) + pos[k])
    A = np.vstack(verts)  # (6n,2)
    best = (None, None, None)
    # Search over rotation phi of the outer hexagon
    for phi in np.arange(0.0, math.pi / 3.0, math.pi / 60.0):
        U = np.array([[math.cos(phi + math.radians(a)), math.sin(phi + math.radians(a))] for a in range(0, 360, 60)])
        hA = A @ U.T  # (6n,6)
        hAmax = hA.max(axis=0)
        hAmin = hA.min(axis=0)
        
        # The optimal center c for a given orientation phi minimizes 
        # max_i (hAmax_i - c.u_i) + max_i (-hAmin_i + c.u_i)
        # This is equivalent to finding the minimum width in all 6 directions.
        # Actually, for a regular hexagon container, the side length S is determined
        # by the support function.
        # Let's use a simpler geometric approach:
        # The minimal enclosing regular hexagon of a set of points.
        # We can approximate the center as the centroid and refine.
        
        cx = (A[:, 0].min() + A[:, 0].max()) / 2.0
        cy = (A[:, 1].min() + A[:, 1].max()) / 2.0
        
        # Refine center slightly
        best_val = float('inf')
        best_cx, best_cy = cx, cy
        for _ in range(2):
            R = max(A[:, 0].max() - A[:, 0].min(), A[:, 1].max() - A[:, 1].min()) / 4.0
            for gcx in np.linspace(cx - R, cx + R, 5):
                for gcy in np.linspace(cy - R, cy + R, 5):
                    c = np.array([gcx, gcy])
                    # Distance from center to each vertex in the 6 hex directions
                    # The side length S is such that the hexagon contains all points.
                    # For a regular hexagon with side S and center c, the constraint is:
                    # max_k | (p_k - c) . u_k | <= S * sqrt(3)/2 ? No.
                    # The distance from center to side is apothem = S * sqrt(3)/2.
                    # The distance from center to vertex is S.
                    # The condition for point p to be inside hexagon centered at c with side S and rot phi:
                    # max_i | (p-c) . u_i | <= S * cos(pi/6) = S * sqrt(3)/2
                    # where u_i are the 6 outward normals.
                    # So S >= max_i | (p-c) . u_i | / (sqrt(3)/2)
                    # We want to minimize S.
                    
                    # Normals are perpendicular to the sides.
                    # If U are vertex directions, normals are rotated by 30 deg.
                    # Let's stick to the previous formulation which was correct:
                    # S = max_i (hAmax_i - c.u_i) + max_i (-hAmin_i + c.u_i) is NOT correct for side length directly.
                    # Let's use the property:
                    # For a regular hexagon with side S, the width in direction u (normal) is S*sqrt(3).
                    # The width of the point set in direction u is hAmax_u - hAmin_u.
                    # We need S*sqrt(3) >= width_u for all 6 normal directions?
                    # No, the hexagon is fixed orientation.
                    
                    # Let's use the vertex support function directly.
                    # If we define the hexagon by its 6 supporting lines with normals n_i:
                    # n_i . x <= c . n_i + S * sqrt(3)/2
                    # Then for all points p_j: n_i . p_j <= c . n_i + S * sqrt(3)/2
                    # => S >= (n_i . p_j - c . n_i) / (sqrt(3)/2)
                    # Also on the other side: -n_i . p_j <= -c . n_i + S * sqrt(3)/2
                    # => S >= (-n_i . p_j + c . n_i) / (sqrt(3)/2)
                    
                    # So S >= max_j | n_i . (p_j - c) | / (sqrt(3)/2)
                    # We need this for all 6 normals n_i.
                    # S(c) = max_i ( max_j | n_i . (p_j - c) | ) / (sqrt(3)/2)
                    
                    # Let's compute this for the current candidate c.
                    # Normals n_i are rotated 30 deg from vertex directions U.
                    nU = np.array([[math.cos(phi + math.radians(a + 30)), math.sin(phi + math.radians(a + 30))] for a in range(0, 360, 60)])
                    
                    # Project all points A onto nU
                    # (6n, 6)
                    proj = (A - c) @ nU.T
                    # max abs projection for each normal
                    max_abs = np.max(np.abs(proj), axis=0)
                    # S is determined by the largest required S
                    S = np.max(max_abs) / (math.sqrt(3) / 2.0)
                    
                    if S < best_val:
                        best_val = S
                        best_cx, best_cy = gcx, gcy
            cx, cy = best_cx, best_cy
            
        if best[0] is None or best_val < best[0] - 1e-12:
            best = (best_val, best_cx, best_cy, phi)
            
    return best  # (side, cx, cy, phi)


def _separating_ok(p1, a1, p2, a2):
    """True if two unit hexagons (center p, angle a) do NOT overlap (touching allowed)."""
    d = p2 - p1
    # Check separation on 6 axes of hex1 and 6 axes of hex2
    for a in (a1, a2):
        c, s = math.cos(a), math.sin(a)
        # Axes are normals to the sides, rotated 30 deg from vertices
        # Or we can just use the 6 vertex directions as axes?
        # Separation axis theorem: check axes perpendicular to edges.
        # For a hexagon, the 6 edge normals are sufficient.
        U = np.array([[math.cos(a + math.radians(k + 30)), math.sin(a + math.radians(k + 30))] for k in range(0, 360, 60)])
        
        R1 = _rot(_BASE_VERTS, math.cos(a1), math.sin(a1))
        R2 = _rot(_BASE_VERTS, math.cos(a2), math.sin(a2))
        
        for u in U:
            proj1 = R1 @ u
            proj2 = R2 @ u
            lo1, hi1 = proj1.min(), proj1.max()
            lo2, hi2 = proj2.min(), proj2.max()
            lo1 += p1 @ u; hi1 += p1 @ u
            lo2 += p2 @ u; hi2 += p2 @ u
            if hi1 < lo2 - EPS or hi2 < lo1 - EPS:
                return True
    return False


def construct():
    t0 = time.time()
    best = None  # (side, pos, ang, cx, cy, phi)

    def try_place(pos, ang):
        nonlocal best
        if time.time() - t0 > DEADLINE:
            return
        # Check feasibility
        for i in range(N):
            for j in range(i + 1, N):
                if not _separating_ok(pos[i], ang[i], pos[j], ang[j]):
                    return
        res = _min_outer_side(pos, ang)
        side, cx, cy, phi = res
        if best is None or side < best[0]:
            best = (side, pos.copy(), ang.copy(), cx, cy, phi)

    # Generate a valid initial configuration: 3 rows of 4 hexagons
    # Hexagon side 1. Distance between centers in a row: 2 (touching)
    # Distance between rows: sqrt(3) * 2 ? No.
    # For flat-top hexagons (rot 0):
    # Width = 2, Height = sqrt(3)
    # Row 0: y=0, x = 0, 2, 4, 6
    # Row 1: y=sqrt(3), x = 1, 3, 5, 7
    # Row 2: y=2*sqrt(3), x = 0, 2, 4, 6
    # This forms a parallelogram-like shape.
    
    def seed_grid():
        pos = []
        ang = []
        # 3 rows, 4 cols
        for r in range(3):
            for c in range(4):
                x = c * 2.0
                if r % 2 == 1:
                    x += 1.0
                y = r * math.sqrt(3)
                pos.append([x, y])
                ang.append(0.0)
        # Center the grid roughly at origin
        pos = np.array(pos)
        pos -= pos.mean(axis=0)
        return pos, np.array(ang)

    pos0, ang0 = seed_grid()
    try_place(pos0, ang0)
    
    # Grid Search Refinement
    # We will perturb positions and angles slightly to find better packings.
    # Since the problem is small, we can do a local search.
    
    rng = np.random.default_rng(42)
    
    # Define a set of perturbations
    # We try to move hexagons closer to each other to reduce the bounding box.
    
    current_pos = pos0.copy()
    current_ang = ang0.copy()
    
    # Local search loop
    step_size = 0.5
    for _ in range(50):  # Max 50 iterations of local search
        if time.time() - t0 > DEADLINE:
            break
            
        improved = False
        # Try moving each hexagon in small steps
        for i in range(N):
            # Try 8 directions + stay
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Check if this move is valid (no overlaps)
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        if not _separating_ok(new_pos[i], current_ang[i], new_pos[j], current_ang[j]):
                            valid = False
                            break
                            
                    if valid:
                        # Check if it improves the score
                        res = _min_outer_side(new_pos, current_ang)
                        if best is None or res[0] < best[0] - 1e-6:
                            current_pos = new_pos
                            best = (res[0], new_pos.copy(), current_ang.copy(), res[1], res[2], res[3])
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try rotating a hexagon
            for i in range(N):
                for d_ang in [-0.1, 0.1]:
                    new_ang = current_ang.copy()
                    new_ang[i] += d_ang
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        if not _separating_ok(current_pos[i], new_ang[i], current_pos[j], current_ang[j]):
                            valid = False
                            break
                    if valid:
                        res = _min_outer_side(current_pos, new_ang)
                        if best is None or res[0] < best[0] - 1e-6:
                            current_ang = new_ang
                            best = (res[0], current_pos.copy(), new_ang.copy(), res[1], res[2], res[3])
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Shrink step size
                step_size *= 0.8
                if step_size < 0.01:
                    break

    if best is None:
        # Fallback
        pos, ang = seed_grid()
        side, cx, cy, phi = _min_outer_side(pos, ang)
        best = (side, pos, ang, cx, cy, phi)

    side, pos, ang, cx, cy, phi = best
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(side), float(math.degrees(phi))
