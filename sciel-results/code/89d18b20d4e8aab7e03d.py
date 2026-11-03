import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 142.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _sat_overlap(v1, v2):
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def _outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner):
    n = len(inner)
    verts = [_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap(verts[i], verts[j])
            if d < -1e-9:
                return False
    return True

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    # Strategy:
    # 1. Use circle packing (radius = inradius = sqrt(3)/2) to find a dense cluster of 12 points.
    #    This is a simpler, convex-ish problem (mostly) and can be solved with Differential Evolution.
    #    The circle packing ensures that the hexagons (which contain the circles) do not overlap?
    #    NO. Two hexagons can overlap even if their inscribed circles do not.
    #    However, if we pack the hexagons such that their centers are far enough apart, they won't overlap.
    #    The minimum distance between centers of two non-overlapping unit hexagons depends on their relative rotation.
    #    For parallel hexagons, min distance is sqrt(3) (touching edges) or 2 (touching vertices)?
    #    Wait, for regular hexagons of side 1:
    #    - If aligned, touching edges means distance = 2 * apothem = sqrt(3).
    #    - Touching vertices means distance = 2.
    #    - If rotated 30 deg, touching vertex to edge?
    #    
    #    Actually, let's just use the exact SAT check in the objective function for the refinement step.
    #    For the global search (DE), we can use a relaxed constraint:
    #    Distance between centers >= 2 * inradius = sqrt(3).
    #    This guarantees non-overlap for ANY rotation?
    #    Let's check. The inradius is sqrt(3)/2. The circumradius is 1.
    #    If two circles of radius sqrt(3)/2 are disjoint, their centers are at least sqrt(3) apart.
    #    Does distance >= sqrt(3) guarantee hexagons don't overlap?
    #    Consider two hexagons centered at (0,0) and (sqrt(3), 0), both aligned.
    #    Rightmost point of left hex is at (1, 0). Leftmost point of right hex is at (sqrt(3)-1, 0) approx (0.732, 0).
    #    1 > 0.732, so they OVERLAP.
    #    So circle packing is NOT a sufficient condition for non-overlap.
    #    
    #    However, circle packing is a NECESSARY condition. If circles overlap, hexagons overlap.
    #    So we can use circle packing to find candidate clusters, then verify with SAT.
    #    Or better: use a penalty based on SAT in the DE objective, but that's expensive.
    #    
    #    Alternative: Use the fact that the minimum distance for non-overlap is at least sqrt(3) (for aligned).
    #    Actually, the minimum distance between centers of two non-overlapping unit hexagons is sqrt(3) (when touching edges).
    #    Wait, if I rotate one 30 degrees, can they get closer?
    #    The support function of a hexagon is h(u) = max(v_i . u).
    #    Two convex sets A, B are disjoint iff there exists u such that h_A(u) + h_B(-u) < 0 (roughly).
    #    
    #    Let's stick to a hybrid approach:
    #    1. Generate diverse initial configurations (grids, rings, random).
    #    2. For each, run a local optimization (Nelder-Mead or L-BFGS-B) minimizing outer hex side
    #       with a hard penalty for overlap.
    #    
    #    To improve on the current seed (which uses Nelder-Mead), let's try:
    #    - More diverse initializations.
    #    - Using a smoother penalty function (squared overlap) which Nelder-Mead handles okay.
    #    - Maybe try L-BFGS-B with gradient approximation? No, Nelder-Mead is robust for non-smooth.
    #    
    #    Key improvement idea:
    #    The current seed uses a fixed set of initial configs.
    #    Let's add a "random perturbation" step to the best found config to escape local minima.
    #    Also, let's try optimizing the outer rotation angle more aggressively.
    #    
    #    Actually, let's look at the "inspiration" strategies.
    #    Inspiration 2 mentions SDF and L-BFGS-B.
    #    Inspiration 3 mentions Differential Evolution.
    #    
    #    Let's implement a DE-based search for the positions, using a penalty for overlap.
    #    DE is global and can handle non-smooth penalties better than local methods if bounds are set.
    #    Bounds: The hexagons must fit inside a hexagon of side ~5-6. So coordinates in [-5, 5].
    #    Angles: [0, 60).
    
    def objective_de(params):
        """
        Objective for Differential Evolution.
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
        Returns: outer_side + penalty
        """
        if time.time() - t0 > DEADLINE - 10:
            return 1e6
        
        inner = params[:36].reshape(12, 3)
        
        verts_list = []
        for i in range(12):
            cx, cy, rot = inner[i]
            v = _hex_vertices(cx, cy, 1.0, rot)
            verts_list.append(v)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                d = _sat_overlap(verts_list[i], verts_list[j])
                if d < 0:
                    penalty += (-d) ** 2 * 100.0  # Strong penalty
        
        # Calculate outer side needed (centered at origin, rotation 0 for now in DE)
        all_pts = np.vstack(verts_list)
        # Center the configuration
        all_pts[:, 0] -= np.mean(all_pts[:, 0])
        all_pts[:, 1] -= np.mean(all_pts[:, 1])
        
        # Try a few outer rotations to find the best fit for this configuration
        # To save time, just use 0 and 30 for DE
        L0 = _outer_hex_side(all_pts, 0.0)
        L30 = _outer_hex_side(all_pts, 30.0)
        L = min(L0, L30)
        
        return L + penalty

    # Bounds for DE
    # Coordinates: roughly -5 to 5
    # Angles: 0 to 60
    bounds = []
    for i in range(12):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 60.0))  # angle

    # Run DE
    best_de = None
    best_de_val = float('inf')
    
    if time_left() > 60:
        try:
            # Use a smaller population for speed
            de_result = differential_evolution(
                objective_de,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=0.01,
                polish=False,  # We will polish manually
                update_interval=10
            )
            if de_result.success or de_result.fun < best_de_val:
                best_de_val = de_result.fun
                best_de = de_result.x
        except Exception:
            pass

    # Now, refine the best DE result and also try other initial configs with local optimization
    candidates = []
    
    if best_de is not None:
        candidates.append(best_de)
    
    # Add standard configs
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = 2.0 + gap
        spacing_y = np.sqrt(3) + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        return pts

    def make_ring(gap=0.0):
        pts = []
        d1 = np.sqrt(3) + gap
        d2 = 2 * np.sqrt(3) + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        return pts

    configs = [
        make_hex_grid(4, 3, gap=0.1),
        make_hex_grid(3, 4, gap=0.1),
        make_hex_grid(2, 6, gap=0.1),
        make_hex_grid(6, 2, gap=0.1),
        make_ring(gap=0.1),
        make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        make_hex_grid(3, 4, rot_offset=30, gap=0.1),
    ]
    
    # Add random configs
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)

    for cfg in configs:
        if time_left() < 10:
            break
        candidates.append(cfg.flatten())

    best_L = float('inf')
    best_params = None

    # Local refinement function
    def refine(params):
        """
        Refine a candidate configuration using Nelder-Mead.
        params: [x0, y0, a0, ..., x11, y11, a11, ox, oy, oang]
        """
        if time.time() - t0 > DEADLINE - 5:
            return None, float('inf')
        
        def obj(p):
            if time.time() - t0 > DEADLINE - 2:
                return 1e6
            inner = p[:36].reshape(12, 3)
            ox = p[36]
            oy = p[37]
            oang = p[38]
            
            verts_list = []
            for i in range(12):
                cx, cy, rot = inner[i]
                v = _hex_vertices(cx, cy, 1.0, rot)
                verts_list.append(v)
            
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    d = _sat_overlap(verts_list[i], verts_list[j])
                    if d < 0:
                        penalty += (-d) ** 2 * 50.0
            
            all_pts = np.vstack(verts_list)
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            
            L = _outer_hex_side(all_pts, oang)
            return L + penalty
        
        # Ensure params has correct length
        if len(params) == 36:
            params = np.concatenate((params, [0.0, 0.0, 0.0]))
        
        try:
            res = minimize(
                obj, 
                params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            return res.x, res.fun
        except Exception:
            return None, float('inf')

    # Process candidates
    for cand in candidates:
        if time_left() < 15:
            break
        
        # Try different outer rotations for initialization
        for oang in [0.0, 30.0, 15.0]:
            if time_left() < 5:
                break
            
            init_params = np.concatenate((cand[:36], [0.0, 0.0, oang]))
            refined, val = refine(init_params)
            
            if refined is not None:
                # Check validity
                inner = refined[:36].reshape(12, 3)
                valid = _check_disjoint(inner)
                
                if valid:
                    ox = refined[36]
                    oy = refined[37]
                    oang = refined[38]
                    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                    all_pts[:, 0] -= ox
                    all_pts[:, 1] -= oy
                    L = _outer_hex_side(all_pts, oang)
                    
                    if L < best_L:
                        best_L = L
                        best_params = refined.copy()

    if best_params is None:
        # Fallback
        cfg = make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        best_L = _outer_hex_side(np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in cfg]), 0.0)

    # Extract final solution
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    if not _check_disjoint(inner):
        cfg = make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
