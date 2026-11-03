import math
import numpy as np
import time
import random
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 4.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagons_disjoint_axial(axial_list):
        n = len(axial_list)
        for i in range(n):
            for j in range(i + 1, n):
                q1, r1 = axial_list[i]
                q2, r2 = axial_list[j]
                dq = q1 - q2
                dr = r1 - r2
                dist = max(abs(dq), abs(dr), abs(dq + dr))
                if dist < 2:
                    return False
        return True

    def hexagon_vertices(center, side=1.0, angle=0.0):
        """Return 6 vertices of a regular hexagon."""
        cx, cy = center
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3.0
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def point_in_hexagon(px, py, cx, cy, side=1.0, angle=0.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform to local coords
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        ca = math.cos(-angle)
        sa = math.sin(-angle)
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        # Check against 6 half-planes of unit hexagon
        for i in range(6):
            a = i * math.pi / 3.0
            nx = math.cos(a)
            ny = math.sin(a)
            if lx * nx + ly * ny > APOTHEM:
                return False
        return True

    def hexagons_disjoint_cartesian(pts, angle_deg, eps=1e-8):
        """Check if hexagons at pts (with rotation angle_deg) are pairwise disjoint."""
        angle_rad = math.radians(angle_deg)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if any vertex of hex i is inside hex j, or vice versa
                verts_i = hexagon_vertices(pts[i], 1.0, angle_rad)
                verts_j = hexagon_vertices(pts[j], 1.0, angle_rad)
                for vx, vy in verts_i:
                    if point_in_hexagon(vx, vy, pts[j][0], pts[j][1], 1.0, angle_rad):
                        return False
                for vx, vy in verts_j:
                    if point_in_hexagon(vx, vy, pts[i][0], pts[i][1], 1.0, angle_rad):
                        return False
                # Also check if centers are too close (quick reject)
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                if dx * dx + dy * dy < (2 * APOTHEM - eps) ** 2:
                    # Need more careful check - vertices might still be disjoint
                    pass
        return True

    # Generate candidate configurations on axial lattice
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    # Force-directed relaxation: start from axial configs, relax in cartesian space
    def force_directed_relax(pts_initial, n_iter=200, dt=0.01):
        """Relax hexagon centers using repulsive forces to find compact packing."""
        pts = pts_initial.astype(float).copy()
        n = len(pts)
        
        # Repulsive potential: strong repulsion when hexagons overlap
        # Use a smooth approximation
        for it in range(n_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-10:
                        dx = 0.01
                        dy = 0.01
                        dist = 0.0141
                    # Minimum distance for non-overlapping hexagons is 2*APOTHEM
                    min_dist = 2 * APOTHEM
                    if dist < min_dist:
                        # Strong repulsive force
                        overlap = min_dist - dist
                        f = 10.0 * overlap / dist
                        forces[i, 0] += f * dx
                        forces[i, 1] += f * dy
                        forces[j, 0] -= f * dx
                        forces[j, 1] -= f * dy
                    elif dist < min_dist * 1.5:
                        # Weak attractive force to encourage compactness
                        f = -0.1 * (dist - min_dist) / dist
                        forces[i, 0] += f * dx
                        forces[i, 1] += f * dy
                        forces[j, 0] -= f * dx
                        forces[j, 1] -= f * dy
            
            # Update positions
            pts += dt * forces
            # Damping
            dt *= 0.995
        
        return pts

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_orientation_only(pts):
        """1D scalar minimization over outer hexagon orientation beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        n_grid = 97
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1)
                lo = max(0.0, best_beta - 2 * window)
                hi = min(math.pi / 6, best_beta + 2 * window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1) * 0.5
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def optimize_shape(pts):
        """Hybrid optimization for fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        S1, beta1, center1 = optimize_orientation_only(pts)
        if center1 is not None:
            best_S = S1
            best_beta = beta1
            best_center = center1

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.02, 0.01]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                delta_vals = [0.0, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02]
                polish_starts = []
                for db in delta_vals:
                    for dcx in [0.0, 0.002, -0.002, 0.005]:
                        for dcy in [0.0, 0.002, -0.002, 0.005]:
                            polish_starts.append([
                                best_center[0] + dcx,
                                best_center[1] + dcy,
                                best_beta + db
                            ])
                seen = set()
                unique_starts = []
                for p in polish_starts:
                    key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                    if key not in seen:
                        seen.add(key)
                        unique_starts.append(p)

                for init in unique_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Score and sort candidates
    def quick_estimate(pts):
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    # Generate both axial and relaxed candidates
    all_pts_candidates = []
    seen_configs = set()

    for axial in candidates:
        pts = axial_to_cartesian(axial)
        # Add original
        all_pts_candidates.append(pts)
        
        # Add force-directed relaxed version
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pts_relaxed = force_directed_relax(pts, n_iter=150, dt=0.01)
            # Check if relaxed config is valid (hexagons disjoint)
            # Use a quick check: ensure no two centers are too close
            valid = True
            n = len(pts_relaxed)
            min_dist_sq = (2 * APOTHEM * 0.99) ** 2
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pts_relaxed[i, 0] - pts_relaxed[j, 0]
                    dy = pts_relaxed[i, 1] - pts_relaxed[j, 1]
                    if dx * dx + dy * dy < min_dist_sq:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                all_pts_candidates.append(pts_relaxed)

    # Add random force-directed starts
    random.seed(42)
    for _ in range(30):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Start from a random perturbation of a compact config
        base = candidates[random.randint(0, len(candidates) - 1)]
        pts = axial_to_cartesian(base).astype(float)
        # Add random perturbation
        for i in range(len(pts)):
            pts[i, 0] += random.uniform(-0.1, 0.1)
            pts[i, 1] += random.uniform(-0.1, 0.1)
        pts_relaxed = force_directed_relax(pts, n_iter=200, dt=0.01)
        valid = True
        n = len(pts_relaxed)
        min_dist_sq = (2 * APOTHEM * 0.99) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts_relaxed[i, 0] - pts_relaxed[j, 0]
                dy = pts_relaxed[i, 1] - pts_relaxed[j, 1]
                if dx * dx + dy * dy < min_dist_sq:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            all_pts_candidates.append(pts_relaxed)

    # Score and sort
    scored_candidates = []
    for pts in all_pts_candidates:
        est = quick_estimate(pts)
        scored_candidates.append((est, pts))
    scored_candidates.sort(key=lambda x: x[0])

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, (est, pts) in enumerate(scored_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale slightly to avoid boundary issues
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
