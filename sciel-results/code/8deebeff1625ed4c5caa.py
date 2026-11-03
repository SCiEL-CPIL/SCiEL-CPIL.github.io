import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, angle):
        """6 vertices of unit hexagon centered at (cx,cy) with rotation angle."""
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            verts.append((cx + math.cos(a), cy + math.sin(a)))
        return np.array(verts)

    def hex_normals(angle):
        """6 outward normals of unit hexagon with rotation angle."""
        return np.array([
            (math.cos(angle + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(angle + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D_all(pts, angles, outer_center, outer_beta):
        """Compute D (max projection) for all hexagons with individual rotations."""
        outer_normals = hex_normals(outer_beta)
        D = 0.0
        for i in range(len(pts)):
            cx, cy = pts[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, ang)
            verts_centered = verts - np.array(outer_center)
            projs = verts_centered @ outer_normals.T
            D = max(D, float(np.max(np.abs(projs))))
        return D

    def smooth_D_all(pts, angles, outer_center, outer_beta, temp=0.1):
        """Smooth approximation of D for gradient-based optimization."""
        outer_normals = hex_normals(outer_beta)
        all_vals = []
        for i in range(len(pts)):
            cx, cy = pts[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, ang)
            verts_centered = verts - np.array(outer_center)
            projs = verts_centered @ outer_normals.T
            all_vals.append(projs.ravel())
            all_vals.append((-projs).ravel())
        all_vals = np.concatenate(all_vals)
        max_val = float(np.max(all_vals))
        if max_val > 500:
            return 500.0
        shifted = (all_vals - max_val) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        D = temp * log_sum
        return D

    def overlap_penalty(pts, angles, temp=0.05):
        """Penalty for overlapping hexagons using smooth max of signed distances."""
        penalty = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                # Check if hex i and j overlap using vertex-edge distance
                vi = hex_vertices(pts[i][0], pts[i][1], angles[i])
                vj = hex_vertices(pts[j][0], pts[j][1], angles[j])
                # For each vertex of i, check if it's inside j (and vice versa)
                # Use a simpler proxy: center distance vs sum of inradii
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Two unit hexagons with same orientation: inradius = APOTHEM
                # With different orientations, minimum separation is complex
                # Use conservative estimate: if dist < 2*APOTHEM, likely overlapping
                min_sep = 2.0 * APOTHEM - 0.1  # small margin for different rotations
                if dist < min_sep + 0.5:
                    # Smooth penalty: how much they "penetrate"
                    penetration = min_sep - dist
                    if penetration > 0:
                        penalty += temp * math.log(1.0 + math.exp(penetration / temp))
        return penalty

    def objective(params, outer_beta, temp=0.1):
        """Full objective: container size + overlap penalty."""
        n = N
        # params: [x1,y1,a1, x2,y2,a2, ..., xn,yn,an, cx, cy]
        pts = []
        angles = []
        for i in range(n):
            pts.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        outer_center = (params[3*n], params[3*n+1])
        D = smooth_D_all(pts, angles, outer_center, outer_beta, temp)
        S = (D + APOTHEM) / APOTHEM
        pen = overlap_penalty(pts, angles, temp * 0.5)
        return S + 100.0 * pen

    def lattice_relaxation(initial_pts, initial_angles, outer_beta, time_budget):
        """Run lattice relaxation optimization."""
        n = N
        # Initial params
        params0 = []
        for i in range(n):
            params0.extend([initial_pts[i][0], initial_pts[i][1], initial_angles[i]])
        # Initial outer center: mean of all points
        cx0 = float(np.mean([p[0] for p in initial_pts]))
        cy0 = float(np.mean([p[1] for p in initial_pts]))
        params0.extend([cx0, cy0])
        params0 = np.array(params0, dtype=float)

        best_S = float('inf')
        best_params = params0.copy()

        # Multi-start with different temperatures
        temps = [0.5, 0.2, 0.1, 0.05, 0.02]
        for temp in temps:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(lambda p: objective(p, outer_beta, temp),
                               params0, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S and res.fun < 100.0:
                    best_S = float(res.fun)
                    best_params = res.x.copy()
            except Exception:
                pass

        # Extract best configuration
        pts = []
        angles = []
        for i in range(n):
            pts.append((float(best_params[3*i]), float(best_params[3*i+1])))
            angles.append(float(best_params[3*i+2]))
        outer_center = (float(best_params[3*n]), float(best_params[3*n+1]))

        # Verify: compute exact D
        D_exact = compute_D_all(pts, angles, outer_center, outer_beta)
        S_exact = (D_exact + APOTHEM) / APOTHEM

        # Check for overlaps
        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 2.0 * APOTHEM - 0.05:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            return float('inf'), pts, angles, outer_center

        return S_exact, pts, angles, outer_center

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def rigid_optimize(pts, outer_beta):
        """Rigid optimization: all hexagons same rotation, only center and outer_beta vary."""
        best_S = float('inf')
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        # Coarse grid over outer_beta
        n_coarse = 31
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            outer_normals = hex_normals(beta)
            # LP for center
            n_pts = len(pts)
            A_ub = np.zeros((12 * n_pts, 3))
            b_ub = np.zeros(12 * n_pts)
            for k in range(6):
                nx, ny = outer_normals[k]
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
                    D = float(res.x[2])
                    cx, cy = float(res.x[0]), float(res.x[1])
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_S:
                        best_S = S
                        best_center = (cx, cy)
            except Exception:
                pass

        return best_S, best_center

    # --- Generate initial configurations ---
    configs = []

    # Standard hexagonal lattice patterns
    hex_patterns = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
    ]

    for pattern in hex_patterns:
        pts = axial_to_cartesian(pattern)
        configs.append(pts)

    # Random compact configurations
    import random
    rng = random.Random(123)
    for _ in range(30):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        # Generate random cluster
        pts_list = [(0.0, 0.0)]
        used = set()
        used.add((0, 0))
        for _ in range(10):
            # Pick a random existing point and add a neighbor
            base = pts_list[rng.randint(0, len(pts_list) - 1)]
            # Convert to approximate axial
            q_approx = int(round(base[0] / math.sqrt(3) - base[1] / 3.0))
            r_approx = int(round(base[1] / 1.5))
            # Try 6 neighbors
            neighbors = [(q_approx+1,r_approx),(q_approx-1,r_approx),
                        (q_approx,r_approx+1),(q_approx,r_approx-1),
                        (q_approx+1,r_approx-1),(q_approx-1,r_approx+1)]
            rng.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in used and abs(nq + nr) <= 5:
                    used.add((nq, nr))
                    x = math.sqrt(3) * (nq + nr / 2.0)
                    y = 1.5 * nr
                    pts_list.append((x, y))
                    added = True
                    break
            if not added:
                # Try random position
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                pts_list.append((x, y))
        configs.append(np.array(pts_list))

    # --- Main optimization loop ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_angles = None
    best_global_center = None
    best_global_beta = 0.0

    # First: try rigid optimization for each config
    for pts in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, center = rigid_optimize(pts, 0.0)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts.tolist()
            best_global_angles = [0.0] * N
            best_global_center = center
            best_global_beta = 0.0

    # Second: try lattice relaxation on top configs
    # Sort by rigid S
    config_scores = []
    for idx, pts in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, center = rigid_optimize(pts, 0.0)
        config_scores.append((S, idx, pts, center))

    config_scores.sort(key=lambda x: x[0])

    # Try lattice relaxation on best configs
    for S_rigid, idx, pts, center in config_scores[:8]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Try a few outer betas
        for outer_beta in [0.0, math.pi/12, math.pi/6 * 0.3, math.pi/6 * 0.7]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            initial_pts = pts.tolist()
            initial_angles = [0.0] * N
            S_relax, pts_relax, angles_relax, center_relax = lattice_relaxation(
                initial_pts, initial_angles, outer_beta, 10.0)
            if S_relax < best_global_S:
                best_global_S = S_relax
                best_global_pts = pts_relax
                best_global_angles = angles_relax
                best_global_center = center_relax
                best_global_beta = outer_beta

    # --- Final refinement: polish the best configuration ---
    if best_global_pts is not None and best_global_S < float('inf'):
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Try to polish with tighter temperature
            n = N
            params0 = []
            for i in range(n):
                params0.extend([best_global_pts[i][0], best_global_pts[i][1], best_global_angles[i]])
            params0.extend([best_global_center[0], best_global_center[1]])
            params0 = np.array(params0, dtype=float)

            for temp in [0.01, 0.005]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(lambda p: objective(p, best_global_beta, temp),
                                   params0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_global_S:
                        pts_new = []
                        angles_new = []
                        for i in range(n):
                            pts_new.append((float(res.x[3*i]), float(res.x[3*i+1])))
                            angles_new.append(float(res.x[3*i+2]))
                        center_new = (float(res.x[3*n]), float(res.x[3*n+1]))
                        # Verify
                        D_new = compute_D_all(pts_new, angles_new, center_new, best_global_beta)
                        S_new = (D_new + APOTHEM) / APOTHEM
                        # Check overlap
                        ok = True
                        for i in range(n):
                            for j in range(i+1, n):
                                dx = pts_new[i][0] - pts_new[j][0]
                                dy = pts_new[i][1] - pts_new[j][1]
                                if math.sqrt(dx*dx + dy*dy) < 2.0*APOTHEM - 0.05:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok and S_new < best_global_S:
                            best_global_S = S_new
                            best_global_pts = pts_new
                            best_global_angles = angles_new
                            best_global_center = center_new
                except Exception:
                    pass

    # --- Fallback ---
    if best_global_pts is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(hex_patterns[0])
        best_global_pts = pts.tolist()
        best_global_angles = [0.0] * N
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        outer_normals = hex_normals(0.0)
        D = 0.0
        for i in range(N):
            verts = hex_vertices(best_global_pts[i][0], best_global_pts[i][1], 0.0)
            verts_c = verts - np.array(best_global_center)
            projs = verts_c @ outer_normals.T
            D = max(D, float(np.max(np.abs(projs))))
        best_global_S = (D + APOTHEM) / APOTHEM

    # --- Build output ---
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    inner = []
    for i in range(N):
        x, y = best_global_pts[i]
        a = best_global_angles[i]
        inner.append([float(x), float(y), float(math.degrees(a))])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
