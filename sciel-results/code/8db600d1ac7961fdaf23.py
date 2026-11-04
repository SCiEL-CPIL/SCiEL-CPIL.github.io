import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
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

    # Generate candidate configurations - diverse polyhex shapes for 11 hexagons
    candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(2,1),(-2,1),(0,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,1),(-1,1),(0,3),(3,1)],
        [(0,0),(1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,3),(3,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
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
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(0,-2),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,-1),(2,1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(0,1),(-1,0)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(1,-2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,3),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(-1,-1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(1,3),(2,3),(0,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(0,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(3,0)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,1),(2,-1),(-2,1)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(2,3),(3,3),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,3),(3,1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(1,-1),(-1,-1)],
        [(0,0),(1,0),(2,0),(-1,0),(-2,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(-2,1),(2,-1),(-2,-1)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,-1),(1,-1),(2,-1)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(2,0),(-2,0)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(-2,1),(1,-1),(-1,-1)],
        [(0,0),(1,0),(2,0),(-1,0),(-2,0),(0,1),(1,1),(-1,1),(2,1),(-2,1),(0,-1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(2,1),(-2,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(0,-1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,-1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(0,2)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(0,2),(0,3)],
    ]

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

    def fast_min_enclosing_hex(pts):
        """Fast iterative method for minimum enclosing hexagon."""
        if len(pts) < 2:
            cx = float(pts[0, 0]) if len(pts) > 0 else 0.0
            cy = float(pts[0, 1]) if len(pts) > 0 else 0.0
            return (cx, cy), 0.0

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        best_cx, best_cy, best_D = cx, cy, float('inf')

        for beta in [0.0, math.pi/12, math.pi/6]:
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])

            for it in range(200):
                pts_centered = pts - np.array([cx, cy])
                projs = pts_centered @ normals.T
                abs_projs = np.abs(projs)
                D = np.max(abs_projs)

                if D < best_D - 1e-15:
                    best_D = D
                    best_cx, best_cy = cx, cy

                active_normals = []
                for k in range(6):
                    if np.max(projs[:, k]) >= D - 1e-12:
                        active_normals.append(normals[k])
                    if np.min(projs[:, k]) <= -D + 1e-12:
                        active_normals.append(-normals[k])

                if not active_normals:
                    break

                active_arr = np.array(active_normals)
                subgrad_x = np.mean(active_arr[:, 0])
                subgrad_y = np.mean(active_arr[:, 1])

                step = min(0.1, D / (10 * (abs(subgrad_x) + abs(subgrad_y) + 1e-10)))
                new_cx = cx + step * subgrad_x
                new_cy = cy + step * subgrad_y

                new_pts_centered = pts - np.array([new_cx, new_cy])
                new_projs = new_pts_centered @ normals.T
                new_D = np.max(np.abs(new_projs))

                if new_D < D - 1e-14:
                    cx, cy = new_cx, new_cy
                else:
                    found = False
                    for factor in [0.5, 0.25, 0.1]:
                        s = step * factor
                        tcx, tcy = cx + s * subgrad_x, cy + s * subgrad_y
                        tpc = pts - np.array([tcx, tcy])
                        tp = tpc @ normals.T
                        tD = np.max(np.abs(tp))
                        if tD < D - 1e-14:
                            cx, cy = tcx, tcy
                            found = True
                            break
                    if not found:
                        break

        return (best_cx, best_cy), best_D

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

    def hexagon_vertices(cx, cy, angle_deg):
        """Return 6 vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + k * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
        return verts

    def poly_overlap(v1, v2, tol=1e-7):
        """Check if two convex polygons overlap (interior intersection)."""
        xs1 = [v[0] for v in v1]; ys1 = [v[1] for v in v1]
        xs2 = [v[0] for v in v2]; ys2 = [v[1] for v in v2]
        if max(xs1) < min(xs2) + tol or max(xs2) < min(xs1) + tol:
            return False
        if max(ys1) < min(ys2) + tol or max(ys2) < min(ys1) + tol:
            return False
        for poly in (v1, v2):
            n = len(poly)
            for i in range(n):
                ax1, ay1 = poly[i]
                ax2, ay2 = poly[(i+1) % n]
                dx = ax2 - ax1
                dy = ay2 - ay1
                nx, ny = -dy, dx
                norm = math.sqrt(nx*nx + ny*ny)
                if norm < 1e-12:
                    continue
                nx /= norm; ny /= norm
                proj1 = [v[0]*nx + v[1]*ny for v in v1]
                proj2 = [v[0]*nx + v[1]*ny for v in v2]
                if max(proj1) < min(proj2) + tol or max(proj2) < min(proj1) + tol:
                    return False
        return True

    def check_all_disjoint(pts, angles_deg, tol=1e-7):
        """Check all pairs of hexagons are disjoint."""
        n = len(pts)
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(pts[i,0], pts[i,1], angles_deg[i]))
        for i in range(n):
            for j in range(i+1, n):
                if poly_overlap(verts_list[i], verts_list[j], tol):
                    return False
        return True

    def lattice_deform(pts, M, beta, center):
        """Apply affine transformation M to points relative to their centroid."""
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid
        transformed = shifted @ M.T + centroid
        return transformed

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts.copy()

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, pts

        # Stage 0: Fast initial estimate
        fast_center, fast_D = fast_min_enclosing_hex(pts)
        if fast_D < float('inf'):
            fast_S = (fast_D + APOTHEM) / APOTHEM
            best_S = min(best_S, fast_S)
            if best_center is None or fast_S < best_S:
                best_center = fast_center

        # Stage 1: Finer grid search over beta in [0, pi/6]
        n_grid = 121
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
            return float('inf'), 0.0, None, pts

        # Stage 2: Scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Stage 2.5: Lattice deformation optimization
        # Try affine transformations of the lattice to squeeze into container
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 10:
            centroid = np.mean(pts, axis=0)
            
            # Parameterize M as [a, b; c, d] - 4 parameters
            # Start with identity and try small deformations
            best_deform_S = best_S
            best_deform_M = np.eye(2)
            
            # Try various deformation patterns
            deform_patterns = [
                # Anisotropic scaling
                [[1.0+s, 0], [0, 1.0-s]] for s in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            ] + [
                [[1.0-s, 0], [0, 1.0+s]] for s in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            ] + [
                # Shear
                [[1.0, s], [0, 1.0]] for s in [0.05, 0.1, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2]
            ] + [
                [[1.0, 0], [s, 1.0]] for s in [0.05, 0.1, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2]
            ] + [
                # Combined scale + shear
                [[1.0+s, t], [0, 1.0-s]] for s in [0.1, 0.2] for t in [0.1, -0.1]
            ] + [
                [[1.0-s, t], [0, 1.0+s]] for s in [0.1, 0.2] for t in [0.1, -0.1]
            ] + [
                # Rotation + scale
                [[math.cos(th)*(1.0+s), -math.sin(th)*(1.0+s)], [math.sin(th)*(1.0-s), math.cos(th)*(1.0-s)]]
                for th in [0.1, 0.2, -0.1, -0.2] for s in [0.05, 0.1]
            ]
            
            for M_pattern in deform_patterns:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                M = np.array(M_pattern, dtype=float)
                deformed_pts = lattice_deform(pts, M, best_beta, best_center)
                
                # Check if deformed hexagons are still disjoint
                angles_deg = [math.degrees(best_beta)] * len(deformed_pts)
                if not check_all_disjoint(deformed_pts, angles_deg, tol=1e-6):
                    continue
                
                # Optimize container for deformed configuration
                S_def, center_def, D_def = calc_S_for_beta(deformed_pts, best_beta)
                if S_def < float('inf') and S_def < best_deform_S:
                    # Try to further optimize beta for deformed config
                    try:
                        res_def = minimize_scalar(lambda b: calc_S_for_beta(deformed_pts, b)[0],
                                                 bounds=(0, math.pi / 6), method='bounded',
                                                 options={'maxiter': 300, 'xatol': 1e-10})
                        if res_def.fun < S_def:
                            S_def = res_def.fun
                            best_beta_def = res_def.x
                            _, center_def, _ = calc_S_for_beta(deformed_pts, best_beta_def)
                            if S_def < best_deform_S:
                                best_deform_S = S_def
                                best_deform_M = M
                                best_pts = deformed_pts
                                best_beta = best_beta_def
                                best_center = center_def
                    except Exception:
                        if S_def < best_deform_S:
                            best_deform_S = S_def
                            best_deform_M = M
                            best_pts = deformed_pts
                            best_center = center_def

        # Stage 3: Joint (cx, cy, beta) optimization with smooth surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(best_pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03, 0.0, math.pi / 12, best_beta + 0.06, best_beta - 0.06]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(best_pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 4: High-precision polish with exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(best_pts, cx, cy, beta)

                delta = 0.005
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + delta],
                    [best_center[0], best_center[1], best_beta - delta],
                    [best_center[0] + delta, best_center[1], best_beta],
                    [best_center[0] - delta, best_center[1], best_beta],
                    [best_center[0], best_center[1] + delta, best_beta],
                    [best_center[0], best_center[1] - delta, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Stage 5: Final LP-based center refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_final, center_final, D_final = calc_S_for_beta(best_pts, best_beta)
                if S_final < best_S:
                    best_S = S_final
                    best_center = center_final
        except Exception:
            pass

        return best_S, best_beta, best_center, best_pts

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_bbox_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])

    sorted_candidates = sorted(candidates, key=quick_bbox_score)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in sorted_candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Evaluate candidates
    for idx, axial in enumerate(unique_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, pts_opt = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_beta = beta
            best_global_center = center
        
        if idx >= 15 and best_global_S < 3.5:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 10:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(unique_candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Continuous joint optimization using smooth surrogate
    def continuous_joint_optimize(initial_pts, initial_beta, initial_center, max_time=25.0):
        """
        Optimizes positions of N hexagons and outer hexagon parameters using smooth surrogate.
        """
        n_pts = len(initial_pts)
        x0 = np.zeros(2 * n_pts + 3)
        for i in range(n_pts):
            x0[2*i] = initial_pts[i, 0]
            x0[2*i+1] = initial_pts[i, 1]
        x0[2*n_pts] = initial_center[0]
        x0[2*n_pts+1] = initial_center[1]
        x0[2*n_pts+2] = initial_beta

        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')
            
            pts = np.empty((n_pts, 2))
            for i in range(n_pts):
                pts[i, 0] = params[2*i]
                pts[i, 1] = params[2*i+1]
            
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            S_smooth = smooth_calc_S(pts, cx, cy, beta, temperature=0.05)
            
            penalty = 0.0
            min_dist = 2.0 * APOTHEM - 0.01
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return S_smooth + 100.0 * penalty

        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < float('inf'):
                best_params = res.x
                pts_opt = np.empty((n_pts, 2))
                for i in range(n_pts):
                    pts_opt[i, 0] = best_params[2*i]
                    pts_opt[i, 1] = best_params[2*i+1]
                
                cx_opt = best_params[2*n_pts]
                cy_opt = best_params[2*n_pts+1]
                beta_opt = best_params[2*n_pts+2]
                beta_opt = ((beta_opt % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                angles_deg = [math.degrees(beta_opt)] * n_pts
                if check_all_disjoint(pts_opt, angles_deg, tol=1e-6):
                    S_exact = exact_S(pts_opt, cx_opt, cy_opt, beta_opt)
                    return pts_opt, beta_opt, (cx_opt, cy_opt), S_exact
        except Exception:
            pass
        
        return None

    # Try continuous joint optimization on the best candidate
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
        result = continuous_joint_optimize(best_global_pts, best_global_beta, best_global_center)
        
        if result is not None:
            rp, rb, rc, rS = result
            if rS < best_global_S:
                best_global_S = rS
                best_global_pts = rp
                best_global_beta = rb
                best_global_center = rc

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final: compute side length
    pts_final = best_global_pts.copy()
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    S_final = (max_proj + APOTHEM) / APOTHEM

    # Verify disjointness
    angles_deg_final = [angle_deg] * len(pts_final)
    if not check_all_disjoint(pts_final, angles_deg_final, tol=1e-6):
        # Fallback to lattice configuration
        pts_lattice = axial_to_cartesian(unique_candidates[0])
        max_proj_lattice = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts_lattice[:, 0] * nx + pts_lattice[:, 1] * ny
            max_proj_lattice = max(max_proj_lattice, np.max(projs), -np.min(projs))
        S_final = (max_proj_lattice + APOTHEM) / APOTHEM
        pts_final = pts_lattice
        cx, cy = float(np.mean(pts_final[:, 0])), float(np.mean(pts_final[:, 1]))
        angle_deg = 0.0

    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
