import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _all_vertices(centers, angles):
    """centers (11,2), angles (11,) in degrees. Returns (11,6,2)."""
    a = np.deg2rad(angles)[:, None]
    th = a + np.deg2rad(60.0) * np.arange(6)
    x = centers[:, 0:1] + np.cos(th)
    y = centers[:, 1:2] + np.sin(th)
    return np.stack([x, y], axis=-1)

def _edge_normals(V):
    """V (6,2) -> (6,2) outward edge normals."""
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        if L < 1e-12:
            n[i] = [1.0, 0.0]
        else:
            n[i] = np.array([e[1], -e[0]]) / L
    return n

def _separation_penalty(Vs, eps=1e-6):
    """Sum of squared violations for all pairs. Returns scalar >= 0."""
    pen = 0.0
    n = len(Vs)
    for i in range(n):
        for j in range(i + 1, n):
            A, B = Vs[i], Vs[j]
            nA = _edge_normals(A)
            nB = _edge_normals(B)
            best = -np.inf
            for nrm in list(nA) + list(nB):
                loA, hiA = (A @ nrm).min(), (A @ nrm).max()
                loB, hiB = (B @ nrm).min(), (B @ nrm).max()
                sep = max(loB - hiA, loA - hiB)
                if sep > best:
                    best = sep
            # best is the max gap; we need best > eps for validity
            # penalty: if best <= eps, push apart
            if best <= eps:
                pen += (eps - best) ** 2
    return pen

def _containment_penalty(Vs, OV):
    """Sum of squared violations for containment. Returns scalar >= 0."""
    pen = 0.0
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            e = p2 - p1
            # cross product: (p2-p1) x (p - p1)
            cross = e[0] * (V[:, 1] - p1[1]) - e[1] * (V[:, 0] - p1[0])
            viol = np.minimum(cross, 0.0)
            pen += np.sum(viol ** 2)
    return pen

def _outer_side_for_centers(centers, outer_angle_deg):
    """Given centers (11,2) and outer hexagon angle, return (side, outer_vertices)."""
    a = np.deg2rad(outer_angle_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    normals = np.column_stack((np.cos(th), np.sin(th)))
    # We need to find the apothem such that all hexagon vertices are inside.
    # But for the objective, we can approximate by using centers + max vertex extent.
    # Actually, for the outer hexagon to contain all inner hexagons,
    # each inner vertex p must satisfy p . n_k <= apothem for all k.
    # So apothem = max over all inner vertices and all normals of (p . n_k).
    # But we don't have all inner vertices here efficiently.
    # Let's compute: for each normal n_k, the max projection of any inner vertex.
    # Inner vertices: for hexagon i, vertices are c_i + R(theta_i) * unit_hex_verts.
    # p . n = c_i . n + (R(theta_i)*u) . n = c_i . n + u . R(-theta_i)*n
    # max over u of u . v is |v| = 1 (since u is unit and v has unit norm).
    # So max vertex projection on n is c_i . n + 1.
    # Therefore apothem = max_i (c_i . n_k) + 1 for each k, then max over k.
    proj = centers @ normals.T  # (11, 6)
    maxproj = proj.max(axis=0) + 1.0  # (6,)
    apothem = maxproj.max()
    side = 2.0 * apothem / np.sqrt(3.0)
    # Outer vertices
    OV = np.column_stack((side * np.cos(th), side * np.sin(th)))
    return side, OV

def _objective(params):
    """params: 33 (x,y,angle_deg) for 11 hexagons + 1 (outer_angle_deg) = 34."""
    if len(params) != 34:
        return 1e10
    centers = params[:33].reshape(11, 3)[:, :2]
    angles = params[:33].reshape(11, 3)[:, 2]
    outer_angle = params[33]
    
    # Compute side length
    side, OV = _outer_side_for_centers(centers, outer_angle)
    
    if side < np.sqrt(11.0) - 1e-3:
        return 1e6 + (np.sqrt(11.0) - side) ** 2
    
    # Compute separation penalty
    Vs = _all_vertices(centers, angles)
    sep_pen = _separation_penalty(Vs)
    
    # Containment penalty (should be ~0 by construction of apothem, but check)
    # Actually the apothem formula guarantees containment of vertices,
    # so containment penalty should be 0. But let's be safe.
    # For speed, skip containment penalty in objective since it's structurally satisfied.
    
    return side + 100.0 * sep_pen

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 580.0  # 20s safety margin
    
    def _make_initial_configs():
        configs = []
        
        def axial_to_cartesian(q, r):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            return x, y
        
        # Config 0: 7-flower + 4 extensions
        centers0 = [
            axial_to_cartesian(0, 0),
            axial_to_cartesian(1, 0),
            axial_to_cartesian(0, 1),
            axial_to_cartesian(-1, 1),
            axial_to_cartesian(-1, 0),
            axial_to_cartesian(0, -1),
            axial_to_cartesian(1, -1),
            axial_to_cartesian(2, 0),
            axial_to_cartesian(0, 2),
            axial_to_cartesian(-2, 0),
            axial_to_cartesian(0, -2),
        ]
        configs.append(np.array(centers0))
        
        # Config 1: 3x3 grid + 2
        centers1 = [
            axial_to_cartesian(0, 0),
            axial_to_cartesian(1, 0),
            axial_to_cartesian(2, 0),
            axial_to_cartesian(0, 1),
            axial_to_cartesian(1, 1),
            axial_to_cartesian(2, 1),
            axial_to_cartesian(0, -1),
            axial_to_cartesian(1, -1),
            axial_to_cartesian(2, -1),
            axial_to_cartesian(3, 0),
            axial_to_cartesian(3, 1),
        ]
        configs.append(np.array(centers1))
        
        # Config 2: Compact cluster
        centers2 = [
            (0, 0),
            (2, 0),
            (1, 1.732),
            (-1, 1.732),
            (3, 1.732),
            (2, 3.464),
            (0, 3.464),
            (-2, 3.464),
            (-3, 1.732),
            (-1, -1.732),
            (1, -1.732),
        ]
        configs.append(np.array(centers2))
        
        # Config 3: Linear-ish with bends
        centers3 = [
            (0, 0),
            (2, 0),
            (4, 0),
            (1, 1.732),
            (3, 1.732),
            (5, 1.732),
            (2, 3.464),
            (4, 3.464),
            (-1, 1.732),
            (-1, -1.732),
            (1, -1.732),
        ]
        configs.append(np.array(centers3))
        
        # Config 4: Diamond
        centers4 = [
            (0, 0),
            (2, 0),
            (-2, 0),
            (0, 2.309),
            (0, -2.309),
            (2, 2.309),
            (-2, 2.309),
            (2, -2.309),
            (-2, -2.309),
            (4, 0),
            (-4, 0),
        ]
        configs.append(np.array(centers4))
        
        return configs
    
    def _full_validity(inner, outer, side):
        """Full validity check."""
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11.0) - 1e-9:
            return False
        Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
        for i in range(11):
            for j in range(i + 1, 11):
                A, B = Vs[i], Vs[j]
                nA = _edge_normals(A)
                nB = _edge_normals(B)
                best = -np.inf
                for nrm in list(nA) + list(nB):
                    loA, hiA = (A @ nrm).min(), (A @ nrm).max()
                    loB, hiB = (B @ nrm).min(), (B @ nrm).max()
                    sep = max(loB - hiA, loA - hiB)
                    if sep > best:
                        best = sep
                if best <= 1e-6:
                    return False
        cx, cy, ca = outer[0], outer[1], outer[2]
        OV = _hex_vertices(cx, cy, side, ca)
        for V in Vs:
            for i in range(6):
                p1 = OV[i]
                p2 = OV[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
                if np.any(cross < -1e-6):
                    return False
        for V in Vs:
            Vs2 = V - np.array([cx, cy])
            for i in range(6):
                L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
                if abs(L - 1.0) > 1e-8:
                    return False
        return True
    
    best_val = -np.inf
    best_result = None
    
    configs = _make_initial_configs()
    
    for cfg_idx, centers in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Center the cluster
        centers = centers - centers.mean(axis=0)
        
        # Initial params: x, y, angle=0 for each, plus outer_angle=0
        x0 = np.zeros(34)
        x0[:22:3] = centers[:, 0]
        x0[1:23:3] = centers[:, 1]
        x0[2:34:3] = 0.0  # angles
        x0[33] = 0.0  # outer angle
        
        # Bounds
        bounds = [(-10, 10)] * 11 + [(-10, 10)] * 11 + [(-180, 180)] * 11 + [(0, 60)]
        
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if res.success or res.fun < 1e5:
            params = res.x
            centers_opt = params[:33].reshape(11, 3)[:, :2]
            angles_opt = params[:33].reshape(11, 3)[:, 2]
            outer_angle_opt = params[33]
            
            inner = np.column_stack((centers_opt, angles_opt))
            side, OV = _outer_side_for_centers(centers_opt, outer_angle_opt)
            outer = np.array([0.0, 0.0, outer_angle_opt])
            
            if _full_validity(inner, outer, side):
                val = 1.0 / side
                if val > best_val:
                    best_val = val
                    best_result = (inner, outer, side)
    
    # If no valid result from optimization, try a simple fallback
    if best_result is None:
        # Use a known loose packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        # Compute actual side
        centers = inner[:, :2]
        best_s = np.inf
        best_a = 0
        for a_deg in np.linspace(0, 60, 361):
            a = np.deg2rad(a_deg)
            th = a + np.deg2rad(60.0) * np.arange(6)
            normals = np.column_stack((np.cos(th), np.sin(th)))
            proj = centers @ normals.T
            maxproj = proj.max(axis=0) + 1.0
            apothem = maxproj.max()
            s_val = 2.0 * apothem / np.sqrt(3.0)
            if s_val < best_s:
                best_s = s_val
                best_a = a_deg
        outer = np.array([0.0, 0.0, best_a])
        best_result = (inner, outer, best_s)
    
    return best_result
