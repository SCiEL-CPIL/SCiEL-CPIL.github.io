import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses flower packing with optimized rotations via support function minimization.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Unit hexagon vertices (flat-top, side=1, circumradius=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_verts = np.array([[np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)])
    
    # Flower packing centers
    # Ring 0: center at origin
    # Ring 1: 6 hexagons at distance 2*sqrt(3) at angles 0,60,...,300
    # Ring 2: 4 hexagons at distance 4*sqrt(3) at angles 30,90,150,210 (gap positions)
    d1 = 2.0 * np.sqrt(3)
    d2 = 4.0 * np.sqrt(3)
    
    centers = np.array([[0.0, 0.0]])
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers = np.vstack([centers, [[d1*np.cos(a), d1*np.sin(a)]]])
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [[d2*np.cos(a), d2*np.sin(a)]]])
    
    n = len(centers)  # 11
    
    # Precompute rotation matrices for base vertices
    def rotate_verts(phi):
        """Return rotated vertices for angle phi (radians)."""
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        return base_verts @ R.T
    
    def support_function(theta, phis, beta=None):
        """
        Compute outer hexagon side length via support function.
        theta: outer hexagon rotation (radians)
        phis: array of inner hexagon rotations (radians)
        beta: if set, use soft-max with this temperature; else exact max
        """
        # Outer hexagon face normals (perpendicular to edges)
        # For a regular hexagon rotated by theta, normals are at theta + 30 + 60k
        normals = np.array([[np.cos(np.deg2rad(30 + 60*k) + theta),
                             np.sin(np.deg2rad(30 + 60*k) + theta)] for k in range(6)])
        
        all_projs = []
        for i in range(n):
            v = rotate_verts(phis[i]) + centers[i]
            dots = v @ normals.T  # shape (6, 6)
            all_projs.append(dots.ravel())
        
        all_projs = np.concatenate(all_projs)
        
        if beta is None:
            max_proj = np.max(all_projs)
        else:
            m = np.max(all_projs)
            log_sum_exp = m + np.log(np.sum(np.exp((all_projs - m) * beta))) / beta
            max_proj = log_sum_exp
        
        # Side = 2 * apothem = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def overlap_penalty(phis, weight=10.0):
        """Penalty for overlapping inner hexagons using distance between closest edges."""
        penalty = 0.0
        for i in range(n):
            vi = rotate_verts(phis[i]) + centers[i]
            for j in range(i+1, n):
                vj = rotate_verts(phis[j]) + centers[j]
                # Quick check: center distance
                dc = np.linalg.norm(centers[i] - centers[j])
                if dc > 2.0*np.sqrt(3) + 0.5:
                    continue
                # Check min distance between edges
                min_dist = 1e10
                for a in range(6):
                    p1, p2 = vi[a], vi[(a+1)%6]
                    for b in range(6):
                        p3, p4 = vj[b], vj[(b+1)%6]
                        d = seg_seg_dist(p1, p2, p3, p4)
                        if d < min_dist:
                            min_dist = d
                if min_dist < 0.0:
                    penalty += weight * (-min_dist) ** 2
        return penalty
    
    def seg_seg_dist(p1, p2, p3, p4):
        """Min distance between segments p1-p2 and p3-p4."""
        p1, p2, p3, p4 = map(lambda x: np.asarray(x, dtype=float), (p1, p2, p3, p4))
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(p1 - p3)
        if a < 1e-12:
            t = np.clip(f/e, 0, 1)
            return np.linalg.norm(p1 - (p3 + t*d2))
        c = np.dot(d1, r)
        if e < 1e-12:
            s = np.clip(c/a, 0, 1)
            return np.linalg.norm((p1 + s*d1) - p3)
        b = np.dot(d1, d2)
        denom = a*e - b*b
        if abs(denom) < 1e-12:
            t = np.clip(f/e, 0, 1)
            return np.linalg.norm(p1 - (p3 + t*d2))
        s = (b*f - c*e) / denom
        t = (b*s + f) / e
        if s < 0:
            s = 0
            t = np.clip(f/e, 0, 1)
        elif s > 1:
            s = 1
            t = np.clip((b+f)/e, 0, 1)
        elif t < 0:
            t = 0
            s = np.clip(-c/a, 0, 1)
        elif t > 1:
            t = 1
            s = np.clip((b-c)/a, 0, 1)
        return np.linalg.norm((p1 + s*d1) - (p3 + t*d2))
    
    def objective(params, use_soft=True, beta=30.0):
        theta = params[0]
        phis = params[1:]
        side = support_function(theta, phis, beta=beta if use_soft else None)
        pen = overlap_penalty(phis, weight=5.0)
        return side + pen
    
    def objective_exact(params):
        theta = params[0]
        phis = params[1:]
        return support_function(theta, phis, beta=None)
    
    # Initial guess: align each inner hexagon so a face points radially outward
    x0 = np.zeros(n + 1)
    for i in range(n):
        if i == 0:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(centers[i][1], centers[i][0])
            # Align a face normal to radial direction
            # Face normals in local frame are at 30+60k degrees
            # We want one face normal to point along alpha
            # So phi = alpha - (30+60k) degrees for some k
            # Choose k=0: phi = alpha - 30 deg
            x0[i+1] = alpha - np.deg2rad(30)
    
    best_val = float('inf')
    best_params = x0.copy()
    
    # Phase 1: Multi-start Powell with soft objective
    starts = [x0.copy()]
    # Additional diverse starts
    for trial in range(8):
        if elapsed() > deadline - 40:
            break
        x_trial = x0.copy()
        x_trial[0] = np.random.uniform(0, np.pi/3)
        x_trial[1:] += np.random.uniform(-0.2, 0.2, n)
        starts.append(x_trial)
    
    for idx, x_start in enumerate(starts):
        if elapsed() > deadline - 30:
            break
        try:
            res = minimize(objective, x_start, args=(True, 30.0), method='Powell',
                          options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    if elapsed() < deadline - 15:
        try:
            res2 = minimize(objective, best_params, args=(True, 50.0), method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 2000, 'adaptive': True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Exact objective refinement with Nelder-Mead
    if elapsed() < deadline - 10:
        try:
            res3 = minimize(objective_exact, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 3000})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Phase 4: 1D golden-section on outer rotation
    if elapsed() < deadline - 5:
        phis_fixed = best_params[1:]
        def outer_obj(theta):
            return support_function(theta, phis_fixed, beta=None)
        
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, np.pi / 3
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = outer_obj(c), outer_obj(d)
        
        for _ in range(100):
            if elapsed() > deadline - 2:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = outer_obj(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = outer_obj(d)
        
        theta_opt = (a + b) / 2
        side_opt = outer_obj(theta_opt)
        
        if side_opt < best_val - 1e-12:
            best_val = side_opt
            best_params = np.concatenate([[theta_opt], phis_fixed])
    
    # Phase 5: Final exact evaluation
    theta_out = best_params[0]
    phis = best_params[1:]
    side = support_function(theta_out, phis, beta=None)
    
    # Verify no overlaps
    pen = overlap_penalty(phis, weight=1.0)
    if pen > 0.001:
        # Fallback to safe configuration
        return _safe_solution()
    
    inner = []
    for i in range(n):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    return inner, (0.0, 0.0), side + 1e-9, np.rad2deg(theta_out)

def _safe_solution():
    """Fallback: all hexagons at 30 degrees, outer at 30 degrees."""
    d1 = 2.0 * np.sqrt(3)
    d2 = 4.0 * np.sqrt(3)
    
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers.append((d1*np.cos(a), d1*np.sin(a)))
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers.append((d2*np.cos(a), d2*np.sin(a)))
    
    inner = [[x, y, 30.0] for x, y in centers]
    
    # Compute side
    base_verts = np.array([[np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)])
    theta = np.deg2rad(30)
    phi = np.deg2rad(30)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    verts = base_verts @ R.T
    
    normals = np.array([[np.cos(np.deg2rad(30 + 60*k) + theta),
                         np.sin(np.deg2rad(30 + 60*k) + theta)] for k in range(6)])
    
    max_proj = 0.0
    for cx, cy in centers:
        v = verts + np.array([cx, cy])
        dots = v @ normals.T
        max_proj = max(max_proj, np.max(dots))
    
    side = 2.0 * max_proj / np.sqrt(3)
    
    return inner, (0.0, 0.0), side + 1e-9, 30.0
