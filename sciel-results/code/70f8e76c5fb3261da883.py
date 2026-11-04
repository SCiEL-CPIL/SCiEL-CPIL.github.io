"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _hex_normals(angle_deg):
    """
    Pre-computed edge normals for a unit hexagon with given rotation.
    The edges of a regular hexagon (vertices at 0, 60, ...) have normals at 30, 90, 150, 210, 270, 330.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Rotate normal
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons (vectorized)."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = np.array([
        [math.cos(a + math.radians(90 + 60 * k)), math.sin(a + math.radians(90 + 60 * k))]
        for k in range(6)
    ])  # shape (6, 2)
    
    cx, cy = center
    # Precompute all vertices
    all_verts = np.array([_hex_vertices(p[0], p[1], p[2]) for p in centers_angles])  # (12, 6, 2)
    # Relative positions
    rel = all_verts - np.array([cx, cy])  # (12, 6, 2)
    # Project onto all normals: (12, 6, 6)
    projections = rel @ normals.T  # (12, 6, 6)
    max_need = projections.max() / apothem_unit
    return float(max_need)

def _outer_hex_side_smooth(centers_angles, center, angle_deg, beta=100.0):
    """Smooth version of outer hex side using log-sum-exp (vectorized)."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = np.array([
        [math.cos(a + math.radians(90 + 60 * k)), math.sin(a + math.radians(90 + 60 * k))]
        for k in range(6)
    ])  # shape (6, 2)
    
    cx, cy = center
    all_verts = np.array([_hex_vertices(p[0], p[1], p[2]) for p in centers_angles])  # (12, 6, 2)
    rel = all_verts - np.array([cx, cy])  # (12, 6, 2)
    projections = (rel @ normals.T).flatten()  # (432,)
    v_arr = projections / apothem_unit
    v_max = np.max(v_arr)
    if v_max < -50:
        return 0.0
    # Numerically stable log-sum-exp
    shifted = beta * (v_arr - v_max)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return v_max + log_sum / beta

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint using vectorized SAT."""
    m = len(centers_angles)
    # Precompute vertices as numpy arrays
    polys = np.array([_hex_vertices(p[0], p[1], p[2]) for p in centers_angles])
    normals = np.array([_hex_normals(p[2]) for p in centers_angles])
    
    for i in range(m):
        for j in range(i + 1, m):
            # Quick center-distance rejection
            dx = centers_angles[j][0] - centers_angles[i][0]
            dy = centers_angles[j][1] - centers_angles[i][1]
            if dx * dx + dy * dy > 4.5:
                continue
            # SAT check: project onto all 12 axes
            axes = np.vstack([normals[i], normals[j]])  # shape (12, 2)
            proj_i = polys[i] @ axes.T  # shape (6, 12)
            proj_j = polys[j] @ axes.T  # shape (6, 12)
            min_i = proj_i.min(axis=0)
            max_i = proj_i.max(axis=0)
            min_j = proj_j.min(axis=0)
            max_j = proj_j.max(axis=0)
            # Check if any axis separates
            separated = np.any(max_i <= min_j + tol) or np.any(max_j <= min_i + tol)
            if not separated:
                return False
    return True

def _sdf_distance(p1, p2, a1, a2):
    """
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    def hex_radius(angle_deg, direction_deg):
        """
        Support function of a unit regular hexagon.
        """
        a_rad = math.radians(angle_deg)
        d_rad = math.radians(direction_deg)
        dx_dir = math.cos(d_rad)
        dy_dir = math.sin(d_rad)
        
        max_val = 0.0
        for k in range(6):
            theta = math.radians(60 * k)
            vx = math.cos(theta)
            vy = math.sin(theta)
            ca = math.cos(a_rad)
            sa = math.sin(a_rad)
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            dot = rvx * dx_dir + rvy * dy_dir
            if dot > max_val:
                max_val = dot
        return max_val
    
    dir_deg = math.degrees(math.atan2(dy, dx))
    r1 = hex_radius(a1, dir_deg)
    r2 = hex_radius(a2, dir_deg + 180)
    
    return dist - r1 - r2

def construct():
    t0 = time.time()
    random.seed(42)
    
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
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_flower():
        """Flower pattern: 1 center + 6 inner ring + 5 outer."""
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
    
    def make_dense_cluster():
        """Tight cluster arrangement."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_two_rings():
        """6 inner + 6 outer ring."""
        pts = []
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.2
        for i in range(6):
            theta = 2 * math.pi * (i + 0.5) / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_grid_rotated():
        """3x4 grid rotated 30 degrees."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        rot = math.radians(30)
        cr, sr = math.cos(rot), math.sin(rot)
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0) - 3.0
                y = r * row_height - row_height
                rx = cr * x - sr * y
                ry = sr * x + cr * y
                pts.append([rx, ry, 30.0])
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    def smooth_objective_lse(v, beta_sdf=60.0, beta_lse=100.0):
        """
        Smooth objective using log-sum-exp for outer side and SDF for overlaps.
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        
        # Compute outer side with log-sum-exp smoothing
        outer_side = _outer_hex_side_smooth(cfg, tuple(center), angle, beta_lse)
        
        # Compute overlap penalties using SDF
        total_penalty = 0.0
        m = len(cfg)
        for i in range(m):
            for j in range(i + 1, m):
                d = _sdf_distance(cfg[i][0], cfg[i][1], cfg[i][2], 
                                  cfg[j][0], cfg[j][1], cfg[j][2])
                if d < 0:
                    total_penalty += math.exp(-beta_sdf * (-d))
        
        return outer_side + 8.0 * total_penalty
    
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return v
    
    def unflatten(v):
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        return cfg, center, angle
    
    def lbfgs_polish(cfg_init, center_init, angle_init, time_budget):
        """L-BFGS-B optimization using smooth objective with LSE."""
        v0 = flatten(cfg_init, center_init, angle_init)
        
        start_time = time.time()
        def callback(xk):
            return time.time() - start_time > (time_budget - 2.0)
        
        result = minimize(
            smooth_objective_lse,
            v0,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        v_opt = result.x
        cfg_opt, center_opt, angle_opt = unflatten(v_opt)
        
        score = full_score(cfg_opt, tuple(center_opt), angle_opt)
        if score is not None:
            return v_opt, score
        return v0, None
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster(),
             make_two_rings(), make_grid_rotated()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-4, 4), random.uniform(-4, 4), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated Annealing phase
    T = 0.6
    T_end = 0.001
    alpha_sa = 0.9996
    
    # Time allocation: SA gets most time, then LBFGS polish + golden section
    sa_time_budget = TIME_LIMIT - 28.0  # leave 28s for LBFGS + golden section + safety
    
    iteration = 0
    while time.time() - t0 < sa_time_budget:
        iteration += 1
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.12 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 3.0 + 2.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.06 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 1.0 + 1.0 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.10 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.45:
                    cfg[idx][0] = old_x
                    cfg[idx][1] = old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0] = old_cx
                    center[1] = old_cy
                elif r < 0.85:
                    angle = old_angle
                else:
                    cfg[idx1][0] = old_x1
                    cfg[idx1][1] = old_y1
                    cfg[idx2][0] = old_x2
                    cfg[idx2][1] = old_y2
        else:
            if r < 0.45:
                cfg[idx][0] = old_x
                cfg[idx][1] = old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0] = old_cx
                center[1] = old_cy
            elif r < 0.85:
                angle = old_angle
            else:
                cfg[idx1][0] = old_x1
                cfg[idx1][1] = old_y1
                cfg[idx2][0] = old_x2
                cfg[idx2][1] = old_y2
        
        T = max(T_end, T * alpha_sa)
    
    # L-BFGS-B polish using smooth objective with LSE
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            lbfgs_v, lbfgs_score = lbfgs_polish(
                best_cfg, best_center, best_angle, remaining_time - 3.0
            )
            if lbfgs_score is not None and lbfgs_score < best_score:
                cfg_lbfgs = []
                for i in range(N):
                    cfg_lbfgs.append([lbfgs_v[3*i], lbfgs_v[3*i+1], lbfgs_v[3*i+2]])
                center_lbfgs = [lbfgs_v[3*N], lbfgs_v[3*N+1]]
                angle_lbfgs = lbfgs_v[3*N+2]
                
                verify_score = full_score(cfg_lbfgs, tuple(center_lbfgs), angle_lbfgs)
                if verify_score is not None and verify_score < best_score:
                    best_score = verify_score
                    best_cfg = cfg_lbfgs
                    best_center = center_lbfgs
                    best_angle = angle_lbfgs
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B (tighter tolerance, LSE objective)
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback2(xk):
                return time.time() - start_time > (remaining_time - 2.0)
            
            result = minimize(
                smooth_objective_lse,
                v0,
                method='L-BFGS-B',
                callback=callback2,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            v_opt = result.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            center_opt = [v_opt[3*N], v_opt[3*N+1]]
            angle_opt = v_opt[3*N+2]
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    # Nelder-Mead final polish for high-precision local convergence
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback_nm(xk):
                return time.time() - start_time > (remaining_time - 3.0)
            
            result_nm = minimize(
                smooth_objective_lse,
                v0,
                method='Nelder-Mead',
                callback=callback_nm,
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            
            v_opt = result_nm.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            center_opt = [v_opt[3*N], v_opt[3*N+1]]
            angle_opt = v_opt[3*N+2]
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    # Golden-section search on outer hexagon rotation for fixed inner config
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 2.0:
        try:
            def outer_side_vs_angle(ang):
                return _outer_hex_side(best_cfg, tuple(best_center), ang)
            
            # Golden section search over [0, 60] degrees (hexagon symmetry)
            gr = (math.sqrt(5) - 1) / 2.0
            a_lo, a_hi = 0.0, 60.0
            a1 = a_hi - gr * (a_hi - a_lo)
            a2 = a_lo + gr * (a_lo + a_hi - a_lo)
            f1 = outer_side_vs_angle(a1)
            f2 = outer_side_vs_angle(a2)
            
            gs_deadline = time.time() + remaining_time - 1.0
            while time.time() < gs_deadline and (a_hi - a_lo) > 1e-6:
                if f1 < f2:
                    a_hi = a2
                    a2 = a1
                    f2 = f1
                    a1 = a_hi - gr * (a_hi - a_lo)
                    f1 = outer_side_vs_angle(a1)
                else:
                    a_lo = a1
                    a1 = a2
                    f1 = f2
                    a2 = a_lo + gr * (a_hi - a_lo)
                    f2 = outer_side_vs_angle(a2)
            
            best_gs_angle = (a_lo + a_hi) / 2.0
            gs_score = _outer_hex_side(best_cfg, tuple(best_center), best_gs_angle)
            if gs_score < best_score:
                best_score = gs_score
                best_angle = best_gs_angle
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
