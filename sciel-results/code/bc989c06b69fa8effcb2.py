"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
INRADIUS = np.sqrt(3) / 2.0

def _hex_vertices_array(cx, cy, side, angle_rad):
    """Vectorized hex vertices."""
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _sat_disjoint_fast(hex1, hex2, eps=1e-6):
    """Fast SAT check for two unit hexagons."""
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    verts1 = _hex_vertices_array(x1, y1, 1.0, np.radians(a1))
    verts2 = _hex_vertices_array(x2, y2, 1.0, np.radians(a2))
    
    for verts in [verts1, verts2]:
        for i in range(6):
            dx = verts[(i+1)%6, 0] - verts[i, 0]
            dy = verts[(i+1)%6, 1] - verts[i, 1]
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-10:
                continue
            nx = -dy / length
            ny = dx / length
            proj1 = verts1 @ np.array([nx, ny])
            proj2 = verts2 @ np.array([nx, ny])
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
    return False

def _all_disjoint(inner, eps=1e-6):
    """Check if all hexagons are disjoint."""
    for i in range(N):
        for j in range(i+1, N):
            if not _sat_disjoint_fast(inner[i], inner[j], eps):
                return False
    return True

def _overlap_penalty(inner, beta=50.0):
    """Compute smooth overlap penalty for all pairs."""
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            verts1 = _hex_vertices_array(x1, y1, 1.0, np.radians(a1))
            verts2 = _hex_vertices_array(x2, y2, 1.0, np.radians(a2))
            
            min_gap = float('inf')
            for verts in [verts1, verts2]:
                for k in range(6):
                    dx = verts[(k+1)%6, 0] - verts[k, 0]
                    dy = verts[(k+1)%6, 1] - verts[k, 1]
                    length = np.sqrt(dx*dx + dy*dy)
                    if length < 1e-10:
                        continue
                    nx = -dy / length
                    ny = dx / length
                    proj1 = verts1 @ np.array([nx, ny])
                    proj2 = verts2 @ np.array([nx, ny])
                    gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                    if gap < min_gap:
                        min_gap = gap
            
            if min_gap < 0:
                penalty += -min_gap * beta
    return penalty

def _compute_min_side(inner, cx, cy, theta_deg):
    """Compute minimum outer side length."""
    theta = np.radians(theta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = theta + np.pi/6 + k * np.pi/3
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        for x, y, a in inner:
            verts = _hex_vertices_array(x, y, 1.0, np.radians(a))
            proj = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
            max_proj = max(max_proj, proj.max())
    return 2.0 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def make_hex_lattice(rows, cols, rotation_deg=0.0):
        """Create hexagonal lattice of unit hexagons."""
        spacing = 2 * np.sqrt(3)
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing
                y = r * spacing * 1.5
                if r % 2 == 1:
                    x += spacing / 2
                pts.append([x, y, rotation_deg])
        return pts
    
    def center_config(config):
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        return [[p[0] - cx0, p[1] - cy0, p[2]] for p in config]
    
    def decode(x):
        """Decode parameter vector to inner points and outer angle."""
        inner_pts = []
        for i in range(N):
            ix = x[3*i]
            iy = x[3*i+1]
            ia = np.degrees(x[3*i+2])
            inner_pts.append([ix, iy, ia])
        outer_angle = np.degrees(x[3*N])
        return inner_pts, outer_angle
    
    def smooth_objective(x, beta=50.0):
        """Smooth objective with barrier penalty for overlaps."""
        if time.time() > deadline:
            return 1e10
        inner_pts, outer_angle = decode(x)
        s = _compute_min_side(inner_pts, 0.0, 0.0, outer_angle)
        penalty = _overlap_penalty(inner_pts, beta)
        return s + penalty
    
    # Generate diverse initial configurations
    configs = []
    configs.append(center_config(make_hex_lattice(3, 4)))
    configs.append(center_config(make_hex_lattice(4, 3)))
    configs.append(center_config(make_hex_lattice(2, 6)))
    configs.append(center_config(make_hex_lattice(6, 2)))
    
    np.random.seed(42)
    for base_config in configs[:2]:
        perturbed = []
        for p in base_config:
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            da = np.random.uniform(-15, 15)
            perturbed.append([p[0] + dx, p[1] + dy, p[2] + da])
        configs.append(center_config(perturbed))
    
    # For each config, run Nelder-Mead with smooth barrier
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 20.0:
            break
        
        n_vars = 3 * N + 1
        x0 = []
        for p in config:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(0.0)
        
        try:
            result = minimize(smooth_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            
            if result.fun < 1e7:
                inner_pts, outer_angle = decode(result.x)
                
                if _all_disjoint(inner_pts):
                    s_final = _compute_min_side(inner_pts, 0.0, 0.0, outer_angle)
                    if s_final < best_score:
                        best_score = s_final
                        best_inner = [list(p) for p in inner_pts]
                        best_center = (0.0, 0.0)
                        best_side = s_final
                        best_angle = outer_angle
                        
        except Exception:
            pass
    
    # Local random-walk polish with adaptive step size
    if best_inner is not None and time.time() < deadline - 10.0:
        current_inner = [list(h) for h in best_inner]
        current_cx, current_cy = 0.0, 0.0
        current_theta = best_angle
        current_side = best_score
        step_size = 0.1
        
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for _ in range(30):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                r = np.random.random()
                if r < 0.7:
                    idx = np.random.randint(0, N)
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.85:
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                else:
                    new_theta += np.random.uniform(-2.0, 2.0)
                
                if not _all_disjoint(new_inner):
                    continue
                
                new_side = _compute_min_side(new_inner, new_cx, new_cy, new_theta)
                
                if new_side < current_side - 1e-9:
                    current_inner = new_inner
                    current_cx, current_cy = new_cx, new_cy
                    current_theta = new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.95
                    break
            
            if not improved:
                step_size *= 0.8
                if step_size < 0.001:
                    break
            
            if current_side < best_score:
                best_score = current_side
                best_inner = [list(h) for h in current_inner]
                best_center = (current_cx, current_cy)
                best_side = current_side
                best_angle = current_theta
    
    # Final polish: optimize outer center and angle
    if best_inner is not None and time.time() < deadline - 5.0:
        def outer_objective(x):
            ox, oy, oangle = x
            return _compute_min_side(best_inner, ox, oy, np.degrees(oangle))
        
        x0_outer = [best_center[0], best_center[1], np.radians(best_angle)]
        try:
            result_outer = minimize(outer_objective, x0_outer, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            if result_outer.fun < best_score:
                ox, oy, oangle = result_outer.x
                s_final = _compute_min_side(best_inner, ox, oy, np.degrees(oangle))
                if s_final < best_score:
                    best_score = s_final
                    best_center = (ox, oy)
                    best_side = s_final
                    best_angle = np.degrees(oangle)
        except Exception:
            pass
    
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
