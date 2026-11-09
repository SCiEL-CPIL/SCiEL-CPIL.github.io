# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_s(inner, outer_angle_deg, outer_center):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oang = np.deg2rad(outer_angle_deg)
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # n_k angles in radians
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    # For each vertex v, compute max_k dot(v - C, n_k)
    dots = all_verts @ normals.T - outer_center @ normals.T  # shape (72, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    
    return s_min

def _objective(params, deadline_ref=None):
    """Objective: minimize s. Params: 24 center coords + 12 angles + 3 outer params = 39 params."""
    cx_cy = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_center = params[36:38]
    outer_angle = params[38]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = cx_cy[:, 0]
    inner[:, 1] = cx_cy[:, 1]
    inner[:, 2] = inner_angles
    
    s = _compute_min_s(inner, outer_angle, outer_center)
    
    if not np.isfinite(s) or s < math.sqrt(12) - 1e-9:
        return s + 100.0
    
    outer = np.array([outer_center[0], outer_center[1], outer_angle])
    
    if not _is_valid(inner, outer, s):
        return s + 100.0
    
    return s

def _make_initial_hex_packing():
    """Create a hexagonal packing of 12 unit hexagons."""
    r = 2 * math.sqrt(3)  # center-to-center distance for touching hexagons
    
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 hexagons at distance r
    for k in range(6):
        angle = math.pi / 3 * k
        positions.append((r * math.cos(angle), r * math.sin(angle)))
    # Ring 2: 5 more hexagons (positions 7-11)
    # Use positions at distance ~2r in directions between ring 1
    for k in range(5):
        angle = math.pi / 3 * (2*k + 1)  # offset angles
        dist = 2 * r
        positions.append((dist * math.cos(angle), dist * math.sin(angle)))
    
    return positions[:12]

def _make_initial_grid_packing():
    """Create a grid-based initial packing."""
    positions = []
    # 4x3 grid with hexagonal offset
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (0.5 if row % 2 == 1 else 0)
            y = row * math.sqrt(3)
            positions.append((x - 1.5, y - math.sqrt(3)))
    return positions[:12]

def hexagon_packing_12():
    start_time = time.time()
    deadline = 610.0
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    def _update_best(params):
        nonlocal best_s, best_inner, best_outer
        cx_cy = params[:24].reshape(12, 2)
        inner_angles = params[24:36]
        outer_center = params[36:38]
        outer_angle = params[38]
        
        inner = np.zeros((12, 3))
        inner[:, 0] = cx_cy[:, 0]
        inner[:, 1] = cx_cy[:, 1]
        inner[:, 2] = inner_angles
        
        s = _compute_min_s(inner, outer_angle, outer_center)
        outer = np.array([outer_center[0], outer_center[1], outer_angle])
        
        if s < best_s and _is_valid(inner, outer, s):
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    def _check_time():
        return time.time() > deadline
    
    # Try multiple initial configurations
    initial_configs = [
        _make_initial_hex_packing(),
        _make_initial_grid_packing(),
    ]
    
    # Add a few random perturbations
    rng = np.random.RandomState(42)
    for _ in range(2):
        base = _make_initial_hex_packing()
        perturbed = [(x + rng.randn() * 0.3, y + rng.randn() * 0.3) for x, y in base]
        initial_configs.append(perturbed)
    
    for init_idx, initial_positions in enumerate(initial_configs):
        if _check_time():
            break
        
        # Build initial params
        initial_params = np.zeros(39)
        for i in range(12):
            initial_params[2*i] = initial_positions[i][0]
            initial_params[2*i+1] = initial_positions[i][1]
        # Angles default to 0, outer center (0,0), outer angle 0
        
        # Evaluate initial
        _update_best(initial_params)
        
        # Phase 1: Differential Evolution
        if _check_time():
            break
        
        bounds = [(-8, 8)] * 24  # centers
        bounds += [(-180, 180)] * 12  # inner angles
        bounds += [(-5, 5), (-5, 5)]  # outer center
        bounds += [(-30, 30)]  # outer angle
        
        def cb1(xk, convergence):
            _update_best(xk)
            return _check_time()
        
        try:
            from scipy.optimize import differential_evolution
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=150,
                tol=1e-8,
                mut=(0.5, 1.5),
                recomb=0.7,
                seed=42 + init_idx,
                polish=False,
                callback=cb1
            )
            _update_best(res.x)
        except Exception as e:
            pass
        
        if _check_time():
            break
        
        # Phase 2: Local refinement with Nelder-Mead
        if best_inner is not None and time.time() < deadline - 30:
            best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2],
                                      best_outer[:2], [best_outer[2]]])
            try:
                from scipy.optimize import minimize
                res2 = minimize(
                    _objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                _update_best(res2.x)
            except Exception as e:
                pass
    
    # Final multi-start local refinement from best
    if best_inner is not None and time.time() < deadline - 20:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2],
                                  best_outer[:2], [best_outer[2]]])
        
        # Try several small perturbations
        rng2 = np.random.RandomState(123)
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            perturbed_x = best_x + rng2.randn(39) * 0.05
            try:
                from scipy.optimize import minimize
                res = minimize(
                    _objective, perturbed_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
                )
                _update_best(res.x)
            except Exception:
                pass
    
    # Fallback if nothing was found
    if best_inner is None:
        positions = _make_initial_hex_packing()
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = positions[i][0]
            inner[i, 1] = positions[i][1]
            inner[i, 2] = 0
        outer = np.array([0.0, 0.0, 0.0])
        s = _compute_min_s(inner, 0.0, np.array([0.0, 0.0]))
        if not _is_valid(inner, outer, s):
            s = 10.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
