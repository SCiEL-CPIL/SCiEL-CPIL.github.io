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

def _lattice_packing(params):
    """
    Crystallographic lattice packing: 12 hexagons on triangular lattice.
    params: [a1x, a1y, a2x, a2y, global_rotation_deg]
    Triangular lattice: basis vectors a1, a2 with |a1|=|a2|=d, angle 60 deg.
    12 sites: center + ring1(6) + ring2(5 selected for compactness)
    """
    a1 = np.array([params[0], params[1]])
    a2 = np.array([params[2], params[3]])
    rot = np.deg2rad(params[4])
    
    # Rotation matrix
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    
    # Triangular lattice sites (integer coordinates in lattice basis)
    # Center: (0,0)
    # Ring 1: (±1,0), (0,±1), (±1,±1) -> 6 sites
    # Ring 2: need 5 more. Choose (2,0), (0,2), (2,2), (-2,0), (-2,-2) or similar
    # Actually for 12 hexagons in hexagonal arrangement:
    # Layer 0: (0,0) -> 1
    # Layer 1: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1) -> 6
    # Layer 2: (2,0),(1,1),(-1,2),(-2,1),(-2,0),(-1,-1),(1,-2),(2,-1) -> 8, but we need only 5 more
    # Let's pick 5 from layer 2 to make 12 total: (2,0), (1,1), (-1,2), (-2,0), (0,-2)
    
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2)
    ]
    
    centers = []
    for (m, n) in lattice_coords:
        pos = R @ (m * a1 + n * a2)
        centers.append(pos)
    
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers[i][0]
        inner[i, 1] = centers[i][1]
        inner[i, 2] = 0.0  # all same orientation
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    s = _compute_min_s(inner, outer_angle, outer_center)
    
    return inner, np.array([0.0, 0.0, 0.0]), s

def _objective_lattice(params):
    """Objective for lattice-based packing."""
    inner, outer, s = _lattice_packing(params)
    if not np.isfinite(s) or s < math.sqrt(12) - 1e-9:
        return s + 100.0
    if not _is_valid(inner, outer, s):
        return s + 100.0
    return s

def _free_objective(params):
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

def _initial_hexagonal_packing():
    """Create initial hexagonal packing configuration."""
    r = 2 * math.sqrt(3)  # center-to-center distance for touching
    
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = math.pi / 3 * k
        positions.append((r * math.cos(angle), r * math.sin(angle)))
    # Ring 2: 5 more hexagons
    for k in range(0, 12, 2):
        if len(positions) >= 12:
            break
        angle = math.pi / 3 * k + math.pi / 6
        positions.append((2 * r * math.cos(angle), 2 * r * math.sin(angle)))
    
    # Ensure we have exactly 12
    while len(positions) < 12:
        positions.append((0, 0))
    positions = positions[:12]
    
    params = np.zeros(39)
    for i in range(12):
        params[2*i] = positions[i][0]
        params[2*i+1] = positions[i][1]
    # Angles default to 0, outer center (0,0), outer angle 0
    return params

def _initial_lattice_params():
    """Initial lattice parameters: equilateral triangular lattice."""
    d = 2 * math.sqrt(3)
    a1 = np.array([d, 0.0])
    a2 = np.array([d * math.cos(math.pi/3), d * math.sin(math.pi/3)])
    return np.array([a1[0], a1[1], a2[0], a2[1], 0.0])

def _simulated_annealing(params, deadline, best_state, obj_func, param_count):
    """Simulated annealing refinement."""
    current_params = params.copy()
    current_obj = obj_func(current_params)
    
    best_params = current_params.copy()
    best_obj = current_obj
    
    T = 0.1
    T_min = 1e-10
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > T_min:
        step_size = 0.05 * (T / 0.1) + 0.001
        perturbation = rng.normal(0, step_size, size=param_count)
        
        new_params = current_params + perturbation
        new_obj = obj_func(new_params)
        
        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            if current_obj < best_obj:
                best_params = current_params.copy()
                best_obj = current_obj
                # Update best_state
                if param_count == 5:
                    inner, outer, s = _lattice_packing(current_params)
                    if _is_valid(inner, outer, s):
                        best_state['inner'] = inner.copy()
                        best_state['outer'] = outer.copy()
                        best_state['s'] = s
                else:
                    cx_cy = current_params[:24].reshape(12, 2)
                    inner_angles = current_params[24:36]
                    outer_center = current_params[36:38]
                    outer_angle = current_params[38]
                    inner = np.zeros((12, 3))
                    inner[:, 0] = cx_cy[:, 0]
                    inner[:, 1] = cx_cy[:, 1]
                    inner[:, 2] = inner_angles
                    s = _compute_min_s(inner, outer_angle, outer_center)
                    outer = np.array([outer_center[0], outer_center[1], outer_angle])
                    if _is_valid(inner, outer, s):
                        best_state['inner'] = inner.copy()
                        best_state['outer'] = outer.copy()
                        best_state['s'] = s
        else:
            delta = new_obj - current_obj
            if delta < 0:
                current_params = new_params
                current_obj = new_obj
            else:
                if rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_params = new_params
                    current_obj = new_obj
        
        T *= cooling
    
    return best_params, best_obj

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    best_state = {'inner': None, 'outer': None, 's': float('inf')}
    
    def _update_best_from_inner(inner, outer, s):
        if s < best_state['s'] and _is_valid(inner, outer, s):
            best_state['inner'] = inner.copy()
            best_state['outer'] = outer.copy()
            best_state['s'] = s
    
    def _update_best_free(params):
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
        
        _update_best_from_inner(inner, outer, s)
    
    # Phase 1: Lattice-based optimization (reduced search space)
    lattice_params = _initial_lattice_params()
    inner, outer, s = _lattice_packing(lattice_params)
    _update_best_from_inner(inner, outer, s)
    
    # Optimize lattice with Nelder-Mead
    try:
        from scipy.optimize import minimize
        res_lattice = minimize(
            _objective_lattice, lattice_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        inner, outer, s = _lattice_packing(res_lattice.x)
        _update_best_from_inner(inner, outer, s)
    except Exception as e:
        pass
    
    # Phase 2: Free-position optimization with DE
    initial_params = _initial_hexagonal_packing()
    _update_best_free(initial_params)
    
    bounds = [(-8, 8)] * 24
    bounds += [(-180, 180)] * 12
    bounds += [(-5, 5), (-5, 5)]
    bounds += [(-30, 30)]
    
    def cb1(xk, convergence):
        _update_best_free(xk)
        return time.time() > deadline
    
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            _free_objective,
            bounds,
            maxiter=200,
            tol=1e-8,
            mut=(0.5, 1.5),
            recomb=0.7,
            seed=42,
            polish=False,
            callback=cb1
        )
        _update_best_free(res.x)
    except Exception as e:
        pass
    
    # Phase 3: Refined DE with tighter bounds
    if best_state['inner'] is not None and time.time() < deadline - 80:
        best_x = np.concatenate([best_state['inner'][:, :2].flatten(), best_state['inner'][:, 2],
                                  best_state['outer'][:2], [best_state['outer'][2]]])
        tight_bounds = []
        for i in range(24):
            tight_bounds.append((best_x[i] - 0.3, best_x[i] + 0.3))
        for i in range(12):
            tight_bounds.append((best_x[24+i] - 20, best_x[24+i] + 20))
        tight_bounds.append((best_x[36] - 0.5, best_x[36] + 0.5))
        tight_bounds.append((best_x[37] - 0.5, best_x[37] + 0.5))
        tight_bounds.append((best_x[38] - 10, best_x[38] + 10))
        
        def cb3(xk, convergence):
            _update_best_free(xk)
            return time.time() > deadline
        
        try:
            from scipy.optimize import differential_evolution
            res3 = differential_evolution(
                _free_objective, tight_bounds,
                maxiter=150, tol=1e-10,
                mut=(0.3, 1.0), recomb=0.8,
                seed=123, polish=False,
                callback=cb3
            )
            _update_best_free(res3.x)
        except Exception as e:
            pass
    
    # Phase 4: Simulated Annealing on lattice
    if time.time() < deadline - 60:
        sa_params, sa_obj = _simulated_annealing(
            lattice_params, deadline - 60, best_state, _objective_lattice, 5
        )
    
    # Phase 5: Final Nelder-Mead refinement
    if best_state['inner'] is not None and time.time() < deadline - 20:
        best_x = np.concatenate([best_state['inner'][:, :2].flatten(), best_state['inner'][:, 2],
                                  best_state['outer'][:2], [best_state['outer'][2]]])
        try:
            from scipy.optimize import minimize
            res4 = minimize(
                _free_objective, best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best_free(res4.x)
        except Exception as e:
            pass
    
    # Fallback
    if best_state['inner'] is None:
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.165, 0], [1.25, 2.165, 0],
            [-1.25, -2.165, 0], [1.25, -2.165, 0],
            [-3.75, 2.165, 0], [3.75, 2.165, 0],
            [-3.75, -2.165, 0], [3.75, -2.165, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_state['inner'], best_state['outer'], best_state['s']

# EVOLVE-BLOCK-END
