import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """Compute minimum side length of regular hexagon enclosing all unit hexagons."""
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    # Normals to edges of the outer hexagon
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    """Validate according to task contract."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i + 1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k + 1) % 6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12:
                    continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated:
                continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k + 1) % 6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12:
                    continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Hexagonal lattice positions for 12 hexagons (side 1):
    # Center + 6 ring neighbors + 5 more from second ring
    # Using axial coordinates (q, r) -> cartesian
    def axial_to_cartesian(q, r, d):
        """Convert axial coordinates to cartesian for hexagon centers."""
        x = d * (q + 0.5 * r)
        y = d * (np.sqrt(3) / 2 * r)
        return np.array([x, y])
    
    # 12 positions in hexagonal lattice (axial coords)
    # Center (0,0), 6 neighbors, then 5 from second ring
    axial_coords = [
        (0, 0),      # center
        (1, 0),      # 6 neighbors
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        # Second ring (pick 5 to make 12 total)
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
        (2, -2),
    ]
    
    def get_positions(d, angle_deg):
        """Get positions of 12 hexagons given lattice spacing d and outer angle."""
        positions = np.array([axial_to_cartesian(q, r, d) for q, r in axial_coords])
        return positions
    
    def objective(params):
        d = params[0]
        angle_deg = params[1]
        if d < 1.9:  # Must be at least ~2 for non-overlapping unit hexagons
            return 1e10
        positions = get_positions(d, angle_deg)
        s = _min_outer_side(positions, angle_deg)
        return s
    
    # Multi-start Nelder-Mead over 2D parameter space (d, angle)
    np.random.seed(42)
    
    # Initial guesses: vary d from 1.95 to 3.5, angle from 0 to 60
    initial_guesses = []
    for d0 in [1.95, 2.0, 2.05, 2.1, 2.2, 2.3, 2.5, 2.8, 3.0, 3.5]:
        for a0 in [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]:
            initial_guesses.append([d0, a0])
    
    # Also add some random starts
    for _ in range(20):
        d0 = np.random.uniform(1.95, 4.0)
        a0 = np.random.uniform(0, 60)
        initial_guesses.append([d0, a0])
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                objective,
                np.array(guess),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
        except Exception:
            continue
        
        d_opt = res.x[0]
        angle_opt = res.x[1]
        
        if d_opt < 1.9:
            continue
        
        positions = get_positions(d_opt, angle_opt)
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0
        
        outer_candidate = np.array([0.0, 0.0, angle_opt])
        s_candidate = _min_outer_side(positions, angle_opt)
        
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s_candidate
    
    # Also try: optimize individual positions with Nelder-Mead (24 vars + 1 angle)
    # Start from the best lattice solution found above
    if best_inner is not None:
        if time.time() < deadline - 100:
            # Refine: allow individual position adjustments
            def objective_full(x):
                positions = x[:24].reshape(12, 2)
                angle_deg = x[24]
                s = _min_outer_side(positions, angle_deg)
                return s
            
            x0 = np.concatenate([best_inner[:, :2].flatten(), [best_outer[2]]])
            
            try:
                res = minimize(
                    objective_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                positions = res.x[:24].reshape(12, 2)
                angle_deg = res.x[24]
                
                inner_candidate = np.zeros((12, 3))
                inner_candidate[:, :2] = positions
                inner_candidate[:, 2] = 0.0
                
                outer_candidate = np.array([0.0, 0.0, angle_deg])
                s_candidate = _min_outer_side(positions, angle_deg)
                
                if _check_validity(inner_candidate, outer_candidate, s_candidate):
                    score_candidate = 1.0 / s_candidate
                    if score_candidate > best_score:
                        best_score = score_candidate
                        best_inner = inner_candidate.copy()
                        best_outer = outer_candidate.copy()
                        best_s = s_candidate
            except Exception:
                pass
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: standard hexagonal packing
        h = np.sqrt(3) / 2
        inner_fallback = np.array([
            [0, 0, 0],
            [2*h, 0, 0],
            [h, 1.5, 0],
            [-h, 1.5, 0],
            [-2*h, 0, 0],
            [-h, -1.5, 0],
            [h, -1.5, 0],
            [3*h, 1.5, 0],
            [0, 3, 0],
            [-3*h, 1.5, 0],
            [-3*h, -1.5, 0],
            [3*h, -1.5, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback
