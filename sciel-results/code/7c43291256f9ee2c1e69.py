# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid_fast(inner, outer_center, outer_angle, s):
    """
    Fast validity check.
    1. Check finite.
    2. Check s >= sqrt(12).
    3. Check inner hexagons don't overlap (using SAT on vertices).
    4. Check inner vertices inside outer hexagon.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_center)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Check each inner hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        
        # Check if all vertices are inside outer hexagon
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

        # Check overlap with other inner hexagons
        for j in range(i+1, 12):
            cx2, cy2, ang2 = inner[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, ang2)
            
            # Separating Axis Test
            normals = []
            for v in [verts, verts2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_n = np.linalg.norm(normal)
                    if norm_n < 1e-9: continue
                    normal = normal / norm_n
                    normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts, normal)
                proj_j = np.dot(verts2, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False

    return True

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    shifted_centers = inner[:, :2] - outer_center
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Ensure normal points outward
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _generate_lattice_seed():
    """Generate a seed based on a triangular lattice."""
    L = 2 * np.sqrt(3) + 0.1
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * np.sqrt(3)/2 * L
            points.append([x, y, 0.0])
            
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
        
    return np.array(points)

def _get_topologies():
    """Generate candidate topologies as lists of axial coordinates (q, r)."""
    topologies = []
    
    # 3x4 Grid
    t1 = [(q, r) for r in range(3) for q in range(4)]
    topologies.append(t1)
    
    # 2x6 Grid
    t2 = [(q, r) for r in range(2) for q in range(6)]
    topologies.append(t2)
    
    # 4x3 Grid (Same as 3x4 but different orientation in space if rotated, but topology is same)
    # Let's add a "Hexagon of 7 + 5"
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Pick 5 from Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1)
    t3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)]
    topologies.append(t3)
    
    # Another variation: 3x4 but "staggered" differently? No, lattice is fixed.
    # Let's add the topology from the original seed if possible.
    # Original seed was roughly:
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (-1,1), (1,-1), (0,-1) -- Wait, standard hex neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # The seed had 6 neighbors.
    # Then 5 more.
    # Let's just use the 3 above. They cover the main compact shapes.
    
    return topologies

def _axial_to_cartesian(axial_coords, L, theta_deg):
    """Convert axial coordinates to cartesian, scaled by L and rotated by theta."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    pts = []
    for q, r in axial_coords:
        x = L * (q + r/2.0)
        y = L * (r * np.sqrt(3)/2.0)
        # Rotate
        xr = x * c - y * s
        yr = x * s + y * c
        pts.append([xr, yr])
    pts = np.array(pts)
    # Center
    pts -= np.mean(pts, axis=0)
    return pts

def _evaluate_lattice(topology, L, theta_deg, phi_deg):
    """Evaluate a specific lattice configuration."""
    centers = _axial_to_cartesian(topology, L, theta_deg)
    inner = np.column_stack([centers, np.zeros(12)])
    outer_center = np.array([0.0, 0.0])
    
    s = _compute_min_outer_side(inner, outer_center, phi_deg)
    valid = _is_valid_fast(inner, outer_center, phi_deg, s)
    
    if valid:
        return s
    else:
        return s + 100.0

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    topologies = _get_topologies()
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    
    # Initial seeds
    seeds = []
    seeds.append(_generate_lattice_seed())
    
    initial_inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.1650635094610965, 0],
            [1.25, 2.1650635094610965, 0],
            [-1.25, -2.1650635094610965, 0],
            [1.25, -2.1650635094610965, 0],
            [-3.75, 2.1650635094610965, 0],
            [3.75, 2.1650635094610965, 0],
            [-3.75, -2.1650635094610965, 0],
            [3.75, -2.1650635094610965, 0],
            [0, -4.330127018922193, 0],
        ]
    )
    seeds.append(initial_inner)
    
    # Evaluate static seeds
    for seed in seeds:
        for angle_deg in [0, 30]:
            s = _compute_min_outer_side(seed, np.array([0.0, 0.0]), angle_deg)
            if _is_valid_fast(seed, np.array([0.0, 0.0]), angle_deg, s):
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_outer_angle = angle_deg

    # Optimize each topology
    for topo in topologies:
        if time.time() > deadline - 120.0:
            break
            
        # Objective for Nelder-Mead on (L, theta, phi)
        def obj(params):
            L, theta, phi = params
            if L < 0.1: return 1e10
            return _evaluate_lattice(topo, L, theta, phi)
        
        # Initial guess
        L0 = 3.5 # Approx 2*sqrt(3)
        theta0 = 0.0
        phi0 = 0.0
        
        x0 = np.array([L0, theta0, phi0])
        
        try:
            result = minimize(
                obj, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            L_opt, theta_opt, phi_opt = result.x
            s_opt = result.fun
            
            if s_opt < best_s:
                # Reconstruct inner
                centers = _axial_to_cartesian(topo, L_opt, theta_opt)
                new_inner = np.column_stack([centers, np.zeros(12)])
                new_s = _compute_min_outer_side(new_inner, np.array([0.0, 0.0]), phi_opt)
                
                if _is_valid_fast(new_inner, np.array([0.0, 0.0]), phi_opt, new_s):
                    best_s = new_s
                    best_inner = new_inner
                    best_outer_angle = phi_opt
                    
        except Exception:
            pass

    # Final Polish: 39D Nelder-Mead on the best result
    if time.time() < deadline - 60.0 and best_inner is not None:
        def objective_39(params):
            if time.time() > deadline - 10.0:
                return 1e10
            inner = params[:36].reshape(12, 3).copy()
            outer_angle = params[36]
            
            s = _compute_min_outer_side(inner, np.array([0.0, 0.0]), outer_angle)
            if not _is_valid_fast(inner, np.array([0.0, 0.0]), outer_angle, s):
                return s + 10.0
            return s

        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        
        try:
            result = minimize(
                objective_39, x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            if result.fun < best_s:
                candidate_inner = result.x[:36].reshape(12, 3).copy()
                candidate_angle = result.x[36]
                candidate_s = _compute_min_outer_side(candidate_inner, np.array([0.0, 0.0]), candidate_angle)
                if _is_valid_fast(candidate_inner, np.array([0.0, 0.0]), candidate_angle, candidate_s):
                    best_inner = candidate_inner
                    best_s = candidate_s
                    best_outer_angle = candidate_angle
        except Exception:
            pass

    # Fallback
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s = _compute_min_outer_side(best_inner, np.array([0.0, 0.0]), 0)
        best_outer_angle = 0

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
