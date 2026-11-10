import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _sat_gap(v1, v2):
    """
    Compute the minimum gap between two convex polygons using SAT.
    Returns positive if separated, negative if overlapping (penetration depth).
    """
    n1 = len(v1)
    n2 = len(v2)
    min_gap = np.inf
    
    for verts in [v1, v2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i+1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            gap = min(min2 - max1, min1 - max2)
            if gap < min_gap:
                min_gap = gap
    
    return min_gap

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    gap = _sat_gap(v1, v2)
    return gap > tol

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        # Check unit edges
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        # Check inside outer
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all points at distance up to 1 from each center.
    """
    normals = _hex_edge_normals(0.0)
    unit_verts = _hex_vertices(0, 0, 1.0, 0.0)
    
    max_required_s = 0.0
    for center in centers:
        for n in normals:
            support = np.max(unit_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _smooth_sat_penalty(v1, v2, beta=100.0):
    """
    Compute a smooth penalty for overlap between two polygons.
    Uses soft-min of the SAT gaps.
    """
    gaps = []
    for verts in [v1, v2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i+1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            gap = min(min2 - max1, min1 - max2)
            gaps.append(gap)
    
    if not gaps:
        return 0.0
    
    gaps_arr = np.array(gaps)
    # Soft min: if all gaps are positive, penalty is 0
    # If any gap is negative, we want a smooth measure of overlap
    min_gap = np.min(gaps_arr)
    if min_gap > 0:
        return 0.0
    
    # Use soft-min to get a smooth approximation of min(gaps)
    # soft_min(x, beta) = -log(sum(exp(-beta*x)))/beta
    # This approximates min(x) as beta -> inf
    # We want penalty = max(0, -min_gap)^2, but smoothed
    # Let's use: penalty = -soft_min(gaps, beta) when min_gap < 0
    # soft_min(gaps) approximates min(gaps)
    # We want to penalize when min_gap < 0
    
    # Use a simpler smooth approach:
    # penalty = sum(max(0, -gap)^2) with smoothing
    # Or use: penalty = -beta * log(1 + sum(exp(-beta * max(0, gap))))
    # This is getting complex. Let's use the epigraph form:
    # Minimize t + lambda * t^2 subject to t >= -gap_i for all i
    # This is equivalent to minimizing max(0, -min_gap) with smoothness
    
    # Simpler: use the fact that for two hexagons, the minimum gap axis
    # can be computed directly
    # For now, use a simple smooth max:
    # smooth_max(0, -gaps) with soft-max
    neg_gaps = -gaps_arr
    # We want soft_max(0, neg_gaps)
    # = (1/beta) * log(1 + sum(exp(beta * neg_gaps)))
    # But this can overflow. Use log-sum-exp trick
    m = np.max(neg_gaps)
    if m < -1/beta:
        return 0.0
    soft_max = m + (1/beta) * np.log(1 + np.sum(np.exp(beta * (neg_gaps - m))))
    if soft_max < 0:
        return 0.0
    return soft_max ** 2

def _boundary_penalty(verts, outer_verts, beta=100.0):
    """
    Compute a smooth penalty for vertices outside the outer hexagon.
    """
    penalty = 0.0
    for v in verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
            if cross < 0:
                # Smooth penalty
                penalty += (-cross) ** 2
    return penalty

def _objective(params):
    """
    Objective function for optimization.
    params: [x1, y1, x2, y2, ..., x12, y12, s]
    Returns s + boundary_penalty + collision_penalty
    """
    inner_xy = params[:24].reshape(12, 2)
    s_val = params[24]
    
    if s_val < np.sqrt(12):
        return 1e6
    
    outer_verts = _hex_vertices(0, 0, s_val, 0)
    
    penalty = 0.0
    
    # Check each inner hexagon
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
        inner_verts_list.append(verts)
        
        # Boundary penalty
        penalty += 100.0 * _boundary_penalty(verts, outer_verts)
    
    # Check separation from other hexagons
    for i in range(12):
        for j in range(i+1, 12):
            verts1 = inner_verts_list[i]
            verts2 = inner_verts_list[j]
            penalty += 100.0 * _smooth_sat_penalty(verts1, verts2)
    
    return s_val + penalty

def _objective_grad(params):
    """
    Objective with numerical gradient.
    """
    return _objective(params)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial placement in a honeycomb pattern
    # 4 columns, 3 rows
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    inner_data = np.array(positions)
    centers = inner_data[:, :2]
    
    # Compute minimum enclosing hexagon side
    s = _min_enclosing_hex_side(centers)
    s *= 1.001
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Validate initial solution
    if not _validate(inner_data, outer_data, s):
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
    
    best_value = 1.0 / s
    best_result = (inner_data.copy(), outer_data.copy(), s)
    
    # Initial params
    init_params = np.zeros(25)
    init_params[:24] = inner_data[:, :2].flatten()
    init_params[24] = s
    
    # Use L-BFGS-B for faster convergence with gradient
    # Then polish with Nelder-Mead
    
    def try_optimize(params0, method, options, label=""):
        nonlocal best_value, best_result
        try:
            result = minimize(
                _objective,
                params0,
                method=method,
                options=options
            )
            
            if result.fun < 100:
                final_params = result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
                            print(f"{label}: improved to {value:.6f} (s={s_final:.6f})")
                            return final_params
        except Exception as e:
            print(f"{label}: exception {e}")
        return None
    
    # Multiple restarts with different initializations
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            perturbation = 0.5 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = s + 0.5 * np.random.randn()
        
        # First try L-BFGS-B
        params1 = try_optimize(
            params0,
            'L-BFGS-B',
            {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
            f"LBFGS-B restart {restart}"
        )
        
        if time.time() > deadline - 30:
            break
        
        # Polish with Nelder-Mead
        if params1 is not None:
            try_optimize(
                params1,
                'Nelder-Mead',
                {'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
                f"NM polish restart {restart}"
            )
        else:
            # Try Nelder-Mead directly
            try_optimize(
                params0,
                'Nelder-Mead',
                {'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True},
                f"NM restart {restart}"
            )
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 10:
        final_params = np.zeros(25)
        final_params[:24] = best_result[0][:, :2].flatten()
        final_params[24] = best_result[2]
        
        try_optimize(
            final_params,
            'Nelder-Mead',
            {'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            "Final polish"
        )
    
    return best_result
