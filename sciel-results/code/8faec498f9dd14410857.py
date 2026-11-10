# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Correct close-packed lattice for unit hexagons (side 1)
    # Neighbor distance is sqrt(3)
    # Using basis vectors for a hex grid
    e1 = np.array([1.5, math.sqrt(3)/2])
    e2 = np.array([0, math.sqrt(3)])
    
    # Generate 12 points in a 3x4 grid on this lattice
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * e1 + j * e2
            positions.append([pos[0], pos[1], 0.0])
            
    inner = np.array(positions)
    
    # Center the configuration
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Find optimal outer angle
    best_s = float('inf')
    best_inner = inner.copy()
    best_outer = np.array([0, 0, 0.0])
    
    all_verts = _get_all_inner_vertices(inner)
    
    for angle_deg in np.linspace(0, 60, 61):
        s_cand = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_inner = inner.copy()
            best_outer = np.array([0, 0, angle_deg])
            
    # Local optimization using L-BFGS-B with penalty method
    # Variables: 12 centers (x,y), 12 angles, s, outer_angle
    # Total 38 variables
    
    def _penalty_objective(x, penalty_weight=100.0):
        """
        Penalty-based objective: minimize s + penalty for violations.
        x = [cx1..cx12, cy1..cy12, angle1..angle12, s, outer_angle]
        Total: 12*2 + 12 + 1 + 1 = 38 variables
        """
        cx = x[0:12]
        cy = x[12:24]
        angles = x[24:36]
        s = x[36]
        outer_angle = x[37]
        
        # Compute inner hexagon vertices
        inner_verts_list = []
        for i in range(12):
            v = _hex_vertices(cx[i], cy[i], 1.0, angles[i])
            inner_verts_list.append(v)
            
        # Compute outer hexagon vertices
        outer_verts = _hex_vertices(0, 0, s, outer_angle)
        
        # Objective: minimize s
        obj = s
        
        # Penalty for inner-inner separation violations
        # For each pair, check SAT. If not separated, add penalty based on overlap depth
        for i in range(12):
            for j in range(i + 1, 12):
                # Check separation
                separated = False
                min_overlap = 0
                for verts in (inner_verts_list[i], inner_verts_list[j]):
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k + 1) % 6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / (np.linalg.norm(n) + 1e-15)
                        proj1 = inner_verts_list[i] @ n
                        proj2 = inner_verts_list[j] @ n
                        # If projections overlap, compute overlap depth
                        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                        if overlap > 0:
                            if overlap > min_overlap:
                                min_overlap = overlap
                            # Not separated
                        else:
                            # Separated along this axis
                            separated = True
                            break
                    if separated:
                        break
                
                if not separated:
                    # Penalty for overlap
                    obj += penalty_weight * (min_overlap ** 2)
        
        # Penalty for inner vertices outside outer hexagon
        for i in range(12):
            for v in inner_verts_list[i]:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < 0:
                        # Violation: vertex is outside
                        # Approximate penetration depth as -cross / |edge|
                        edge_len = np.linalg.norm(p2 - p1)
                        depth = -cross / (edge_len + 1e-15)
                        obj += penalty_weight * (depth ** 2)
        
        return obj

    # Initial guess from best lattice configuration
    x0 = np.zeros(38)
    x0[0:12] = best_inner[:, 0]
    x0[12:24] = best_inner[:, 1]
    x0[24:36] = best_inner[:, 2]
    x0[36] = best_s
    x0[37] = best_outer[2]
    
    # Bounds
    # Centers can move freely, angles 0-360, s > 0, outer_angle 0-360
    bounds = []
    for i in range(24):
        bounds.append((None, None))
    for i in range(12):
        bounds.append((0, 360))
    bounds.append((1.0, None))  # s must be positive
    bounds.append((0, 360))
    
    # Run L-BFGS-B
    # We do multiple restarts with different penalty weights and initial perturbations
    best_result_s = best_s
    best_result_inner = best_inner.copy()
    best_result_outer = best_outer.copy()
    
    np.random.seed(42)
    
    # Increase penalty weight gradually to enforce constraints
    penalty_weights = [10, 50, 100, 500, 1000]
    
    for pw in penalty_weights:
        if time.time() > deadline:
            break
            
        # Run optimization
        try:
            res = minimize(
                _penalty_objective,
                x0,
                args=(pw,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                x_opt = res.x
                cx_opt = x_opt[0:12]
                cy_opt = x_opt[12:24]
                angles_opt = x_opt[24:36]
                s_opt = x_opt[36]
                outer_angle_opt = x_opt[37]
                
                # Construct inner and outer data
                trial_inner = np.column_stack((cx_opt, cy_opt, angles_opt))
                trial_outer = np.array([0, 0, outer_angle_opt])
                
                # Check validity
                if _is_valid(trial_inner, trial_outer, s_opt):
                    if s_opt < best_result_s:
                        best_result_s = s_opt
                        best_result_inner = trial_inner.copy()
                        best_result_outer = trial_outer.copy()
                        
                # Update x0 for next iteration
                x0 = x_opt
        except Exception:
            pass
        
        # Add small random perturbation to escape local minima
        if time.time() > deadline:
            break
        x0[0:24] += np.random.normal(0, 0.01, 24)
        x0[24:36] += np.random.normal(0, 0.1, 12)
        
    # Final check and return
    if not _is_valid(best_result_inner, best_result_outer, best_result_s):
        # Fallback to initial lattice
        best_result_inner = best_inner
        best_result_outer = best_outer
        best_result_s = best_s

    return best_result_inner, best_result_outer, best_result_s

# EVOLVE-BLOCK-END
