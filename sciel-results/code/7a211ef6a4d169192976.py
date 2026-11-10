# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, differential_evolution
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 550  # 600s wall-clock, leave safety margin
    
    def hex_vertices(cx, cy, d, a_deg):
        a = np.deg2rad(a_deg)
        return np.array([[cx + d * np.cos(a + np.deg2rad(60*i)), 
                         cy + d * np.sin(a + np.deg2rad(60*i))] for i in range(6)])
    
    def hex_edge_normals(v):
        # v is (6,2) array of vertices in CCW order
        # Edge i goes from v[i] to v[(i+1)%6]
        # The outward normal of a CCW polygon edge is rotate CW by 90 deg
        # rotate (dx,dy) CW by 90 -> (dy, -dx)
        edges = np.diff(v, axis=0, append=[v[0]])
        normals = edges.copy()
        normals[:, 0] = edges[:, 1]
        normals[:, 1] = -edges[:, 0]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals  # (6,2) outward unit normals
    
    def is_valid_packing(inner_hex_data, outer_hex_data, s, tol=1e-8):
        """Check validity according to the task contract."""
        # Check finite
        if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
            return False, "non-finite"
        if not np.isfinite(s):
            return False, "s non-finite"
        if s < np.sqrt(12) - 1e-8:
            return False, "s too small"
        
        cx_o, cy_o, a_o = outer_hex_data
        outer_v = hex_vertices(cx_o, cy_o, s, a_o)
        
        # Check outer vertices distinct
        for i in range(6):
            for j in range(i+1, 6):
                if np.linalg.norm(outer_v[i] - outer_v[j]) < 1e-10:
                    return False, "outer vertices not distinct"
        
        # Check unit edge lengths
        outer_edges = np.diff(outer_v, axis=0, append=[outer_v[0]])
        for i in range(6):
            if abs(np.linalg.norm(outer_edges[i]) - s) > 1e-8:
                return False, "outer edge length wrong"
        
        inner_v_list = []
        for i in range(12):
            cx, cy, a = inner_hex_data[i]
            v = hex_vertices(cx, cy, 1.0, a)
            inner_v_list.append(v)
            # Check unit edges
            edges = np.diff(v, axis=0, append=[v[0]])
            for j in range(6):
                if abs(np.linalg.norm(edges[j]) - 1.0) > 1e-8:
                    return False, "inner edge length wrong"
        
        # Check inner hexagons are separated (SAT)
        for i in range(12):
            for j in range(i+1, 12):
                vi = inner_v_list[i]
                vj = inner_v_list[j]
                # Separating axes: edge normals of both hexagons
                normals_i = hex_edge_normals(vi)
                normals_j = hex_edge_normals(vj)
                all_normals = np.vstack([normals_i, normals_j])
                
                proj_i = vi @ all_normals.T  # (6, 12)
                proj_j = vj @ all_normals.T
                min_i = proj_i.min(axis=0)
                max_i = proj_i.max(axis=0)
                min_j = proj_j.min(axis=0)
                max_j = proj_j.max(axis=0)
                
                # Check if any axis has a gap
                # Gap exists if max_i + eps < min_j or max_j + eps < min_i
                gap1 = min_j - max_i  # if positive, separated
                gap2 = min_i - max_j  # if positive, separated
                
                # If neither gap is positive, they overlap or touch
                # For validity, we need separation: some axis must have gap > -1e-6
                # Actually the task says "some edge normal... must give projection intervals more than 1e-6 apart"
                # This means we need at least one axis where the intervals are disjoint with margin
                # Wait, re-reading: "two inner hexagons not separated: some edge normal of either one... must give projection intervals more than 1e-6 apart"
                # This is the INVALID condition. So for VALID, NO axis should give a gap > 1e-6? 
                # No, that's wrong. SAT says: if there EXISTS a separating axis, the shapes are disjoint.
                # The invalid condition is: "two inner hexagons not separated" which happens when NO separating axis exists.
                # The description says the invalid check is: "some edge normal... must give projection intervals more than 1e-6 apart"
                # I think this means: for the packing to be INVALID, we check if the hexagons overlap. 
                # Let me re-read carefully.
                # "two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
                # This is confusing. Standard SAT: if ANY axis shows separation, they are disjoint.
                # If NO axis shows separation, they may overlap.
                # The phrasing "must give... apart" is likely a description of the test for separation.
                # Let's assume: Valid iff for all pairs, there EXISTS an axis where gap > 1e-6.
                # If for all axes, gap <= 1e-6, they are too close/overlapping.
                
                is_separated = False
                for k in range(all_normals.shape[0]):
                    if gap1[k] > 1e-6 or gap2[k] > 1e-6:
                        is_separated = True
                        break
                if not is_separated:
                    return False, f"hexagons {i},{j} not separated"
        
        # Check inner vertices inside outer hexagon
        # For each inner vertex, check it's on the correct side of every outer edge
        outer_normals = hex_edge_normals(outer_v)
        # For a CCW polygon, the interior is to the LEFT of each edge.
        # The outward normal points away from interior.
        # So a point p is inside if (p - v[i]) dot outward_normal[i] <= 0 for all i.
        # Or equivalently, (p - v[i]) cross edge_vector[i] >= 0 if vertices are CCW.
        # Let's use the cross product method as specified in the task:
        # "for every outer edge (p1, p2) in vertex order, cross(p2 - p1, v - p1) must be >= -1e-6"
        
        for i in range(12):
            for vertex in inner_v_list[i]:
                for j in range(6):
                    p1 = outer_v[j]
                    p2 = outer_v[(j+1)%6]
                    edge_vec = p2 - p1
                    point_vec = vertex - p1
                    cross_val = edge_vec[0]*point_vec[1] - edge_vec[1]*point_vec[0]
                    if cross_val < -1e-6:
                        return False, f"vertex outside outer hex"
        
        return True, "valid"
    
    def compute_score(inner_hex_data, outer_hex_data, s):
        if s <= 0:
            return 0
        return 1.0 / s
    
    # Initial seed packing (from the provided program, but we will improve it)
    inner_seed = np.array(
        [
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
            [0, -4, 0],
        ]
    )
    outer_seed = np.array([0, 0, 0])
    s_seed = 8.0
    
    # Let's try to find a better s for the seed packing using LP
    # The LP minimizes s such that all inner vertices are inside the outer hexagon.
    # We can fix the inner positions and optimize the outer hexagon (center, angle, size).
    # This is a non-linear problem because of the angle and size.
    # However, if we fix the angle, the constraints become linear in s and center.
    
    def minimize_outer_hex(inner_positions, angle_deg):
        # inner_positions: (12, 2)
        # angle_deg: orientation of outer hexagon
        # Find min s and center (cx, cy) such that all vertices are inside.
        # Vertices of inner hexagons (assuming unit side, angle 0 for simplicity, or given)
        # Let's assume inner hexagons have angle 0 for this LP subproblem.
        # Actually, we should pass the full inner data.
        pass # Placeholder, will implement properly below

    # Let's use a more direct approach: 
    # 1. Fix inner hexagon orientations (all 0 for now, or optimize later).
    # 2. Optimize inner positions and outer hexagon parameters jointly? 
    # This is hard. Let's stick to the known good structure: 12 hexagons in a compact cluster.
    
    # The seed is a "flower" shape. Let's see if we can reduce s for this exact configuration.
    # We can optimize the outer hexagon center and angle for a fixed s, or minimize s.
    # Let's minimize s for the fixed inner positions.
    
    def get_min_s_for_fixed_inner(inner_hex_data, angle_candidates=None):
        # We'll try a few angles for the outer hexagon to find the tightest fit.
        best_s = np.inf
        best_center = np.array([0,0])
        best_angle = 0
        
        if angle_candidates is None:
            angle_candidates = np.linspace(0, 60, 13)
        
        for a_o in angle_candidates:
            # For this angle, the outer hexagon is defined by 6 half-planes.
            # We want to find the smallest s and center (cx, cy) such that
            # all inner vertices satisfy: (p - c) dot n_i <= s * apothem? 
            # No, the distance from center to edge is s * apothem_unit?
            # Apothem of hex with side s is s * sqrt(3)/2.
            # The constraint is: for each outer edge normal n_i (unit, pointing out),
            # (p - c) dot n_i <= s * sqrt(3)/2
            # => p dot n_i - c dot n_i <= s * sqrt(3)/2
            # => c dot n_i >= p dot n_i - s * sqrt(3)/2
            
            # This is linear in c and s.
            # Minimize s
            # Variables: [s, cx, cy]
            # Constraints: for each vertex p and each normal n_i:
            # -n_i_x * cx - n_i_y * cy + (sqrt(3)/2) * s >= p dot n_i
            # A_ub * [s, cx, cy] <= b_ub
            # -sqrt(3)/2 * s + n_i_x * cx + n_i_y * cy <= - (p dot n_i)
            
            # Let's collect all inner vertices
            vertices = []
            for i in range(12):
                cx, cy, a = inner_hex_data[i]
                v = hex_vertices(cx, cy, 1.0, a)
                vertices.append(v)
            vertices = np.vstack(vertices) # (72, 2)
            
            # Compute normals for the outer hexagon at angle a_o
            outer_v = hex_vertices(0, 0, 1.0, a_o) # Unit hexagon, just to get normals
            normals = hex_edge_normals(outer_v) # (6, 2)
            
            A_ub = []
            b_ub = []
            for p in vertices:
                for n in normals:
                    # Constraint: n dot (p - c) <= s * sqrt(3)/2
                    # n_x p_x + n_y p_y - n_x c_x - n_y c_y <= s * sqrt(3)/2
                    # -n_x c_x - n_y c_y - sqrt(3)/2 s <= - (n_x p_x + n_y p_y)
                    A_ub.append([-n[0], -n[1], -np.sqrt(3)/2])
                    b_ub.append(-(n[0]*p[0] + n[1]*p[1]))
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Objective: minimize s
            c_obj = [1, 0, 0]
            
            # Bounds: s >= 0, cx, cy unbounded (or large bounds)
            bounds = [(0, None), (None, None), (None, None)]
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                s_val = res.x[0]
                cx_val = res.x[1]
                cy_val = res.x[2]
                if s_val < best_s:
                    best_s = s_val
                    best_center = np.array([cx_val, cy_val])
                    best_angle = a_o
        
        return best_s, best_center, best_angle
    
    # Step 1: Optimize the outer hexagon for the seed inner positions
    s_opt, center_opt, angle_opt = get_min_s_for_fixed_inner(inner_seed)
    
    # Construct the candidate
    outer_hex_data = np.array([center_opt[0], center_opt[1], angle_opt])
    inner_hex_data = inner_seed.copy()
    s_val = s_opt
    
    # Verify validity
    valid, msg = is_valid_packing(inner_hex_data, outer_hex_data, s_val)
    if not valid:
        print(f"Seed optimization failed validity: {msg}")
        # Fall back to seed if invalid
        inner_hex_data = inner_seed
        outer_hex_data = outer_seed
        s_val = s_seed
    
    current_score = compute_score(inner_hex_data, outer_hex_data, s_val)
    print(f"Initial optimized score: {current_score}")
    
    # Step 2: Try to improve by perturbing inner positions?
    # This is complex. Let's try a local search on the inner positions and outer parameters.
    # However, the "floral" arrangement is likely optimal for this symmetry class.
    # The main gain came from LP optimization of the outer shell.
    
    # Let's check if we can rotate the inner cluster to fit better.
    # The seed has 6-fold symmetry. The outer hexagon also has 6-fold symmetry.
    # So relative rotation doesn't change the fit if the cluster is perfectly symmetric.
    # But the cluster might not be perfectly symmetric in terms of vertex distances.
    
    # Let's try a few inner cluster rotations and re-optimize the outer shell.
    best_total_score = current_score
    best_inner = inner_hex_data.copy()
    best_outer = outer_hex_data.copy()
    best_s = s_val
    
    for rot in np.linspace(0, 60, 7):
        if time.time() > deadline: break
        # Rotate inner positions around the center of the inner cluster
        # The seed center is roughly (0,0) but let's compute centroid
        centroid = inner_seed[:, :2].mean(axis=0)
        angles = np.deg2rad(rot)
        c, s_rot = np.cos(angles), np.sin(angles)
        R = np.array([[c, -s_rot], [s_rot, c]])
        
        inner_rotated = inner_seed.copy()
        for i in range(12):
            p = inner_seed[i, :2] - centroid
            p_new = R @ p
            inner_rotated[i, :2] = p_new + centroid
            inner_rotated[i, 2] = inner_seed[i, 2] + rot # Rotate orientation too?
            # If we rotate the positions but not the orientation, the hexagons are tilted relative to each other?
            # No, if we rotate the whole cluster, we should rotate the orientations too.
            # But if all orientations are 0, and we rotate positions by rot, the hexagons are now "tilted" relative to the lattice?
            # Actually, if I rotate the center positions but keep angle=0, the hexagons are still aligned with axes, but their centers are rotated.
            # This breaks the symmetry of the cluster relative to the axes.
            # Let's keep it simple: rotate positions AND angles.
            # But the LP assumes specific vertices.
            pass
        
        # Re-optimize outer shell
        s_new, center_new, angle_new = get_min_s_for_fixed_inner(inner_rotated)
        
        # Construct candidate
        outer_new = np.array([center_new[0], center_new[1], angle_new])
        
        # Validate
        valid, msg = is_valid_packing(inner_rotated, outer_new, s_new)
        if valid:
            score_new = compute_score(inner_rotated, outer_new, s_new)
            if score_new > best_total_score:
                best_total_score = score_new
                best_inner = inner_rotated
                best_outer = outer_new
                best_s = s_new
                print(f"Found better score with rot={rot}: {score_new}")
    
    # Return the best found
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
