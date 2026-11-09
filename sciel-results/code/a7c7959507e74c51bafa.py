import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def hexagon_packing_11():
    start_time = time.time()
    deadline = 580  # Leave 20s margin for 600s limit

    # Constants
    n_inner = 11
    epsilon = 1e-6
    apothem_ratio = np.sqrt(3) / 2.0

    # Outer hexagon normals (for angle 0, flat-top? No, vertex at 0 is pointy-side? 
    # Prompt: v_k = (cx + s*cos(a + 60i), ...).
    # If a=0, v0=(s,0). v1=(s/2, s*sqrt(3)/2).
    # Edge 0-1 midpoint: (3s/4, s*sqrt(3)/4).
    # Edge vector: (-s/2, s*sqrt(3)/2).
    # Normal: (s*sqrt(3)/2, s/2). Angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330.
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    
    # Inner hexagon normals (same set, since angle is fixed to 0 for all)
    inner_normals = outer_normals

    def get_hex_vertices(x, y, angle_deg=0):
        # Returns (6, 2) array of vertices for a unit hexagon at (x,y) with given angle
        a_rad = np.radians(angle_deg)
        angles = a_rad + np.radians([0, 60, 120, 180, 240, 300])
        return np.column_stack([x + np.cos(angles), y + np.sin(angles)])

    def check_validity(inner, outer, side):
        # inner: (11, 3)
        # outer: (3,)
        # side: float
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
            
        ox, oy, oa = outer
        # Shift coordinates so outer center is origin for validation
        inner_shifted = inner.copy()
        inner_shifted[:, 0] -= ox
        inner_shifted[:, 1] -= oy
        
        # Check edge lengths
        for i in range(11):
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            # v is unit hex. Check lengths
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False

        # Check non-overlap
        for i in range(11):
            vi = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            for j in range(i+1, 11):
                vj = get_hex_vertices(inner_shifted[j, 0], inner_shifted[j, 1], inner_shifted[j, 2])
                separated = False
                # We need to check separation on the 12 edge normals of the pair.
                # The task says: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
                # The 12 normals are the 6 normals of hex i and 6 normals of hex j.
                normals_ij = np.vstack([inner_normals, inner_normals]) # This is wrong if angles differ.
                
                # Let's compute normals properly for each hex
                def get_hex_normals(x, y, angle_deg):
                    a_rad = np.radians(angle_deg)
                    # Normals are at angles a + 30, a + 90, ...
                    # Or simply rotate the base normals by angle_deg
                    # Base normals (for angle 0): 30, 90, 150, 210, 270, 330
                    base_angles = np.radians([30, 90, 150, 210, 270, 330])
                    angles = base_angles + a_rad
                    return np.column_stack([np.cos(angles), np.sin(angles)])
                
                n_i = get_hex_normals(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
                n_j = get_hex_normals(inner_shifted[j, 0], inner_shifted[j, 1], inner_shifted[j, 2])
                all_normals = np.vstack([n_i, n_j])
                
                for n in all_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    gap = min(min_j - max_i, min_i - max_j)
                    if gap > epsilon:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        # Outer hexagon vertices
        oa_rad = np.radians(oa)
        outer_v = np.column_stack([
            ox + side * np.cos(oa_rad + np.radians(60*k)),
            oy + side * np.sin(oa_rad + np.radians(60*k))
        ] for k in range(6))
        
        # Edges of outer hexagon
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i+1)%6]
            edge_vec = p2 - p1
            # Normal pointing INWARDS?
            # Prompt: cross(p2-p1, p-p1) >= -1e-6
            # This checks if p is to the left of p1->p2 (CCW) or on the line.
            # For a CCW polygon, inside is to the left.
            # Our outer_v is generated CCW?
            # cos(0)=1, sin(0)=0. cos(60)=0.5, sin(60)=0.866.
            # (1,0) to (0.5, 0.866). Vector (-0.5, 0.866).
            # Cross product with (0,1) (inside) -> -0.5*1 - 0.866*0 = -0.5 < 0.
            # Wait.
            # Let's check the winding.
            # v0=(1,0), v1=(0.5, 0.866).
            # If we walk v0->v1, the interior is to the RIGHT?
            # Center (0,0) is to the right of v0->v1?
            # Vector v1-v0 = (-0.5, 0.866).
            # Vector v0-C = (1,0).
            # Cross(v1-v0, v0-C) = (-0.5)(0) - (0.866)(1) = -0.866.
            # So interior is to the RIGHT.
            # Prompt: cross(p2-p1, p-p1) >= -1e-6.
            # If p is inside, cross should be negative?
            # Let's re-read: "every inner vertex p satisfies cross(p2 - p1, p - p1) >= -1e-6".
            # If the polygon is defined CW, inside is to the left.
            # If CCW, inside is to the right (negative cross).
            # My outer_v is CCW.
            # So for p inside, cross(p2-p1, p-p1) should be <= 0.
            # The condition >= -1e-6 allows small positive values (outside) or negative (inside).
            # So it effectively means "not strictly outside".
            
            for i_h in range(11):
                p = np.array([inner_shifted[i_h, 0], inner_shifted[i_h, 1]])
                # Get all vertices of this inner hex
                vi = get_hex_vertices(inner_shifted[i_h, 0], inner_shifted[i_h, 1], inner_shifted[i_h, 2])
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -epsilon:
                        return False
        return True

    def compute_min_enclosing_side(coords, angles):
        # coords: (11, 2), angles: (11,)
        # Returns minimal side length of a regular hexagon enclosing all these hexagons,
        # with the enclosing hexagon centered at origin and aligned to axes (angle 0).
        # Actually, the enclosing hexagon can be rotated. But by symmetry, we can fix its angle to 0
        # and allow the inner hexagons to be rotated? No, the problem allows any angle for inner hexagons.
        # However, for a given set of inner hexagons (positions and angles), the minimal enclosing regular hexagon
        # (centered at origin) with any orientation can be found by checking 6 orientations?
        # Actually, the minimal enclosing regular hexagon (with fixed center) for a set of points is determined
        # by the maximum projection on the 6 normals of the hexagon.
        # If we fix the enclosing hexagon's angle to 0, its normals are at 30, 90, 150, 210, 270, 330.
        # The side length S must satisfy: for all normals n_k, max_p dot(p, n_k) <= S * apothem.
        # So S >= max_k (max_p dot(p, n_k)) / apothem.
        # But the enclosing hexagon can also be rotated. To find the global minimum, we might need to optimize over its angle.
        # However, for the optimization, we can treat the enclosing hexagon's angle as fixed (0) and let the inner hexagons
        # move and rotate to minimize the maximum projection. This is a valid approach because any optimal solution can be
        # rotated to have the enclosing hexagon at angle 0.
        
        # Compute all vertices of all inner hexagons
        all_vertices = []
        for i in range(11):
            v = get_hex_vertices(coords[i, 0], coords[i, 1], angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        # Project onto the 6 normals of the enclosing hexagon (angle 0)
        max_proj = 0
        for n in outer_normals:
            proj = np.dot(all_vertices, n)
            max_p = np.max(proj)
            # The constraint is max_p <= S * apothem_ratio
            # So S >= max_p / apothem_ratio
            s_required = max_p / apothem_ratio
            if s_required > max_proj:
                max_proj = s_required
        
        return max_proj

    def objective(x):
        # x: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
        coords = x[0:33].reshape(11, 2)
        angles = x[2:34:3]
        return compute_min_enclosing_side(coords, angles)

    def make_non_overlap_constraint(i, j):
        def constraint(x):
            ci = x[3*i:3*i+2]
            ai = x[3*i+2]
            cj = x[3*j:3*j+2]
            aj = x[3*j+2]
            
            vi = get_hex_vertices(ci[0], ci[1], ai)
            vj = get_hex_vertices(cj[0], cj[1], aj)
            
            def get_hex_normals(angle_deg):
                a_rad = np.radians(angle_deg)
                base_angles = np.radians([30, 90, 150, 210, 270, 330])
                angles = base_angles + a_rad
                return np.column_stack([np.cos(angles), np.sin(angles)])
            
            n_i = get_hex_normals(ai)
            n_j = get_hex_normals(aj)
            all_normals = np.vstack([n_i, n_j])
            
            max_gap = -np.inf
            for n in all_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                gap = min(min_j - max_i, min_i - max_j)
                if gap > max_gap:
                    max_gap = gap
            return max_gap - epsilon
        
        return constraint

    # Build constraints list
    cons = []
    for i in range(11):
        for j in range(i+1, 11):
            cons.append({'type': 'ineq', 'fun': make_non_overlap_constraint(i, j)})

    # Initial Guess: 6-fold symmetric layout
    x0 = np.zeros(33)
    # Hex 0 at center
    x0[0] = 0
    x0[1] = 0
    x0[2] = 0
    # Hex 1-6 in ring
    for i in range(1, 7):
        angle = np.radians(60 * (i-1))
        r = np.sqrt(3)
        x0[3*i] = r * np.cos(angle)
        x0[3*i+1] = r * np.sin(angle)
        x0[3*i+2] = 0
    # Hex 7-10 further out
    angles_pos = [30, 150, 210, 330]
    for i in range(7, 11):
        a = np.radians(angles_pos[i-7])
        r = 2.5
        x0[3*i] = r * np.cos(a)
        x0[3*i+1] = r * np.sin(a)
        x0[3*i+2] = 0

    best_result = None
    best_score = -np.inf

    # Try SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', 
                      bounds=[(-10, 10)]*22 + [(-180, 180)]*11, 
                      constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            x_opt = res.x
            inner = np.zeros((11, 3))
            for i in range(11):
                inner[i, 0] = x_opt[3*i]
                inner[i, 1] = x_opt[3*i+1]
                inner[i, 2] = x_opt[3*i+2]
            outer = np.array([0, 0, 0])
            side = compute_min_enclosing_side(x_opt[0:33].reshape(11, 2), x_opt[2:34:3])
            
            if check_validity(inner, outer, side):
                score = 1.0 / side
                if score > best_score:
                    best_score = score
                    best_result = (inner, outer, side)
    except Exception as e:
        pass

    # If SLSQP failed or didn't find valid, try a few random restarts
    if time.time() < deadline - 60:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb initial guess
            x0_pert = x0 + np.random.normal(0, 0.5, 33)
            x0_pert[2:34:3] = np.random.uniform(-30, 30, 11)
            try:
                res = minimize(objective, x0_pert, method='SLSQP', 
                              bounds=[(-10, 10)]*22 + [(-180, 180)]*11, 
                              constraints=cons, 
                              options={'maxiter': 500})
                if res.success:
                    x_opt = res.x
                    inner = np.zeros((11, 3))
                    for i in range(11):
                        inner[i, 0] = x_opt[3*i]
                        inner[i, 1] = x_opt[3*i+1]
                        inner[i, 2] = x_opt[3*i+2]
                    outer = np.array([0, 0, 0])
                    side = compute_min_enclosing_side(x_opt[0:33].reshape(11, 2), x_opt[2:34:3])
                    if check_validity(inner, outer, side):
                        score = 1.0 / side
                        if score > best_score:
                            best_score = score
                            best_result = (inner, outer, side)
            except:
                pass

    if best_result is None:
        # Fallback to seed program result
        inner_hex_data = np.array(
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result
