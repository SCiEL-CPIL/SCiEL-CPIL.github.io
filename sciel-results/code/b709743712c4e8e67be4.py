import numpy as np
import time
from scipy.optimize import minimize

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

    def get_hex_vertices(x, y):
        # Returns (6, 2) array of vertices for a unit hexagon at (x,y) with angle 0
        angles = np.radians([0, 60, 120, 180, 240, 300])
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
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1])
            # v is unit hex. Check lengths
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False

        # Check non-overlap
        for i in range(11):
            vi = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1])
            for j in range(i+1, 11):
                vj = get_hex_vertices(inner_shifted[j, 0], inner_shifted[j, 1])
                separated = False
                for n in inner_normals:
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
                vi = get_hex_vertices(inner_shifted[i_h, 0], inner_shifted[i_h, 1])
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -epsilon:
                        return False
        return True

    def objective_and_constraints(x):
        # x: [x0, y0, ..., x10, y10, S]
        S = x[-1]
        coords = x[:-1].reshape(11, 2)
        
        # Objective: Minimize S
        # We will handle constraints separately for SLSQP
        return S

    def constraints_func(x):
        S = x[-1]
        coords = x[:-1].reshape(11, 2)
        cons = []
        
        # 1. Containment Constraints
        # For each inner vertex, and each outer normal n:
        # dot(v, n) <= S * apothem
        # This is linear in v and S.
        # v depends on coords.
        # n is fixed.
        # dot(v, n) = dot(center, n) + dot(offset, n)
        # offset is fixed for unit hex.
        
        # Precompute max projection of unit hex on each normal
        # Hex vertices relative to center:
        hex_v = get_hex_vertices(0, 0)
        
        for k, n in enumerate(outer_normals):
            # Max projection of hex vertices on n
            proj = np.dot(hex_v, n)
            max_proj = np.max(proj)
            
            # Constraint: dot(center, n) + max_proj <= S * apothem_ratio
            # dot(center, n) - S * apothem_ratio - max_proj <= 0
            
            # We have 11 centers.
            # Each center (cx, cy) gives: cx*nx + cy*ny - S*apothem - max_proj <= 0
            for i in range(11):
                cx, cy = coords[i]
                val = cx * n[0] + cy * n[1] - S * apothem_ratio - max_proj
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k, v=val: v}) 
                # Note: This lambda captures val from the loop, which is wrong for SLSQP.
                # We need to return the function that computes it from x.
                
        # Let's restructure to return a list of functions that take x.
        
    # Better approach for SLSQP constraints:
    
    def make_containment_constraint(i, k):
        # i: inner hex index
        # k: normal index
        n = outer_normals[k]
        hex_v = get_hex_vertices(0, 0)
        max_proj = np.max(np.dot(hex_v, n))
        
        def constraint(x):
            S = x[-1]
            cx, cy = x[2*i], x[2*i+1]
            return S * apothem_ratio - (cx * n[0] + cy * n[1] + max_proj)
        return constraint

    def make_non_overlap_constraint(i, j):
        # Constraint: max_k Gap_ij_k >= epsilon
        # Gap_ij_k is linear in x, y.
        # Max of linear functions is convex.
        # We want Max(Gaps) - epsilon >= 0.
        
        def constraint(x):
            S = x[-1] # Not used in gap, but part of x
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            
            vi = get_hex_vertices(ci[0], ci[1])
            vj = get_hex_vertices(cj[0], cj[1])
            
            max_gap = -np.inf
            for n in inner_normals:
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
    # Containment
    for i in range(11):
        for k in range(6):
            cons.append({'type': 'ineq', 'fun': make_containment_constraint(i, k)})
            
    # Non-overlap
    for i in range(11):
        for j in range(i+1, 11):
            cons.append({'type': 'ineq', 'fun': make_non_overlap_constraint(i, j)})

    # Initial Guess
    x0 = np.zeros(23)
    x0[-1] = 4.0 # S
    # Hex 0 at center
    x0[0] = 0
    x0[1] = 0
    # Hex 1-6 in ring
    for i in range(1, 7):
        angle = np.radians(60 * (i-1))
        r = np.sqrt(3)
        x0[2*i] = r * np.cos(angle)
        x0[2*i+1] = r * np.sin(angle)
    # Hex 7-10 further out
    for i in range(7, 11):
        angle = np.radians(30 + 120 * (i-7)) # 30, 150, 270, 30? No.
        # Place them in the gaps of the ring
        # Ring angles: 0, 60, 120, 180, 240, 300.
        # Gaps at: 30, 90, 150, 210, 270, 330.
        # We have 4 more. Let's put them at 30, 150, 210, 330?
        angles = [30, 150, 210, 330]
        a = np.radians(angles[i-7])
        r = 2.5
        x0[2*i] = r * np.cos(a)
        x0[2*i+1] = r * np.sin(a)

    best_result = None
    best_score = -np.inf

    # Try SLSQP
    try:
        res = minimize(objective_and_constraints, x0, method='SLSQP', bounds=[(-10, 10)]*22 + [(2, 10)], constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            x_opt = res.x
            # Construct result
            inner = np.zeros((11, 3))
            for i in range(11):
                inner[i, 0] = x_opt[2*i]
                inner[i, 1] = x_opt[2*i+1]
                inner[i, 2] = 0
            outer = np.array([0, 0, 0])
            side = x_opt[-1]
            
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
            x0_pert = x0 + np.random.normal(0, 0.5, 23)
            x0_pert[-1] = 4.5
            try:
                res = minimize(objective_and_constraints, x0_pert, method='SLSQP', bounds=[(-10, 10)]*22 + [(2, 10)], constraints=cons, options={'maxiter': 500})
                if res.success:
                    x_opt = res.x
                    inner = np.zeros((11, 3))
                    for i in range(11):
                        inner[i, 0] = x_opt[2*i]
                        inner[i, 1] = x_opt[2*i+1]
                        inner[i, 2] = 0
                    outer = np.array([0, 0, 0])
                    side = x_opt[-1]
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

# EVOLVE-BLOCK-END
