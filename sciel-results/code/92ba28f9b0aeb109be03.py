import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(positions, angles):
        """
        Check if inner hexagons are disjoint using Shapely or SAT.
        Since Shapely might be slow, we use a simple geometric check for convex polygons.
        For robustness in final check, we can use the support function or a simple bounding box pre-check.
        However, for speed in the loop, we rely on the LP constraints.
        For the final verification, we use a strict geometric test.
        """
        polys = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            polys.append(verts)
        
        # Using a simple convex polygon intersection test
        # For two convex polygons, they intersect if one vertex of one is inside the other,
        # or if their edges intersect.
        # A faster check: Separating Axis Theorem (SAT).
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                if _sat_intersects(p1, p2):
                    return False
        return True

    def _sat_intersects(p1, p2):
        """
        Check if two convex polygons intersect using SAT.
        Returns True if they intersect.
        """
        for verts in [p1, p2]:
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge = v2 - v1
                n = np.array([-edge[1], edge[0]])
                n = n / (np.linalg.norm(n) + 1e-9)
                
                proj1 = p1 @ n
                proj2 = p2 @ n
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    return False # Found a separating axis
        return True # No separating axis found -> they intersect

    # LP Formulation
    # Variables: x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}
    # Objective: Minimize R (outer hexagon circumradius, proportional to side)
    # Actually, minimizing the side length is equivalent to minimizing the circumradius R.
    # Side = R * sqrt(3). So minimizing R is minimizing Side.
    # Wait, the objective function in LP should be linear.
    # The condition "hexagon i is inside outer hexagon" is:
    # For each normal direction n_k (k=0..5) of the outer hexagon:
    #   max_{v in vertices of hex i} (v . n_k) <= R
    #   This is equivalent to: (center_i + max_vertex_offset_i . n_k) <= R
    #   where max_vertex_offset_i is the max projection of the unit hexagon (at origin) onto n_k, given its rotation.
    
    # Since the LP assumes fixed rotations (we will fix them to 0 or try a few),
    # the "max_vertex_offset" is constant for each hexagon and normal.
    
    # Constraint: Hex i and Hex j do not overlap.
    # For convex polygons, they don't overlap if there exists a separating axis.
    # This is non-linear.
    # However, for hexagons on a lattice, we can enforce separation along specific axes.
    # A common approach for LP packing: enforce separation along the normals of the faces of the *inner* hexagons.
    # If hex i and hex j are separated along any of the 6 face normals of hex i (or j), they don't overlap.
    # We can add constraints:
    # For each pair (i,j) and each normal n (from a set of candidate normals):
    #   (center_j . n) - (center_i . n) >= D(i,j,n)
    #   OR
    #   (center_i . n) - (center_j . n) >= D(i,j,n)
    #   where D(i,j,n) is the minimum distance required along n for non-overlap.
    #   This is a disjunctive constraint, which LP cannot handle directly.
    #   
    # Alternative: Use the fact that for regular hexagons of same size, if they are on a lattice,
    # we can enforce separation by bounding box or specific directional constraints.
    # 
    # Let's try a simpler LP relaxation:
    # Minimize R
    # Subject to:
    # 1. Containment:
    #    For each hex i and each outer normal n_k:
    #    center_i . n_k + r_in(i, n_k) <= R
    #    -center_i . n_k + r_in(i, n_k) <= R  (since we can translate the outer hex center, but we fix it to 0)
    #    Actually, if outer hex is centered at 0, the constraint is |center_i . n_k| + r_in(i, n_k) <= R?
    #    No. The condition is that the entire hex i is inside.
    #    The support function of the outer hex (centered at 0) in direction n is R.
    #    The support function of inner hex i in direction n is center_i . n + h_i(n).
    #    So we need: center_i . n + h_i(n) <= R for all n in {outer normals}.
    #    AND: -center_i . n + h_i(-n) <= R ?
    #    Actually, the outer hexagon is symmetric. The 6 normals are opposite pairs.
    #    So if we check n, we effectively check -n as well if we use absolute value?
    #    No, the LP constraint is linear.
    #    center_i . n_k + h_i(n_k) <= R
    #    -center_i . n_k + h_i(-n_k) <= R  (which is the same as center_i . (-n_k) + h_i(-n_k) <= R)
    #    Since n_k and -n_k are both in the set of 6 normals, we just need 6 constraints per hex.
    
    # 2. Separation:
    #    This is the hard part. 
    #    We can use a "soft" constraint or a specific lattice structure.
    #    Given the time limit, let's try a deterministic construction on a honeycomb lattice,
    #    then use the LP to *shrink* the outer hexagon by allowing the inner hexes to move slightly?
    #    No, moving them might break separation.
    #    
    #    Let's stick to the deterministic honeycomb packing, which is known to be optimal or near-optimal for small N.
    #    For N=12, the honeycomb packing (4-4-4 or similar) is very tight.
    #    
    #    Let's calculate the side length of the honeycomb packing.
    #    
    #    Strategy:
    #    1. Generate honeycomb positions.
    #    2. Calculate the minimal outer hexagon side.
    #    3. Try to improve by rotating the outer hexagon (0 to 30 deg).
    #    4. If time permits, try local optimization of positions (fixed rotations) to tighten packing.
    
    def generate_honeycomb():
        coords = []
        # 4x3 grid
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 + (r % 2) * (sqrt3 / 2)
                y = r * 1.5
                coords.append([x, y])
        
        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])
        
        positions = np.array(coords) - np.array([cx, cy])
        angles = np.zeros(N)
        return positions, angles

    def try_local_optimize(positions, angles, max_time=20.0):
        """
        Attempt to improve the packing by local optimization.
        We minimize the outer side length, subject to separation constraints.
        We use SLSQP with gradient-free approximations or simple finite differences.
        """
        end_time = time.time() + max_time
        if time.time() > end_time:
            return positions, angles
        
        # Objective: calc_min_outer_side
        # Constraints: separation distance > 0
        
        # To speed up, we can project the problem to a lower dimensional space?
        # Or just use Nelder-Mead on the positions (fixing angles to 0).
        
        from scipy.optimize import minimize
        
        def objective(params):
            pos = params.reshape(N, 2)
            return calc_min_outer_side(pos, angles, 0.0)
        
        def constraints_func(params):
            pos = params.reshape(N, 2)
            cons = []
            for i in range(N):
                for j in range(i+1, N):
                    # Approximate separation check
                    # Use the support function in the direction connecting centers
                    d = pos[j] - pos[i]
                    dist = np.linalg.norm(d)
                    if dist < 1e-6:
                        return [0.0] * (N*(N-1)//2)
                    n = d / dist
                    # Check if they overlap along this axis? 
                    # Better: use the SAT distance calculated earlier.
                    # But SAT is expensive.
                    # For a quick check, we can use the fact that if the distance between centers
                    # is less than the sum of their "widths" in that direction, they might overlap.
                    # Let's just use the actual SAT distance but limit the number of checks.
                    vi = get_hex_verts(pos[i][0], pos[i][1], 1.0, 0.0)
                    vj = get_hex_verts(pos[j][0], pos[j][1], 1.0, 0.0)
                    dist_sat = _sat_distance(vi, vj)
                    cons.append(dist_sat)
            return cons
        
        def _sat_distance(p1, p2):
            min_dist = float('inf')
            for verts in [p1, p2]:
                for i in range(len(verts)):
                    v1 = verts[i]
                    v2 = verts[(i + 1) % len(verts)]
                    edge = v2 - v1
                    n = np.array([-edge[1], edge[0]])
                    n = n / (np.linalg.norm(n) + 1e-9)
                    proj1 = p1 @ n
                    proj2 = p2 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2:
                        dist = min2 - max1
                        if dist < min_dist:
                            min_dist = dist
                    elif max2 < min1:
                        dist = min1 - max2
                        if dist < min_dist:
                            min_dist = dist
            if min_dist == float('inf'):
                return 0.0
            return min_dist

        x0 = positions.flatten()
        bounds = [(-5, 5)] * (2*N)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraints_func}],
                options={'maxiter': 50, 'ftol': 1e-5}
            )
            if res.fun < calc_min_outer_side(positions, angles, 0.0):
                # Verify validity
                new_pos = res.x.reshape(N, 2)
                if check_disjoint(new_pos, angles):
                    return new_pos, angles
        except:
            pass
        
        return positions, angles

    # 1. Generate initial honeycomb
    pos_init, ang_init = generate_honeycomb()
    best_pos = pos_init
    best_ang = ang_init
    best_side = calc_min_outer_side(best_pos, best_ang, 0.0)
    
    # 2. Try different outer rotations
    for outer_ang in range(0, 31, 5):
        side = calc_min_outer_side(best_pos, best_ang, float(outer_ang))
        if side < best_side:
            best_side = side
            best_outer_ang = outer_ang
        else:
            best_outer_ang = 0
    
    # 3. Try local optimization
    if time.time() < deadline - 20:
        opt_pos, opt_ang = try_local_optimize(best_pos, best_ang, max_time=15.0)
        if check_disjoint(opt_pos, opt_ang):
            # Find best outer rotation for the optimized position
            for outer_ang in range(0, 31, 5):
                side = calc_min_outer_side(opt_pos, opt_ang, float(outer_ang))
                if side < best_side:
                    best_side = side
                    best_pos = opt_pos
                    best_ang = opt_ang
                    best_outer_ang = outer_ang
    
    # 4. Try a few random perturbations + local optimization
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        pert_pos = best_pos + np.random.normal(0, 0.05, best_pos.shape)
        pert_ang = best_ang
        side = calc_min_outer_side(pert_pos, pert_ang, 0.0)
        if side < best_side:
            if check_disjoint(pert_pos, pert_ang):
                best_pos = pert_pos
                best_ang = pert_ang
                best_side = side

    # Final validation
    if not check_disjoint(best_pos, best_ang):
        # Fallback
        pos_init, ang_init = generate_honeycomb()
        best_pos = pos_init
        best_ang = ang_init
        best_side = calc_min_outer_side(best_pos, best_ang, 0.0)
        best_outer_ang = 0

    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], best_ang[i]])
    
    return inner, (0.0, 0.0), best_side, float(best_outer_ang)

# EVOLVE-BLOCK-END
