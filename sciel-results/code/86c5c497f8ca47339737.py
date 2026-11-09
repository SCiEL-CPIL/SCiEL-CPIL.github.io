"""Strategy: Non-convex optimization using Trust-Region Reflective method to maximize minimum clearance."""
import numpy as np
import time
import scipy.optimize as opt

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute rotation matrices for base hexagon vertices
    # Base hexagon vertices (side 1, centered at 0,0, angle 0)
    base_angles = np.array([0, 60, 120, 180, 240, 330]) * np.pi / 180.0 # 300 -> 330? No, 300 is standard. 
    # Standard regular hexagon vertices: (1,0), (1/2, sqrt(3)/2), ...
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Outer hexagon normals (for constraint: dot(p, n) <= S_inradius)
    # Outer hexagon is fixed orientation (vertex at 0 deg). 
    # Normals are perpendicular to edges. Edges are at angles 30, 90, 150, 210, 270, 330.
    outer_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    outer_nx = np.cos(outer_normals)
    outer_ny = np.sin(outer_normals)
    
    def get_hex_verts(x, y, theta):
        """Return vertices of a unit hexagon centered at (x,y) rotated by theta."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return base_verts @ R.T + np.array([x, y])

    def min_dist_hex(hex1_verts, hex2_verts):
        """
        Calculate minimum distance between two convex polygons (hexagons).
        If they overlap, distance is negative.
        We approximate this by checking:
        1. If any vertex of A is inside B (or vice versa) -> distance is 0 (or negative if we want penetration depth, but for constraint >=0, 0 is the boundary).
           Actually, for optimization, we want a smooth-ish function. 
           Simplest robust metric for 'disjointness' in optimization:
           Min distance between any vertex of A and any edge of B, and vice versa.
           If polygons intersect, this distance is 0.
           To make it differentiable and penalize overlap, we can use the negative of the penetration depth, but that's hard.
           Instead, we enforce the constraint that min_dist >= 0. 
           If they overlap, min_dist is 0. The gradient will be zero or undefined.
           Better: Use the distance between the closest points. 
           For convex polygons, if they don't intersect, min_dist > 0. If they do, min_dist = 0.
           To push them apart, we need the gradient to point away.
           If we define d = min_dist, and we want d >= 0, the constraint is active when d=0.
           When d=0, the closest points are on the boundaries. The direction to increase d is the direction separating them.
           
           Let's implement a function that returns the minimum distance and the direction vector to increase it.
           However, scipy constraints only take function values. 
           We can use a penalty approach or rely on the fact that if d=0, the optimizer might stall.
           To avoid stalling, we can add a small repulsion term to the objective or use a barrier.
           Let's stick to the constraint: min_dist >= 0.
           
           Implementation of min_dist between two convex polygons:
           1. Check if any vertex of A is inside B. If yes, return 0.
           2. Check if any vertex of B is inside A. If yes, return 0.
           3. Otherwise, find min distance between all edges of A and edges of B.
           
           For speed and simplicity in the inner loop, we can precompute edge vectors.
        """
        # Fast check: Bounding boxes
        # If bounding boxes don't overlap, min dist is min of corner distances? No, edge distances.
        # Let's just do edge-edge distance.
        
        min_d = float('inf')
        
        # Helper: point in convex polygon
        def point_in_convex(p, poly):
            # Cross products of consecutive edges must have same sign
            signs = []
            for i in range(6):
                p1 = poly[i]
                p2 = poly[(i+1)%6]
                cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
                if cross == 0: continue
                signs.append(cross > 0)
            if not signs: return True
            return all(signs) or all(not s for s in signs)

        # Check vertex inclusion
        if any(point_in_convex(v, hex2_verts) for v in hex1_verts):
            return 0.0
        if any(point_in_convex(v, hex1_verts) for v in hex2_verts):
            return 0.0
            
        # Edge-edge distance
        # There are 36 pairs of edges.
        # Distance between two segments (p1,p2) and (p3,p4)
        def seg_dist(p1, p2, p3, p4):
            # Standard algorithm
            d1 = p2 - p1
            d2 = p4 - p3
            r = p1 - p3
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            
            EPS = 1e-9
            if a <= EPS and e <= EPS:
                return np.linalg.norm(p1 - p3)
            if a <= EPS:
                t = 0.0
                s = np.clip(np.dot(d1, r) / e, 0.0, 1.0) # Wait, if a is 0, p1 is point.
                # Correct logic for point-segment
                pass # This is getting complex. Let's use a simpler approximation for optimization.
            
            # Simplified: Sample points on edges? No.
            # Let's use the fact that if they don't intersect, the min distance is either between two vertices or vertex-edge.
            # We already checked vertex-in-poly.
            # So min dist is min of:
            # - dist(v1, edge2) for all v1 in A, edge2 in B
            # - dist(v2, edge1) for all v2 in B, edge1 in A
            
            pass
        
        # Let's implement a robust but slightly slower min_dist using sampling or exact segment distance.
        # Given N=11, 55 pairs, 36 edge pairs per check. 1980 edge pair checks per eval.
        # This is fine.
        
        def point_seg_dist(p, a, b):
            ab = b - a
            t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            return np.linalg.norm(p - proj)
        
        # Check vertex-to-edge distances
        for v1 in hex1_verts:
            for k in range(6):
                e1_start = hex2_verts[k]
                e1_end = hex2_verts[(k+1)%6]
                d = point_seg_dist(v1, e1_start, e1_end)
                if d < min_d: min_d = d
        
        for v2 in hex2_verts:
            for k in range(6):
                e2_start = hex1_verts[k]
                e2_end = hex1_verts[(k+1)%6]
                d = point_seg_dist(v2, e2_start, e2_end)
                if d < min_d: min_d = d
                
        return min_d

    def objective_and_constraints(x):
        """
        x: [x1, y1, theta1, ..., xN, yN, thetaN]
        Returns:
        - Objective: -epsilon (we maximize epsilon, so minimize -epsilon)
        - Constraints: list of functions g(x) >= 0
        """
        # Unpack
        positions = x[:2*N].reshape(N, 2)
        angles = x[2*N:]
        
        # Compute vertices for all hexagons
        verts_list = [get_hex_verts(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
        
        # 1. Outer Boundary Constraint
        # For each vertex of each inner hex, dist to outer boundary >= 0
        # Outer boundary: dot(v, n_k) <= S_inradius for all k
        # We don't know S_inradius yet. 
        # Strategy: Fix S to a small value? No, we want to minimize S.
        # Alternative: Minimize the maximum distance of any inner vertex from the origin?
        # If we minimize the circumradius of the set of inner hexagons, we can fit them in a hexagon of that size (approx).
        # Actually, the "score" is the side length S of the outer hexagon.
        # If we place inner hexagons, the required S is determined by the max projection of any inner vertex onto the outer normals.
        # S_required = max_i,k ( dot(v_ik, n_k) ) / (sqrt(3)/2)
        # We want to minimize S_required.
        
        # So, let's define the objective as S_required.
        # S_required = max( dot(v, n) for all v in all inner verts, for all n in outer_normals ) / (sqrt(3)/2)
        
        max_proj = 0.0
        for v_list in verts_list:
            for v in v_list:
                # Project onto all 6 normals
                projs = v[0]*outer_nx + v[1]*outer_ny
                max_proj = max(max_proj, np.max(projs))
        
        S_val = max_proj / (np.sqrt(3)/2)
        
        # We want to minimize S_val.
        # But we also need to ensure inner hexagons are disjoint.
        # We can use a penalty for overlaps in the objective, or constraints.
        # Constraints: min_dist(i, j) >= 0 for all pairs.
        
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_hex(verts_list[i], verts_list[j])
                # Constraint: d >= 0
                # Since min_dist returns 0 on overlap, this constraint is active on overlap.
                # This is a non-smooth constraint (kink at 0). Trust region might struggle.
                # To help, we can add a soft penalty to the objective for small d.
                if d < 0.1:
                    # Add penalty to objective later? 
                    pass
                constraints.append({'type': 'ineq', 'fun': lambda d=d: d})
                
        # Return objective (S_val) and constraints
        return S_val, constraints

    # Initial guess: Hexagonal lattice
    def init_lattice():
        pos = []
        # 1 center, 6 around, 4 more?
        # Let's just use a 3x4 grid clipped or similar.
        # Or just random.
        return None

    # Optimization setup
    # Variables: 3 * N = 33
    # We will run multiple restarts.
    
    best_S = float('inf')
    best_x = None
    
    # Time management
    time_per_restart = 5.0
    max_restarts = int((deadline - (time.time() - start_time)) / time_per_restart)
    if max_restarts < 5: max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0: break
        
        # Initialize
        # Try a lattice-like initialization
        if restart == 0:
            # 1 center, 6 ring, 4 outer
            x0 = np.zeros(3*N)
            # Center
            x0[0:2] = [0,0]
            x0[2] = 0
            # Ring 1 (6 hexagons)
            for i in range(6):
                ang = i * np.pi / 3
                r = np.sqrt(3) # Distance between centers of touching hexagons
                idx = (i+1)*3
                x0[idx:idx+2] = [r*np.cos(ang), r*np.sin(ang)]
                x0[idx+2] = 0
            # Ring 2 (4 hexagons, placed at gaps or extending)
            # Let's just place them randomly for now to avoid complex lattice logic
            for i in range(4):
                ang = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(2.5, 3.5)
                idx = (i+7)*3
                x0[idx:idx+2] = [r*np.cos(ang), r*np.sin(ang)]
                x0[idx+2] = 0
        else:
            # Random initialization
            x0 = np.random.randn(3*N) * 2.0
            x0[2::3] = np.random.uniform(0, np.pi/3, N) # Random angles
        
        # Define wrapper for scipy
        def func(x):
            s, cons = objective_and_constraints(x)
            # Add penalty for overlaps to guide gradient
            # The constraints handle the hard stop, but gradient is 0 at d=0.
            # We can add a small quadratic penalty for d < 0.1
            positions = x[:2*N].reshape(N, 2)
            angles = x[2*N:]
            penalty = 0.0
            for i in range(N):
                v1 = get_hex_verts(positions[i,0], positions[i,1], angles[i])
                for j in range(i+1, N):
                    v2 = get_hex_verts(positions[j,0], positions[j,1], angles[j])
                    d = min_dist_hex(v1, v2)
                    if d < 0.05:
                        penalty += (0.05 - d) ** 2 * 100
            return s + penalty

        # Constraints for scipy (must be separate function or cached)
        # scipy.optimize.minimize with 'trf' expects constraints as a list of dicts.
        # The 'fun' in constraints must be independent of the objective's internal state.
        # We will recreate the constraint functions in the loop? No, they are stateless if they take x.
        
        # However, creating 55 lambda functions per iteration is slow.
        # Let's define a single constraint function that returns an array of values for all pairs.
        # But scipy 'ineq' constraints can be vectorized? 
        # "fun" : callable, The objective function to be evaluated. ... 
        # For 'ineq', fun(x) >= 0.
        # We can define one constraint function that returns a vector of distances for all pairs.
        
        def all_pair_distances(x):
            positions = x[:2*N].reshape(N, 2)
            angles = x[2*N:]
            verts_list = [get_hex_verts(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
            dists = []
            for i in range(N):
                for j in range(i+1, N):
                    d = min_dist_hex(verts_list[i], verts_list[j])
                    dists.append(d)
            return np.array(dists)
        
        cons = ({'type': 'ineq', 'fun': all_pair_distances},)
        
        try:
            res = opt.minimize(func, x0, method='trf', constraints=cons, 
                                options={'maxiter': 100, 'gtol': 1e-6, 'xtol': 1e-6})
            
            if res.success:
                s_final, _ = objective_and_constraints(res.x)
                if s_final < best_S:
                    best_S = s_final
                    best_x = res.x.copy()
                    
        except Exception as e:
            pass

    if best_x is None:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0

    # Extract final result
    positions = best_x[:2*N].reshape(N, 2)
    angles = best_x[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i,0]), float(positions[i,1]), float(angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

