import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute unit hexagon vertices for angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # This corresponds to a "pointy-top" orientation relative to x-axis if angle 0 is vertex.
    # Let's use "flat-top" or "pointy-top"? 
    # Standard definition: vertices at k * 60 degrees.
    # If angle=0, vertex at (1,0).
    def get_hex_vertices_unit(angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = math.cos(theta)
            y = math.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    # Helper: Calculate the minimum distance between two convex polygons using SAT.
    # Since we are dealing with regular hexagons of side 1, we can precompute normals.
    # However, the relative angle matters.
    # For simplicity and robustness in SLSQP, we will restrict inner hexagons to angle 0 or 30.
    # Outer hexagon angle 0.
    
    # We will optimize positions (x, y) for 12 hexagons.
    # We will fix the angles of inner hexagons to 0 degrees for the main optimization,
    # but we can try a few fixed angle configurations (all 0, or mixed) in a loop.
    
    # SAT distance calculation for two unit hexagons with relative angle delta.
    # If both are angle 0, relative angle is 0.
    # The support function of a regular hexagon with side 1 (circumradius 1) in direction u is:
    # h(u) = max_i (v_i . u)
    # For a hexagon centered at c, the support in direction u is c . u + h(u).
    # The distance between two convex sets A and B in direction u is (h_A(u) - h_B(-u)).
    # The minimum distance is min over all potential separating axes u.
    # For regular hexagons, the critical axes are the normals to the edges.
    
    # Let's define the normals for a unit hexagon at angle 0.
    # Edges are between vertices.
    # Vertex 0: (1, 0), Vertex 1: (0.5, sqrt(3)/2)
    # Edge 0 vector: V1 - V0 = (-0.5, sqrt(3)/2)
    # Normal to Edge 0 (outward): (sqrt(3)/2, 0.5) -> Angle 30 deg.
    # The 6 outward normals for angle 0 hexagon are at 30, 90, 150, 210, 270, 330 degrees.
    
    normals_0 = np.array([
        [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))] for k in range(6)
    ])
    
    # For angle 30 hexagon, normals are rotated by 30 deg.
    normals_30 = np.array([
        [math.cos(math.radians(60 + 60*k)), math.sin(math.radians(60 + 60*k))] for k in range(6)
    ])
    
    # Support function value for a unit hexagon (centered at origin) in direction n:
    # For angle 0, max dot product with normals_0 is the distance from center to edge, which is sqrt(3)/2.
    # Actually, h(n) = max(v_i . n).
    # For n = (cos 30, sin 30), v_0=(1,0) gives cos 30 = sqrt(3)/2. v_1=(0.5, sqrt(3)/2) gives 0.5*sqrt(3)/2 + 3/4 = sqrt(3)/4 + 3/4? No.
    # v_1 . n = 0.5 * sqrt(3)/2 + sqrt(3)/2 * 1/2 = sqrt(3)/4 + sqrt(3)/4 = sqrt(3)/2.
    # So h(n) = sqrt(3)/2 for all 6 normals.
    # For any direction, the support function is piecewise linear.
    
    # To check separation, we check the 12 normals (6 from hex 1, 6 from hex 2).
    # If for any normal n (from hex 1) or -n (from hex 2), the projection intervals overlap, they intersect.
    # Specifically, if max(v1 . n) + c1 . n < min(v2 . n) + c2 . n, they are separated by n.
    # min(v2 . n) = -max(v2 . -n).
    # So condition: c1 . n + h1(n) < c2 . n - h2(-n)  =>  (c1 - c2) . n < -h1(n) - h2(-n)
    # Wait, standard SAT:
    # Project both onto n.
    # Interval 1: [c1 . n - h1(-n), c1 . n + h1(n)]
    # Interval 2: [c2 . n - h2(-n), c2 . n + h2(n)]
    # Disjoint if: c1 . n + h1(n) < c2 . n - h2(-n)  OR  c2 . n + h2(n) < c1 . n - h1(-n)
    # Since h(n) = h(-n) for regular hexagons (symmetric), let h0 = sqrt(3)/2.
    # Disjoint if: (c1 - c2) . n < -2*h0  OR  (c2 - c1) . n < -2*h0
    # i.e., |(c1 - c2) . n| >= 2*h0 = sqrt(3)
    
    # This holds IF n is a normal to an edge of EITHER hexagon.
    # If both hexagons have the same orientation (angle 0), the normals are the same set.
    # We check the 6 normals. If for ALL 6 normals, the projection distance < sqrt(3), they overlap.
    # To ensure they do NOT overlap, there must exist AT LEAST ONE normal n such that |(c1-c2).n| >= sqrt(3).
    # This is a non-convex constraint (OR of linear inequalities).
    
    # However, if we assume a specific "contact" axis, we can linearize.
    # Or, we can use the fact that for packing, they will likely touch.
    # A simpler approach for SLSQP:
    # Define constraint g_ij = min_k ( (c_i - c_j) . n_k )^2 - 3  <= 0 ? No, that's for overlap.
    # We want NO overlap.
    # The condition for NO overlap is: Exists k such that ((c_i - c_j) . n_k)^2 >= 3.
    # This is hard for SLSQP.
    
    # Alternative: Use the "distance" function which is convex for convex sets?
    # The distance between two convex sets is a convex function of their relative position?
    # No, the distance is convex, but the constraint "distance > 0" is non-convex (outside a convex set).
    
    # Let's try a different strategy:
    # Fix the relative orientation to 0 for all inner hexagons.
    # Use a grid of potential "separating axes" and enforce that for each pair,
    # the minimum over axes of the "gap" is >= 0.
    # Gap_k = (c_i - c_j) . n_k - sqrt(3)
    # We need Max_k ( Gap_k ) >= 0.
    # This is still non-smooth.
    
    # Let's go back to the inspiration: "explicit non-overlap constraints derived from SAT".
    # Perhaps they linearize by assuming the contact axis?
    # Or they use a penalty method with SLSQP?
    
    # Let's try a robust approach:
    # 1. Initialize with a valid honeycomb packing.
    # 2. Use SLSQP to minimize the outer side length R.
    # 3. Constraints:
    #    - Inner hexagons inside outer hexagon (Linear in x,y,R).
    #    - Inner hexagons disjoint.
    
    # For disjointness, we can use a smooth approximation of the collision indicator,
    # or we can rely on the fact that starting from a valid solution, SLSQP might stay valid
    # if we use a penalty for violations in the objective.
    
    # Objective: R + lambda * sum(violations)
    # Violation for pair i,j: max(0, sqrt(3) - max_k |(c_i-c_j).n_k|)
    # This is non-smooth.
    
    # Let's try the "Sequential QP" idea from the prompt?
    # "Formulate the packing as a sequence of 12 convex quadratic programming (QP) problems"
    # This is complex to implement correctly in a short time.
    
    # Let's stick to SLSQP with a smooth penalty for overlap.
    # The function f(d) = max(0, d_max - d)^2 where d is the minimum distance.
    # For hexagons, the minimum distance can be calculated via SAT.
    
    def calc_min_distance(c1, c2, angle1, angle2):
        # c1, c2: centers
        # Returns the minimum distance between the two hexagons (0 if overlapping)
        # Using SAT
        # Get normals for both
        # If angle1 == 0, normals = normals_0
        # If angle1 == 30, normals = normals_30
        
        n1 = normals_0 if angle1 < 15 else normals_30
        n2 = normals_0 if angle2 < 15 else normals_30
        
        # All candidate normals
        all_normals = np.vstack([n1, n2])
        
        min_dist = np.inf
        diff = c1 - c2
        
        for n in all_normals:
            # Project centers
            proj1 = np.dot(c1, n)
            proj2 = np.dot(c2, n)
            
            # Support radii
            # For unit hexagon, support radius in direction n is sqrt(3)/2
            # This is true for ANY direction? No, only for normals to edges.
            # For a regular polygon, the support function h(u) is the distance from center to the supporting line perpendicular to u.
            # For a regular hexagon with circumradius 1, the inradius is sqrt(3)/2.
            # h(u) = inradius * (1 / cos(angle(u, nearest_normal))) ?
            # Actually, for a regular hexagon, h(u) varies between sqrt(3)/2 and 1.
            # h(u) = max(v_i . u).
            # We must calculate h(u) exactly.
            
            def support_hex(angle_deg, u):
                verts = get_hex_vertices_unit(angle_deg)
                return np.max(verts @ u)
            
            h1 = support_hex(angle1, n)
            h2 = support_hex(angle2, -n) # h(-n) = max(v . -n) = -min(v . n)
            
            # Distance in this direction
            dist = abs(proj1 - proj2) - h1 - h2
            
            if dist < min_dist:
                min_dist = dist
                
        return min_dist

    # This is expensive to call in SLSQP for every iteration.
    # 12 hexagons -> 66 pairs. 12 normals per pair.
    # 66 * 12 = 792 dot products per evaluation.
    # SLSQP might do 100-1000 evaluations. 792k ops. Should be fast enough in Python/NumPy.
    
    def objective(params):
        # params: 24 positions (x,y for 12 hexes) + 1 R (outer side)
        # Angles are fixed in the closure
        
        R = params[-1]
        if R < 2.0: # Too small
            return 1e10
            
        positions = params[:-1].reshape(N, 2)
        
        # Penalty for containment
        # Inner hex vertices must be inside outer hex.
        # Outer hex center (0,0), side R, angle 0.
        # Inequalities: n_i . p <= R * sqrt(3) / 2
        # Normals for outer hex (angle 0): normals_0
        max_violation = 0.0
        for i in range(N):
            c = positions[i]
            angle = angles[i]
            verts = c + get_hex_vertices_unit(angle)
            for v in verts:
                for n in normals_0:
                    val = np.dot(v, n)
                    limit = R * math.sqrt(3) / 2
                    if val > limit:
                        viol = val - limit
                        if viol > max_violation:
                            max_violation = viol
        
        # Penalty for overlap
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c1 = positions[i]
                c2 = positions[j]
                a1 = angles[i]
                a2 = angles[j]
                
                d = calc_min_distance(c1, c2, a1, a2)
                if d < 0:
                    # Penalty proportional to depth of penetration
                    overlap_penalty += (-d)**2
                    
        return R + 100.0 * max_violation**2 + 100.0 * overlap_penalty

    # We will try a few fixed angle configurations
    # Config 1: All 0
    # Config 2: Alternating 0, 30
    # Config 3: Random
    
    best_score = 1e10
    best_params = None
    
    np.random.seed(42)
    
    # Initial guess: Honeycomb
    sqrt3 = math.sqrt(3)
    # Generate honeycomb positions
    # Radius 1 hexagons.
    # Distance between centers in honeycomb is 2 * inradius = sqrt(3)
    # No, distance between centers of touching hexagons is 2 * inradius = sqrt(3)?
    # Side length 1. Inradius sqrt(3)/2.
    # Distance between centers = 2 * sqrt(3)/2 = sqrt(3).
    
    # Let's place them in a grid.
    # x step: 1.5 (horizontal distance between columns in pointy-top?)
    # Let's just use a circular distribution for initialization to be safe.
    init_pos = []
    for i in range(N):
        theta = i * 2 * math.pi / N
        r = 2.0
        init_pos.append([r * math.cos(theta), r * math.sin(theta)])
    
    init_params = []
    for p in init_pos:
        init_params.append(p[0])
        init_params.append(p[1])
    init_params.append(3.0) # Initial R
    
    # Bounds for positions: [-5, 5]
    # Bounds for R: [2.0, 10.0]
    bounds = [(-5, 5)] * (2*N) + [(2.0, 10.0)]
    
    angle_configs = [
        [0]*N,
        [0, 30] * (N//2),
        [30]*N
    ]
    
    # Add some random configs
    for _ in range(5):
        angle_configs.append(np.random.choice([0, 30], N).tolist())
        
    for trial, angles in enumerate(angle_configs):
        if time.time() > deadline - 20:
            break
            
        # Perturb initial positions
        x0 = init_params.copy()
        for i in range(2*N):
            x0[i] += np.random.normal(0, 0.5)
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_angles = angles
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback to initial guess
        best_params = init_params
        best_angles = [0]*N
        best_score = objective(init_params)
        
    # Extract results
    positions = best_params[:-1].reshape(N, 2)
    R = best_params[-1]
    
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], best_angles[i]])
        
    return inner, (0.0, 0.0), R, 0.0

# EVOLVE-BLOCK-END
