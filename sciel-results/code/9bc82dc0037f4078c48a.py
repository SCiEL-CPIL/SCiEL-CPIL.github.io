import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define the 6 vertices of a unit regular hexagon centered at origin
    # Vertices are at angles 0, 60, ..., 300 degrees
    # Side length 1, circumradius 1
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * (math.pi / 3)
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Pre-compute rotation matrices for a few discrete orientations to reduce search space
    # However, the directive suggests continuous optimization. 
    # Let's allow continuous rotation for each hexagon? That's 33 variables.
    # Or fix rotation to 0 or 30 degrees? 
    # The current program allows the OUTER hexagon to rotate. 
    # The inner hexagons in the current program are all aligned with the lattice (rotation 0 relative to lattice).
    # Let's try allowing individual rotations for inner hexagons to find better fits.
    # Variables: x_i, y_i, theta_i for i=1..11. Total 33 variables.
    
    # To make it tractable, we can fix the rotation of inner hexagons to 0 or pi/6 (30 deg) initially,
    # or just optimize positions first with fixed rotation, then refine.
    # Let's start with optimizing positions only, assuming all inner hexagons are aligned (theta=0).
    # If that works, we can add rotation.
    # Actually, the outer hexagon rotation is a single variable. 
    # Let's fix outer rotation to 0 for simplicity in the objective function, 
    # and just optimize inner positions to fit in a fixed outer hexagon?
    # No, the objective is to minimize the outer side length.
    # So we need to find the minimal enclosing hexagon for the set of inner hexagons.
    
    # Let's define the objective function:
    # Given inner hexagon centers and rotations, find the minimal enclosing regular hexagon.
    # This is a convex optimization problem (minimize S such that all points are inside).
    # But the inner hexagons must not overlap. This is non-convex.
    
    # Strategy:
    # 1. Multi-start optimization of inner positions (and maybe rotations).
    # 2. For each set of inner configurations, compute the minimal enclosing hexagon side length S.
    # 3. Return the best S.
    
    # Distance between two convex polygons can be approximated.
    # For two regular hexagons, the minimal distance is the minimal distance between their edges/vertices.
    # A smooth approximation of the non-overlap constraint:
    # d_smooth(p1, p2, r1, r2) = soft_min over all vertex-edge pairs of (distance - 0)
    # This is complex.
    
    # Alternative: Use the fact that the Minkowski sum of two unit hexagons is a larger hexagon.
    # Two hexagons do not overlap iff the distance between their centers is >= the support function of the Minkowski sum in the direction of the center difference.
    # For regular hexagons, the Minkowski sum is a regular hexagon with side length 2 (if aligned) or similar.
    # Actually, the Minkowski sum of two unit regular hexagons is a regular hexagon with side length 2 (if same orientation).
    # If orientations differ, it's a 12-gon or similar.
    
    # Let's stick to aligned inner hexagons (theta=0) to simplify the constraint.
    # Constraint: Distance between centers >= 2 * APOTHEM? No.
    # Two aligned unit hexagons touch if distance between centers is 2 * APOTHEM?
    # No, they touch if the distance is 2 * circumradius? No.
    # If you place two unit hexagons side-by-side, the distance between centers is 2 * APOTHEM?
    # Let's check: Hexagon width (flat-to-flat) is 2 * APOTHEM.
    # If aligned, the minimal distance between centers for non-overlap is 2 * APOTHEM?
    # Yes, if they are aligned and touching flat sides.
    # But they could touch vertex-to-vertex? No, flat-to-flat is the closest approach for aligned hexagons.
    # Wait, if I rotate one by 30 degrees, the distance might be different.
    # For aligned hexagons, the minimal center distance is 2 * APOTHEM?
    # Let's verify. Hexagon vertices at (1,0), (0.5, sqrt(3)/2), ...
    # Center to flat side is APOTHEM.
    # So if two hexagons are aligned and touching, the distance between centers is 2 * APOTHEM.
    # So the constraint is ||p_i - p_j|| >= 2 * APOTHEM.
    # This is a simple Euclidean distance constraint!
    
    # So the problem reduces to:
    # Place 11 points in the plane such that pairwise distance >= 2 * APOTHEM.
    # Minimize the side length of the minimal enclosing regular hexagon.
    
    # This is much simpler. The "hexagon" shape of the inner objects only matters for the distance constraint (which is constant for aligned hexagons).
    # If we allow rotation, the distance constraint changes. But let's start with aligned.
    
    # Objective: Minimize S (side length of enclosing hexagon).
    # The enclosing hexagon is defined by its center (cx, cy), rotation beta, and side length S.
    # The apothem of the enclosing hexagon is A = S * APOTHEM.
    # The condition for a point p to be inside is:
    # For all 6 normals n_k(beta): |dot(p - c, n_k)| <= A.
    # We want to minimize S, which is equivalent to minimizing A.
    # For a fixed set of points, the minimal A for a given beta and c is max_k max_i |dot(p_i - c, n_k)|.
    # We can optimize c and beta to minimize this max value.
    # This is a convex problem for fixed points.
    
    # So the overall problem is:
    # Minimize over {p_i} of [ Min over (c, beta) of ( Max_{k,i} |dot(p_i - c, n_k(beta))| / APOTHEM ) ]
    # Subject to ||p_i - p_j|| >= 2 * APOTHEM for all i != j.
    
    # We can use a multi-start local optimization on the positions {p_i}.
    # For each local optimum of positions, we solve the inner convex problem for (c, beta, A).
    
    # To handle the non-convex distance constraint in the outer optimization, we can use a penalty method or just Nelder-Mead with a penalty.
    
    # Let's implement the inner solver first.
    
    def min_enclosing_hexagon(pts):
        """
        Find the minimal side length S and parameters (cx, cy, beta) of a regular hexagon enclosing pts.
        pts: (N, 2) array
        Returns: S, cx, cy, beta
        """
        # We can fix beta to 0 due to symmetry? No, the set of points might have a preferred orientation.
        # But the hexagon has 6-fold symmetry. We can search beta in [0, pi/6).
        
        # For a fixed beta, the problem of finding optimal (cx, cy, A) is:
        # Minimize A
        # Subject to:
        # -A <= dot(p_i - c, n_k) <= A for all i, k.
        # This is a linear program in variables cx, cy, A.
        
        def solve_lp_for_beta(beta):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            # Variables: cx, cy, A
            # Objective: Minimize A -> [0, 0, 1]
            
            # Constraints:
            # dot(p_i - c, n_k) <= A  =>  -nx*cx - ny*cy - A <= -dot(p_i, n_k)
            # dot(p_i - c, n_k) >= -A =>  nx*cx + ny*cy - A <= dot(p_i, n_k)
            
            A_ub = []
            b_ub = []
            
            for nx, ny in normals:
                for px, py in pts:
                    # -nx*cx - ny*cy - A <= -(px*nx + py*ny)
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    
                    # nx*cx + ny*cy - A <= px*nx + py*ny
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            from scipy.optimize import linprog
            res = linprog([0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            
            if res.status == 0:
                return res.x[2] # A
            else:
                return float('inf')
        
        # Search over beta
        best_A = float('inf')
        best_beta = 0.0
        
        # Coarse search
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            A = solve_lp_for_beta(beta)
            if A < best_A:
                best_A = A
                best_beta = beta
        
        # Fine search around best_beta
        if best_A < float('inf'):
            # Use bounded scalar minimization
            try:
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(lambda b: solve_lp_for_beta(b), bounds=(best_beta - 0.01, best_beta + 0.01), method='bounded', options={'xatol': 1e-8})
                if res.fun < best_A:
                    best_A = res.fun
                    best_beta = res.x
            except Exception:
                pass
        
        # Recover cx, cy for best_beta
        # Re-run LP to get cx, cy
        normals = []
        for k in range(6):
            ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = linprog([0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if res.status == 0:
            cx, cy = res.x[0], res.x[1]
        else:
            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            
        S = best_A / APOTHEM
        return S, cx, cy, best_beta

    # Now the outer optimization: minimize S({p_i}) subject to distances.
    
    def objective(params):
        # params: 22 values (x1, y1, ..., x11, y11)
        pts = params.reshape(11, 2)
        
        # Check distance constraints
        # Soft penalty for violations
        penalty = 0.0
        min_dist = 2 * APOTHEM
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        # Calculate S
        S, _, _, _ = min_enclosing_hexagon(pts)
        
        return S + penalty * 10.0

    # Multi-start
    best_S = float('inf')
    best_params = None
    best_cx, best_cy, best_beta = 0, 0, 0
    
    # Initial configurations
    # 1. Lattice based
    # 2. Random
    
    # Lattice configuration (similar to current best)
    # Center (0,0), 6 around, 4 more?
    # Let's use the axial_1 candidate from the previous program as a good start
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    init_pts = []
    for q, r in axial_1:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_pts.append((x, y))
    init_params = np.array(init_pts).flatten()
    
    starts = [init_params]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random points in a circle of radius 4
        radii = rng.uniform(0, 4, 11)
        angles = rng.uniform(0, 2 * math.pi, 11)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts.flatten())
        
    # Optimize each start
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_S:
                # Verify no hard violations (penalty should be 0 or close)
                pts_opt = res.x.reshape(11, 2)
                has_violation = False
                for i in range(11):
                    for j in range(i + 1, 11):
                        if np.linalg.norm(pts_opt[i] - pts_opt[j]) < 2 * APOTHEM - 1e-9:
                            has_violation = True
                            break
                    if has_violation:
                        break
                
                if not has_violation:
                    S, cx, cy, beta = min_enclosing_hexagon(pts_opt)
                    if S < best_S:
                        best_S = S
                        best_params = res.x
                        best_cx, best_cy, best_beta = cx, cy, beta
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial
        pts = np.array(init_pts)
        best_S, best_cx, best_cy, best_beta = min_enclosing_hexagon(pts)
        best_params = pts.flatten()

    # Construct output
    pts_final = best_params.reshape(11, 2)
    inner = []
    for x, y in pts_final:
        # All inner hexagons are aligned (angle 0)
        inner.append([float(x), float(y), 0.0])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation

