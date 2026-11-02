import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Local hexagon vertices (side 1, angle 0)
    # Pointy top: (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # Flat top: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Let's use Flat Top for consistency with standard "hexagon" orientation if angle 0 is flat.
    # The task doesn't specify initial orientation, but angle is a variable.
    # Let's define local vertices for a unit hexagon with flat top (side 1).
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    local_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals for SAT (unit normals to the sides)
    # Sides are between v0-v1, v1-v2, etc.
    # Normal to side v0-v1: perpendicular to v1-v0.
    # v1-v0 = (-0.5, sqrt(3)/2). Normal = (sqrt(3)/2, 0.5) or (-sqrt(3)/2, -0.5).
    # Let's just compute normals from edges.
    edges = np.roll(local_verts, -1, axis=0) - local_verts
    normals = np.zeros((6, 2))
    for i in range(6):
        e = edges[i]
        # Normal is (-e_y, e_x) normalized
        n = np.array([-e[1], e[0]])
        n = n / np.linalg.norm(n)
        normals[i] = n
        
    def get_hex_verts(center, angle_deg):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return local_verts @ R.T + center

    def sat_overlap(v1, v2):
        # v1, v2 are (6,2) arrays
        # Project onto normals of v1 and v2
        # Since v2 is rotated, its normals are rotated too.
        # But we can just project onto the 6 normals of v1 and 6 normals of v2.
        # Actually, for convex polygons, if there's a separating axis, it's normal to an edge of one of the polygons.
        # We need normals of v1 and normals of v2.
        
        # Normals of v1
        n1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.array([[-n[1], n[0]] for n in n1])
        n1 = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
        
        # Normals of v2
        n2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.array([[-n[1], n[0]] for n in n2])
        n2 = n2 / np.linalg.norm(n2, axis=1, keepdims=True)
        
        all_normals = np.vstack([n1, n2])
        
        for n in all_normals:
            p1 = np.dot(v1, n)
            p2 = np.dot(v2, n)
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            if max1 < min2 or max2 < min1:
                return False, np.min(np.max(p1) - np.min(p2), np.max(p2) - np.min(p1)) # Distance is positive if separated
        return True, 0.0 # Overlapping, distance 0 (or negative if we defined it that way, but here we just return 0 for penalty)

    def objective(params):
        # params: 11 * (x, y, angle)
        # x, y in [-5, 5], angle in [0, 60] (symmetry)
        hexes = []
        for i in range(N):
            x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
            hexes.append(get_hex_verts(np.array([x, y]), ang))
        
        # 1. Calculate required outer side length S
        # Outer hexagon: Center (0,0), Angle 0, Side S.
        # Constraints for point (x,y) to be in outer hexagon (Flat Top, Side S):
        # |x| <= S
        # |sqrt(3)*y + x| <= S*sqrt(3) ? 
        # Let's verify Flat Top hexagon inequalities.
        # Vertices: (S,0), (S/2, S*sqrt(3)/2), ...
        # Side 1 (top right): (S,0) to (S/2, S*sqrt(3)/2).
        # Vector: (-S/2, S*sqrt(3)/2). Normal: (S*sqrt(3)/2, S/2) -> (sqrt(3), 1).
        # Eq: sqrt(3)*x + y = S*sqrt(3) + 0 = S*sqrt(3).
        # So: sqrt(3)*x + y <= S*sqrt(3).
        # Side 2 (top): (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2).
        # Vector: (-S, 0). Normal: (0, S) -> (0, 1).
        # Eq: y = S*sqrt(3)/2.
        # So: y <= S*sqrt(3)/2.
        
        # General inequalities for Flat Top Hexagon Side S:
        # 1. y <= S*sqrt(3)/2
        # 2. -y <= S*sqrt(3)/2  => |y| <= S*sqrt(3)/2
        # 3. sqrt(3)*x + y <= S*sqrt(3)
        # 4. -sqrt(3)*x + y <= S*sqrt(3) => |sqrt(3)*x| + y <= S*sqrt(3) ? No.
        # Let's check the vertex (S, 0).
        # sqrt(3)*S + 0 = S*sqrt(3). Correct.
        # Check vertex (S/2, S*sqrt(3)/2).
        # sqrt(3)*S/2 + S*sqrt(3)/2 = S*sqrt(3). Correct.
        
        # So the constraints are:
        # |y| <= S*sqrt(3)/2
        # |sqrt(3)*x + y| <= S*sqrt(3)
        # |sqrt(3)*x - y| <= S*sqrt(3)
        
        # From these, we can derive S for a point (x,y):
        # S >= 2*|y|/sqrt(3)
        # S >= |sqrt(3)*x + y|/sqrt(3)
        # S >= |sqrt(3)*x - y|/sqrt(3)
        
        max_S = 0.0
        for h in hexes:
            # h is (6,2)
            ys = h[:, 1]
            xs = h[:, 0]
            s1 = 2 * np.abs(ys) / np.sqrt(3)
            s2 = np.abs(np.sqrt(3) * xs + ys) / np.sqrt(3)
            s3 = np.abs(np.sqrt(3) * xs - ys) / np.sqrt(3)
            max_S = max(max_S, np.max(s1), np.max(s2), np.max(s3))
            
        # 2. Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                overlap, dist = sat_overlap(hexes[i], hexes[j])
                if overlap:
                    # We want to push them apart.
                    # Use a soft penalty based on how deep they are?
                    # SAT doesn't easily give depth.
                    # Approximation: If overlapping, add penalty proportional to 1/dist? No, dist is 0.
                    # Let's use a simpler metric: Distance between centers.
                    # If centers are too close, they likely overlap.
                    # But angles matter.
                    # For now, just a large constant penalty if overlap.
                    penalty += 10.0
        return max_S + penalty

    # Bounds
    # x, y in [-5, 5]
    # angle in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 60.0))
        
    # Initial guess: Compact cluster
    # 7 hexes in center, 4 extra.
    # We'll use differential evolution for global search.
    # To save time, we'll limit the number of generations and population size.
    
    best_result = None
    best_score = float('inf')
    
    # Use DE for a few seconds, then local polish.
    # Given 150s limit, we can afford a moderate DE run.
    
    # Strategy:
    # 1. Run DE for ~60s.
    # 2. Take best result, run L-BFGS-B polish for ~30s.
    # 3. Repeat with different seeds if time allows.
    
    def run_de(seed):
        nonlocal best_result, best_score
        end_de = time.time() + 50.0
        # DE is stochastic, so we can't easily stop it mid-flight in scipy.
        # We can wrap the objective to check time, but DE calls it many times.
        # Better: Run a fixed number of generations that fits in the time.
        # Est. 1000 evals per gen? 30 dims.
        # Let's try maxiter=100, popsize=10. 1000 evals.
        # If each eval takes 1ms, 1s.
        # If 10ms, 10s.
        # Let's assume 10ms per eval.
        
        res = differential_evolution(
            objective, 
            bounds, 
            seed=seed, 
            maxiter=50, 
            popsize=15, 
            tol=1e-4,
            polish=False
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
        return res

    # Run 2-3 DE iterations
    for seed in [42, 123, 999]:
        if time.time() > deadline - 60:
            break
        run_de(seed)
        
    # Local Polish
    if best_result is not None:
        # L-BFGS-B
        # We need a gradient? No, BFGS approximates.
        # But the objective is non-smooth (max).
        # We can use the soft-max version for polishing.
        
        def objective_smooth(params, beta=10.0):
            hexes = []
            for i in range(N):
                x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
                hexes.append(get_hex_verts(np.array([x, y]), ang))
            
            S_vals = []
            for h in hexes:
                ys = h[:, 1]
                xs = h[:, 0]
                s1 = 2 * np.abs(ys) / np.sqrt(3)
                s2 = np.abs(np.sqrt(3) * xs + ys) / np.sqrt(3)
                s3 = np.abs(np.sqrt(3) * xs - ys) / np.sqrt(3)
                S_vals.extend([np.max(s1), np.max(s2), np.max(s3)])
            
            S_vals = np.array(S_vals)
            # Soft max
            log_sum_exp = np.log(np.sum(np.exp(beta * S_vals))) / beta
            # Shift to avoid overflow
            # Actually, standard log-sum-exp:
            m = np.max(S_vals)
            log_sum_exp = m + np.log(np.sum(np.exp(beta * (S_vals - m)))) / beta
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    overlap, dist = sat_overlap(hexes[i], hexes[j])
                    if overlap:
                        penalty += 10.0
            return log_sum_exp + penalty

        res_polish = minimize(
            objective_smooth, 
            best_result, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res_polish.fun < best_score:
            best_score = res_polish.fun
            best_result = res_polish.x
            
    # Construct final output
    if best_result is None:
        # Fallback: trivial layout
        inner = [[0.0, 0.0, 0.0]] + [[4.0*k, 0.0, 0.0] for k in range(1, 11)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x, y, ang = best_result[3*i], best_result[3*i+1], best_result[3*i+2]
        inner.append([x, y, ang])
        
    # Calculate actual S for the final configuration
    # Re-use logic from objective
    hexes = []
    for i in range(N):
        x, y, ang = best_result[3*i], best_result[3*i+1], best_result[3*i+2]
        hexes.append(get_hex_verts(np.array([x, y]), ang))
        
    max_S = 0.0
    for h in hexes:
        ys = h[:, 1]
        xs = h[:, 0]
        s1 = 2 * np.abs(ys) / np.sqrt(3)
        s2 = np.abs(np.sqrt(3) * xs + ys) / np.sqrt(3)
        s3 = np.abs(np.sqrt(3) * xs - ys) / np.sqrt(3)
        max_S = max(max_S, np.max(s1), np.max(s2), np.max(s3))
        
    return inner, (0.0, 0.0), max_S, 0.0

# EVOLVE-BLOCK-END
