import numpy as np
import time
from scipy.optimize import lsq_nonlinear

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Precompute geometry constants
    sqrt3 = np.sqrt(3.0)
    # Vertices of a unit hexagon centered at origin, angle 0
    # Angles: 0, 60, 124, 180, 240, 300 degrees
    hex_angles = np.radians(np.arange(0, 360, 60))
    unit_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Normals for the 6 sides of the unit hexagon
    # Side 0 connects v0 and v1. Normal is perpendicular to edge, pointing out.
    # Edge vector: v1 - v0. Normal: rotate -90 or +90.
    # For a regular polygon, outward normals point towards the midpoints of sides.
    # Midpoint angle of side 0 (between 0 and 60) is 30 deg.
    normal_angles = np.radians(np.arange(30, 360, 60))
    unit_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Support function for unit hexagon in direction theta:
    # h(theta) = max_k (v_k . n_theta)
    # Since it's a regular hexagon, the support function is piecewise cosine.
    # But for penalty, we can just use the max over the 6 vertices or normals.
    # Actually, for distance between two convex polygons, we can use support functions.
    # However, a simpler differentiable approximation for "distance between two sets"
    # is not trivially smooth. 
    # Alternative: Use a soft penalty based on the maximum overlap depth.
    # Overlap depth of two convex polygons A, B:
    # d = max_{n in normals(A) U normals(B)} (min_{p in A} p.n - max_{q in B} q.n)
    # If d > 0, they are separated. If d < 0, they overlap.
    # We want to maximize the minimum separation.
    # Penalty term: sum over pairs of max(0, -separation_ij)^2
    
    # Let's define a function that computes the minimum separation between two unit hexagons
    # at positions p1 and p2.
    # Separation along normal n: (p2 - p1).n + (h_B(n) - h_A(-n)) ? No.
    # Distance from p1 to p2 along n:
    # Support of A in direction n: h_A(n) = max(v_A . n)
    # Support of B in direction -n: h_B(-n) = max(v_B . (-n)) = -min(v_B . n)
    # The gap along n is: (p2 . n + h_B(n)) - (p1 . n + h_A(n)) ?
    # Let's be precise.
    # Let A be at p1, B at p2.
    # Projected interval of A on axis n: [ (p1 - h_A(-n)).n, (p1 + h_A(n)).n ]
    # Wait, h_A(n) = max_{v in A} v.n. Since A is centered at 0, v are unit hex verts.
    # So interval is [ p1.n - h_A(-n), p1.n + h_A(n) ]
    # Similarly for B: [ p2.n - h_B(-n), p2.n + h_B(n) ]
    # Gap = min(0, (p1.n + h_A(n)) - (p2.n - h_B(-n)) ) is overlap if positive?
    # Let's define separation s_n = (p2.n - h_B(n)) - (p1.n + h_A(n))
    # If s_n > 0, B is to the right of A along n.
    # Minimum separation over all normals n is the actual distance if separated.
    # If they overlap, the minimum separation is negative.
    # We want to penalize negative separations.
    
    # Precompute h_A(n) and h_B(-n) for the 12 normals (6 for A, 6 for B).
    # Since both are unit hexagons, h_A(n) = h_unit(n) and h_B(n) = h_unit(n).
    # h_unit(n) for a unit hexagon:
    # The support function of a regular polygon of side 1 (apothem a = sqrt(3)/2) is:
    # h(theta) = a * cos((theta - theta_k) mod 60 deg) ... roughly.
    # More simply, we can precompute the support values for the 12 relevant normal directions
    # relative to the hexagon's orientation (which is 0 for both).
    
    # The 12 normals we need to check for separation are the 6 normals of Hex 1 and 6 of Hex 2.
    # Since both are aligned, the set of normals is the same 6 directions (and their opposites).
    # Actually, for SAT, we check normals of both. Since they are identical shapes, we check the 6 normals.
    
    # Let's compute h_unit(n_k) for the 6 outward normals.
    # n_k are at 30, 90, ...
    # h_unit(n_k) = max(unit_verts . n_k)
    # Due to symmetry, h_unit(n_k) is constant for all 6 normals?
    # n_0 = (cos30, sin30). v_0=(1,0). dot = cos30 = sqrt(3)/2.
    # v_1=(0.5, sqrt(3)/2). dot = 0.5*sqrt(3)/2 + 3/4 = sqrt(3)/4 + 3/4.
    # Wait, apothem is sqrt(3)/2.
    # The support function in the direction of the normal to a side is exactly the apothem.
    # So h_unit(n_k) = sqrt(3)/2 for all 6 normals.
    # What about the opposite direction? h_unit(-n_k) = sqrt(3)/2.
    
    # So, for two aligned unit hexagons at p1, p2:
    # Separation along normal n_k:
    # s_k = (p2 - p1) . n_k - (h_A(n_k) + h_B(-n_k)) ?
    # Interval A: [p1.n - a, p1.n + a] where a = sqrt(3)/2.
    # Interval B: [p2.n - a, p2.n + a].
    # Gap = (p2.n - a) - (p1.n + a) = (p2 - p1).n - 2a.
    # This gap is positive if they are separated along n.
    # The minimum gap over all 6 normals is the "signed distance" (negative if overlap).
    
    a = sqrt3 / 2.0
    
    def min_separation(p1, p2):
        """
        Returns the minimum separation between two unit hexagons at p1 and p2.
        Positive if separated, negative if overlapping.
        """
        diff = p2 - p1
        # Projections onto the 6 normals
        # normals are unit vectors at 30, 90, 150, 210, 270, 330
        # We can compute dot products efficiently
        # n_0 = (sqrt3/2, 1/2)
        # n_1 = (0, 1)
        # n_2 = (-sqrt3/2, 1/2)
        # n_3 = (-sqrt3/2, -1/2)
        # n_4 = (0, -1)
        # n_5 = (sqrt3/2, -1/2)
        
        dx, dy = diff
        # dot products
        d0 = dx * (sqrt3/2) + dy * 0.5
        d1 = dy
        d2 = -dx * (sqrt3/2) + dy * 0.5
        d3 = -dx * (sqrt3/2) - dy * 0.5
        d4 = -dy
        d5 = dx * (sqrt3/2) - dy * 0.5
        
        # Gaps
        g0 = d0 - 2*a
        g1 = d1 - 2*a
        g2 = d2 - 2*a
        g3 = d3 - 2*a
        g4 = d4 - 2*a
        g5 = d5 - 2*a
        
        return min(g0, g1, g2, g3, g4, g5)

    def objective(params):
        """
        params: 2N array, [x0, y0, x1, y1, ...]
        Returns residual vector for lsq_nonlinear.
        We want to minimize the outer side length AND ensure separation.
        We can formulate this as minimizing a vector of residuals.
        Residuals 0..5: The 6 constraints for the outer hexagon containment.
        Residuals 6..: Penalty terms for overlaps.
        
        Actually, lsq_nonlinear minimizes sum(residuals^2).
        We can scale the overlap penalties to be comparable to the side length.
        """
        pos = params.reshape(N, 2)
        
        # 1. Outer Hexagon Side Length Constraints
        # We want the smallest S such that all points are inside.
        # S >= max_k (p_k . n_k) for all vertices of all inner hexagons?
        # No, the inner hexagons have vertices too.
        # The "required outer side" is determined by the furthest extent of the *union* of inner hexagons.
        # The support function of the union is the max of the support functions.
        # Support of inner hex i in direction n: pos_i . n + h_unit(n).
        # For the 6 directions of the outer hexagon normals (which are the same as inner normals since aligned):
        # h_union(n_k) = max_i (pos_i . n_k + a)
        # The side length S of the outer hexagon must satisfy:
        # S * a >= h_union(n_k) for all k?
        # Recall: Outer hexagon of side S has apothem S*a.
        # Containment condition: max_{p in Union} (p . n_k) <= S * a.
        # So S * a >= max_i (pos_i . n_k + a).
        # S >= (max_i (pos_i . n_k + a)) / a.
        # S >= max_i (pos_i . n_k / a + 1).
        
        # Let's compute the required S for each of the 6 directions.
        # n_k are the 6 normals.
        # We can compute pos_i . n_k for all i, k.
        
        # Vectorized computation of pos . normals
        # pos: (N, 2)
        # normals: (6, 2)
        # dots: (N, 6)
        dots = np.empty((N, 6))
        dots[:, 0] = pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
        dots[:, 1] = pos[:, 1]
        dots[:, 2] = -pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
        dots[:, 3] = -pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
        dots[:, 4] = -pos[:, 1]
        dots[:, 5] = pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
        
        max_dots = np.max(dots, axis=0) # (6,)
        # Required S_k = (max_dots[k] + a) / a = max_dots[k]/a + 1
        required_S = max_dots / a + 1.0
        
        # The actual side length is S = max(required_S)
        # We want to minimize S.
        # In LSQ, we can return residuals that are (S - S_k) for each k?
        # No, that doesn't directly minimize S.
        # Better: return residuals that are the "violation" of a target S?
        # Or, simply return the 6 values of required_S as residuals?
        # Minimizing sum(required_S_k^2) will tend to reduce the max, but also minimize the min.
        # This is a common heuristic. To force balance, we can add a term for the mean.
        # Or, we can use a different objective:
        # Minimize S + Penalty.
        # Since lsq_nonlinear minimizes sum(r^2), we can set:
        # r[0:6] = required_S (scaled)
        # This will try to make all required_S small.
        
        # 2. Overlap Penalties
        # For each pair (i, j), compute min_separation.
        # If sep < 0, penalty = -sep.
        # We want to minimize sum(penalty^2).
        
        residuals = []
        
        # Scale factor for S to make it comparable to overlap penalties.
        # S is around 5-10. Overlap can be deep.
        # Let's use a weight.
        S_weight = 1.0
        overlap_weight = 10.0 # High weight to ensure no overlap
        
        for k in range(6):
            residuals.append(S_weight * required_S[k])
            
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                sep = min_separation(pos[i], pos[j])
                if sep < 0:
                    # Penalty for overlap
                    residuals.append(overlap_weight * (-sep))
                else:
                    # Small residual to keep them close? No, 0 is fine.
                    residuals.append(0.0)
                    
        return np.array(residuals)

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1).flatten())
    
    # 2. Circular
    init2 = np.zeros(N * 2)
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 3.0
        init2[i*2] = r * np.cos(angle)
        init2[i*2+1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, N * 2)
    configs.append(init3)
    
    best_params = None
    best_score = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
            
        try:
            # Use lsq_nonlinear
            # We need a good initial guess. The lattice is usually good.
            # If the initial guess has overlaps, the penalty will be high,
            # and the optimizer should push them apart.
            
            res = lsq_nonlinear(objective, init, method='trf', 
                                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                max_nfev=2000)
            
            # Evaluate final score
            pos = res.x.reshape(N, 2)
            # Calculate exact side length
            dots = np.empty((N, 6))
            dots[:, 0] = pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
            dots[:, 1] = pos[:, 1]
            dots[:, 2] = -pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
            dots[:, 3] = -pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
            dots[:, 4] = -pos[:, 1]
            dots[:, 5] = pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
            max_dots = np.max(dots, axis=0)
            side = np.max(max_dots / a + 1.0)
            
            # Check validity (strict SAT)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if min_separation(pos[i], pos[j]) < -1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and side < best_score:
                best_score = side
                best_params = res.x.copy()
                
        except Exception as e:
            pass

    # Fallback
    if best_params is None:
        best_params = configs[0]
        pos = best_params.reshape(N, 2)
        dots = np.empty((N, 6))
        dots[:, 0] = pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
        dots[:, 1] = pos[:, 1]
        dots[:, 2] = -pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
        dots[:, 3] = -pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
        dots[:, 4] = -pos[:, 1]
        dots[:, 5] = pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
        max_dots = np.max(dots, axis=0)
        best_score = np.max(max_dots / a + 1.0)
        
    # Final check and formatting
    pos = best_params.reshape(N, 2)
    
    # Re-calculate side precisely
    dots = np.empty((N, 6))
    dots[:, 0] = pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
    dots[:, 1] = pos[:, 1]
    dots[:, 2] = -pos[:, 0] * (sqrt3/2) + pos[:, 1] * 0.5
    dots[:, 3] = -pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
    dots[:, 4] = -pos[:, 1]
    dots[:, 5] = pos[:, 0] * (sqrt3/2) - pos[:, 1] * 0.5
    max_dots = np.max(dots, axis=0)
    final_side = float(np.max(max_dots / a + 1.0))
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
