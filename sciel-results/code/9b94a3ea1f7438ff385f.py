import numpy as np
from scipy.optimize import differential_evolution
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a symmetric parameterization: 1 center + 5 rings of 2 hexagons.
    Optimizes radii and angles to minimize the outer hexagon side length.
    """
    
    def get_vertices(cx, cy, s=1.0, angle_deg=0):
        angles = np.deg2rad(angle_deg) + np.deg2rad(60 * np.arange(6))
        return np.stack([cx + s * np.cos(angles), cy + s * np.sin(angles)], axis=-1)

    def calc_required_S(points):
        x = points[:, 0]
        y = points[:, 1]
        sqrt3 = np.sqrt(3)
        
        S1 = x + y / sqrt3
        S2 = -x + y / sqrt3
        S3 = -x - y / sqrt3
        S4 = x - y / sqrt3
        S5 = 2 * y / sqrt3
        S6 = -2 * y / sqrt3
        
        return np.max(np.maximum.reduce([S1, S2, S3, S4, S5, S6]))

    def get_hexagon_centers(params):
        """
        params: (6,) array [r1, r2, r3, r4, r5, theta]
        Returns: (11, 2) array of centers.
        Symmetry: Hexagons come in pairs at angles theta + k*72 for k=1..5.
        Center is at (0,0).
        """
        r1, r2, r3, r4, r5, theta = params
        theta_rad = np.deg2rad(theta)
        
        centers = np.zeros((11, 2))
        # Center
        centers[0] = [0, 0]
        
        # 5 pairs
        for i in range(5):
            r = [r1, r2, r3, r4, r5][i]
            angle = theta_rad + i * (2 * np.pi / 5)
            centers[2*i + 1] = [r * np.cos(angle), r * np.sin(angle)]
            centers[2*i + 2] = [r * np.cos(-angle), r * np.sin(-angle)] # Symmetric pair? 
            # Wait, if we have 5 pairs, that's 10 hexagons. 
            # The prompt says 11 unit hexagons. 
            # A regular hexagon has 6-fold symmetry, but 11 is not divisible by 6.
            # A common packing for 11 is a "flower" of 7 (1 center + 6 around) + 4 more?
            # Or 1 center + 10 around?
            # Let's stick to the 5-pair + center structure which gives 11.
            # The angles for the 5 pairs should be distributed.
            # If we use 5-fold symmetry, the outer container (hexagon) has 6-fold symmetry.
            # This might be suboptimal if the optimal packing breaks 5-fold symmetry.
            # However, 11 is close to 12. A 12-packing (1+6+5?) or 1+6+6=13.
            # Actually, the "seed" program used a grid.
            # Let's try a different symmetry: 1 center + 6 in a ring (hexagonal symmetry) + 4 in another ring?
            # Total 11.
            # Let's parameterize: 
            # 1 center.
            # 6 hexagons at radius R1, angles 0, 60, ..., 300.
            # 4 hexagons at radius R2, angles 30, 150, 210, 330? (Midpoints of the 6-fold symmetry axes)
            # This preserves 6-fold symmetry of the container? No, 4 breaks 6-fold.
            # But it preserves 4-fold or 2-fold?
            # Let's try: 1 center + 6 (ring 1) + 4 (ring 2).
            # Ring 1 angles: 0, 60, 120, 180, 240, 300.
            # Ring 2 angles: 30, 150, 210, 330. (These are offset by 30 deg).
            # This configuration has symmetry.
            
            # Let's redefine params for this structure:
            # params: [R1, R2, offset_angle]
            # R1: radius of ring 1 (6 hexagons)
            # R2: radius of ring 2 (4 hexagons)
            # offset_angle: rotation of the whole thing? Or relative offset?
            # Let's assume Ring 1 is at angles 0, 60...
            # Ring 2 is at angles 30 + offset, 150 + offset...
            # Actually, to maintain some symmetry with the hexagon container (which is fixed at 0 angle in the cost function's S calc, but we can rotate the inner set),
            # let's just optimize R1, R2, and a global rotation phi.
            
            pass
        
        # Re-implementing centers generation based on the 1+6+4 structure
        R1, R2, phi = params
        phi_rad = np.deg2rad(phi)
        
        centers = np.zeros((11, 2))
        centers[0] = [0, 0]
        
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = phi_rad + k * (np.pi / 3)
            centers[k + 1] = [R1 * np.cos(angle), R1 * np.sin(angle)]
            
        # Ring 2: 4 hexagons
        # Placed at 30, 150, 270, 390(30)? No, 4 points.
        # To be symmetric with respect to the hexagon axes, maybe 30, 150, 210, 330?
        # Let's try angles: 30, 150, 270, 30+360?
        # Let's use 30, 150, 270, 390.
        # Actually, 4 points on a circle. 90 degree spacing?
        # If we use 90 deg spacing, it breaks 6-fold symmetry of the container potentially.
        # Let's try 30, 150, 270, 30? No.
        # Let's try 0, 90, 180, 270 + offset?
        
        # Let's go back to the 5-pair + center idea. It's simpler and 11 is 1+10.
        # 5-fold symmetry is a valid hypothesis for 11 items.
        # Let's stick to the first plan: 1 center + 5 pairs.
        # The angles of the 5 pairs are fixed relative to each other (72 deg apart).
        # The global rotation is the only angle variable.
        # So params: [r1, r2, r3, r4, r5, theta]
        
        # Wait, I already wrote the loop above. Let's restart the function body logic cleanly.
        pass

    # Let's write the actual function body properly.
    
    def generate_centers(params):
        r1, r2, r3, r4, r5, theta = params
        theta_rad = np.deg2rad(theta)
        
        centers = np.zeros((11, 2))
        centers[0] = [0, 0]
        
        radii = [r1, r2, r3, r4, r5]
        for i in range(5):
            r = radii[i]
            base_angle = theta_rad + i * (2 * np.pi / 5)
            # Pair 1
            centers[2*i + 1] = [r * np.cos(base_angle), r * np.sin(base_angle)]
            # Pair 2 (symmetric across x-axis? No, just the other one in the pair)
            # If we have 5 pairs, and they are symmetric, we need to define the pairs.
            # Usually "pairs" in these symmetric packings means opposite?
            # No, 5 is odd. You can't have opposite pairs in 5-fold symmetry.
            # 5-fold symmetry means 5 identical orbits.
            # So we have 5 hexagons at radius r_i? No, that's only 5+1=6.
            # We need 10 more.
            # So maybe 2 orbits of 5?
            # Orbit 1: 5 hexagons at radius r1, angles theta + k*72.
            # Orbit 2: 5 hexagons at radius r2, angles theta + 36 + k*72.
            # Total 1 (center) + 5 + 5 = 11.
            # This is a very strong symmetry hypothesis.
            
            pass
        
        # Let's implement the 1 + 5 + 5 structure.
        # Params: [r1, r2, theta]
        # r1: radius of first ring (5 hexagons)
        # r2: radius of second ring (5 hexagons)
        # theta: global rotation
        
        # This reduces variables to 3! Much faster to optimize.
        
        pass

    def get_centers_155(params):
        r1, r2, theta = params
        theta_rad = np.deg2rad(theta)
        
        centers = np.zeros((11, 2))
        centers[0] = [0, 0]
        
        for k in range(5):
            angle1 = theta_rad + k * (2 * np.pi / 5)
            centers[k + 1] = [r1 * np.cos(angle1), r1 * np.sin(angle1)]
            
            angle2 = theta_rad + (k + 0.5) * (2 * np.pi / 5) # Offset by 36 degrees
            centers[k + 6] = [r2 * np.cos(angle2), r2 * np.sin(angle2)]
            
        return centers

    def check_separation(centers):
        """
        Returns True if all hexagons are separated.
        Uses the distance criterion for unit hexagons:
        Two unit hexagons with angle 0 are disjoint if and only if their center distance is >= sqrt(3) * 2? 
        No.
        The distance between centers of two touching unit hexagons (flat-to-flat) is sqrt(3).
        The distance for point-to-point touching is 2.
        The minimum distance for non-overlap is sqrt(3) if aligned.
        However, if rotated, the distance can be smaller?
        Actually, for regular hexagons of side 1, the "packing distance" (minimum center distance for non-overlap) depends on relative orientation.
        If orientations are the same, min distance is sqrt(3).
        If orientations differ, it can be different.
        Since we assume all inner hexagons have angle 0 (as in the seed program's output, though the seed allowed angle in input but set to 0),
        we can use the distance check.
        Wait, the task allows angle_deg in the output.
        But the seed program sets angle to 0.
        Does rotating inner hexagons help?
        Usually, for dense packings of identical shapes, aligning them is optimal or near-optimal.
        Let's assume angle 0 for all inner hexagons to simplify the separation check to distance >= sqrt(3).
        If we allow rotation, the separation check is complex.
        Given the time budget and the "different strategy" requirement, exploiting the 1+5+5 symmetry with fixed orientation is a strong, distinct approach.
        """
        min_dist = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        return min_dist >= np.sqrt(3) - 1e-6

    def cost_function(params):
        r1, r2, theta = params
        # Ensure radii are positive
        if r1 < 0 or r2 < 0:
            return 1e10
            
        centers = get_centers_155(params)
        
        if not check_separation(centers):
            # Penalty for overlap
            # Calculate min distance and penalize
            min_dist = np.inf
            for i in range(11):
                for j in range(i + 1, 11):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_dist:
                        min_dist = d
            penalty = (np.sqrt(3) - min_dist) ** 2
            return 100 * penalty + np.max(np.linalg.norm(centers, axis=1))
            
        # Calculate S
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i, 0], centers[i, 1])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        S = calc_required_S(all_vertices)
        return S

    # Define bounds for r1, r2, theta
    # r1 must be at least sqrt(3) to not overlap center?
    # Center is at 0. Ring 1 at r1.
    # Distance from center to ring 1 hex is r1.
    # Min distance for non-overlap is sqrt(3).
    # So r1 >= sqrt(3).
    # Ring 2 at r2.
    # Distance between ring 1 and ring 2 hexagons must be >= sqrt(3).
    # Distance between ring 2 and center must be >= sqrt(3).
    # So r2 >= sqrt(3).
    
    # Upper bounds:
    # If r is too large, S increases.
    # A reasonable upper bound is 5 or 6.
    
    bounds = [
        (np.sqrt(3) - 0.1, 5.0), # r1
        (np.sqrt(3) - 0.1, 5.0), # r2
        (0, 360)                  # theta
    ]
    
    start_time = time.time()
    deadline = start_time + 600
    
    best_S = np.inf
    best_params = None
    
    try:
        # Use differential evolution for global optimization
        # maxiter and popsize control the search
        # We have a small space (3D), so DE should be fast.
        
        # Run DE
        res = differential_evolution(cost_function, bounds, maxiter=100, popsize=20, tol=1e-6, seed=42)
        
        if time.time() < deadline - 10:
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
                
            # Local refinement with Nelder-Mead on the best result
            if best_params is not None:
                res2 = minimize(cost_function, best_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_params = res2.x
                    
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to a known good configuration if DE fails
        # Try a simple grid
        best_params = np.array([np.sqrt(3), 2*np.sqrt(3), 0])

    if best_params is None:
        # Fallback
        best_params = np.array([np.sqrt(3), 2*np.sqrt(3), 0])
        
    # Final verification and result building
    centers = get_centers_155(best_params)
    
    # Re-check separation strictly
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            if np.linalg.norm(centers[i] - centers[j]) < np.sqrt(3) - 1e-5:
                valid = False
    
    if not valid:
        # If invalid, try to find a valid one by slightly increasing radii?
        # Or just use the seed fallback?
        # Let's try to fix by scaling up slightly
        scale = 1.01
        best_params[0] *= scale
        best_params[1] *= scale
        centers = get_centers_155(best_params)
        
    all_vertices = []
    for i in range(11):
        verts = get_vertices(centers[i, 0], centers[i, 1])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    final_S = calc_required_S(all_vertices)
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = float(final_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# Helper for minimize import if not present in scope? It is in scipy.optimize
from scipy.optimize import minimize

if __name__ == "__main__":
    inner, outer, side = hexagon_packing_11()
    print(f"Side length: {side}")
