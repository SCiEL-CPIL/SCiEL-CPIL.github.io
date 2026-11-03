import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # We are placing N=11 unit hexagons.
    # The current best score is ~9.9256 which corresponds to a side length S ~ 1.0085 (since score = seed/S, seed=10? No, score 1.0 is seed.
    # Wait, the prompt says: "score is the outer side length... The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # Seed program returned side=40.0. So score = 40.0 / side_length.
    # Current score 9.9256 implies side_length = 40.0 / 9.9256 ≈ 4.03.
    # Let's check the current program's side calculation:
    # max_x = (2+0.01)*sqrt(3) ≈ 3.49
    # max_y = 4+0.02 = 4.02
    # s_x = 2*3.49/1.732 ≈ 4.03
    # s_y = 4.02
    # side = max(4.03, 4.02) + 0.01 = 4.04.
    # Score = 40 / 4.04 ≈ 9.90. Close enough.
    
    # The goal is to minimize the outer side length S.
    # To get a higher score, we need a smaller S.
    # The current arrangement is a "flower" of 1 center + 6 neighbors + 4 outer.
    # Is this the optimal packing for 11 hexagons?
    # For hexagonal packing, the densest cluster of n hexagons is generally a hexagonal number shape if possible, or close to it.
    # 11 is not a hexagonal number (1, 7, 19, 37...).
    # 7 is a hexagon of radius 1 (1 center + 6 neighbors). Side length of bounding hexagon for 7 unit hexagons:
    #   Centers at 0 and dist sqrt(3).
    #   Max extent: Center at sqrt(3), vertex at sqrt(3)/2. Total dist = 1.5*sqrt(3) ≈ 2.6.
    #   S for 7: width = sqrt(3)*S >= 2*2.6 => S >= 3.0. Height = 2S >= 2*2.6 => S >= 2.6. So S=3.0.
    #   Score for 7: 40/3.0 = 13.33.
    #   But we must place 11 hexagons.
    
    # Let's try to optimize the 11-hexagon cluster more tightly.
    # The current cluster uses 11 hexagons.
    # Can we pack them tighter?
    # The "flower" of 7 is optimal for 7.
    # For 11, we add 4 more.
    # The current program adds 4 at distance 3+2eps.
    # Maybe we can interlock them better?
    
    # Let's use a local optimization / simulated annealing approach to minimize the bounding box side length.
    # We will represent the positions of the 11 hexagons.
    # Objective: Minimize S such that all hexagons fit inside a regular hexagon of side S, and are disjoint.
    # Disjointness constraint: Distance between centers >= 2 * apothem? No.
    # Two unit hexagons are disjoint if the distance between their centers is >= 2 * (distance from center to side) * 2?
    # No, the "collision radius" for two identical regular hexagons depends on their relative rotation.
    # However, if we fix all rotations to be the same (e.g., 0 or 30 degrees), the Minkowski sum of two hexagons is a hexagon.
    # The minimum distance between centers for non-overlap is the width of the Minkowski sum in that direction.
    # For two hexagons with the same orientation, the minimum center distance to be disjoint is 2 * apothem? No.
    # The apothem is sqrt(3)/2.
    # If two hexagons touch side-to-side, the distance between centers is 2 * apothem = sqrt(3).
    # If they touch vertex-to-vertex, the distance is 2 * circumradius = 2.
    # The "forbidden region" for one center relative to another is the Minkowski sum of the hexagon and the negative of the other hexagon.
    # If orientations are fixed, this is a fixed polygon.
    # To simplify, we can assume all hexagons have the same orientation (e.g., flat-top).
    # Then the constraint is that the center of hex j must be outside the Minkowski sum of hex i (translated to origin) and -hex i.
    # Actually, it's easier to just check the distance if we assume the worst case or specific orientations.
    # But the task allows rotation.
    
    # Let's stick to the discrete geometry insight:
    # The optimal packing of hexagons often aligns them with the outer hexagon or in a lattice.
    # Let's try to optimize the positions of the 11 centers using gradient descent or random restarts with local search.
    
    N = 11
    
    # Initial guess: The current program's layout
    # 1 center, 6 neighbors, 4 outer.
    # Let's generate a few candidate layouts and optimize them.
    
    def get_bounding_side(centers, angles):
        # Calculate the minimum side length of a regular hexagon (centered at 0,0, rotatable) 
        # that contains all points (vertices of the inner hexagons).
        # Actually, we need to contain the *hexagons*, not just centers.
        # So we need to find the support function of the union of hexagons in the directions of the outer hexagon sides.
        
        # Outer hexagon with side S, center (0,0), rotation phi.
        # The condition for a point p to be inside the outer hexagon is:
        # max over k=0..5 of dot(p, n_k) <= S * apothem_outer?
        # No. The distance from center to side is S * sqrt(3)/2.
        # Let d_k be the unit normal to the k-th side of the outer hexagon.
        # The constraint is: for all inner hexagons H_i, and for all vertices v of H_i:
        # dot(v, d_k) <= S * sqrt(3)/2.
        # This is equivalent to: max_{v in H_i} dot(v, d_k) <= S * sqrt(3)/2.
        # max_{v in H_i} dot(v, d_k) = dot(center_i, d_k) + max_{u in H_0} dot(u, d_k).
        # H_0 is a unit hexagon at origin.
        # The support function of a unit hexagon in direction d is:
        # h(d) = max_{v in H_0} dot(v, d).
        
        # So for a given outer rotation phi, the required side S is:
        # S >= (max_i [ dot(c_i, d_k(phi)) + h(d_k(phi)) ]) / (sqrt(3)/2)
        
        # We can choose phi to minimize S.
        # S(phi) = max_k [ (max_i dot(c_i, d_k(phi)) + h(d_k(phi))) / (sqrt(3)/2) ]
        
        # We want min_phi S(phi).
        
        # Precompute vertices of unit hexagon (rotated by angle_degrees).
        # Note: The inner hexagons can also rotate. The task says "each inner hexagon may translate and rotate".
        # This adds complexity. We can fix inner rotations to optimize outer, or optimize both.
        # Given the time limit, let's assume inner rotations are fixed to 0 or 30 to simplify, or optimize them too.
        # Let's assume inner rotations are 0 (flat top) for now, and outer rotation is variable.
        # Actually, the current program uses inner angle 30 and outer angle 30.
        # Let's allow outer rotation to vary.
        
        # Vertices of unit hexagon (side 1) with flat top (angle 0):
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Wait, side length 1.
        # Vertices at angles 0, 60, 120...
        # v_k = (cos(k*60), sin(k*60)).
        
        # Support function h(d) for unit hexagon:
        # If d is at angle theta, h(d) = max_k dot(v_k, d).
        
        # Let's compute S for a given set of centers and a given outer rotation phi.
        # d_k(phi) are the normals to the sides of the outer hexagon.
        # For a regular hexagon, the normals are at angles phi, phi+60, ..., phi+300.
        # (Assuming side is perpendicular to normal).
        
        # h(d) for unit hexagon (vertices at 0, 60, ...):
        # If d is at angle 0, h(d) = 1.
        # If d is at angle 30, h(d) = cos(30) = sqrt(3)/2.
        
        # We need to be careful with the definition of "side length" vs "apothem".
        # Apothem a = S * sqrt(3)/2.
        # Constraint: max_i (dot(c_i, d_k) + h(d_k)) <= a = S * sqrt(3)/2.
        # So S >= 2/sqrt(3) * max_k (max_i dot(c_i, d_k) + h(d_k)).
        
        # Let's define a function to compute the required S for given centers and outer rotation phi.
        
        # Vertices of unit hexagon (angle 0):
        # v0 = (1, 0)
        # v1 = (0.5, sqrt(3)/2)
        # ...
        
        # h(d) where d = (cos theta, sin theta)
        # h(d) = max( cos(theta), cos(theta-60), cos(theta-120), cos(theta-180), cos(theta-240), cos(theta-300) )
        # This is the same as max_k cos(theta - k*60).
        # This function is periodic with period 60.
        # It peaks at 1 when theta is multiple of 60, and is sqrt(3)/2 when theta is 30+60k.
        
        def h_theta(theta):
            # theta in radians
            # Reduce to [0, 60)
            t = theta % (np.pi/3)
            # In [0, 60), the max is cos(t) (since cos(0)=1, cos(60)=0.5, and the other cosines are smaller)
            # Wait.
            # cos(0) = 1
            # cos(-60) = 0.5
            # cos(-120) = -0.5
            # So max is cos(t) for t in [0, 30] and cos(60-t) for t in [30, 60]?
            # Let's check t=30. cos(30)=0.866. cos(-30)=0.866.
            # So h(theta) = cos(min(t, pi/3 - t))?
            # Yes, for t in [0, pi/3], h(t) = cos(t) if t <= pi/6, else cos(pi/3 - t).
            # Actually, simpler: h(theta) = cos( (theta % (pi/3)) ) is not quite right for the second half.
            # Let's just compute the 6 values.
            vals = [np.cos(theta - k * np.pi/3) for k in range(6)]
            return np.max(vals)
            
        # Outer rotation phi. Normals at phi + k*60.
        # We want to minimize S(phi) = max_k [ (max_i dot(c_i, d_k) + h(d_k)) * 2/sqrt(3) ]
        
        # To find min S, we can sample phi or use optimization.
        # Since N=11 is small, we can do a local search on centers AND phi.
        
        # For now, let's just compute S for a fixed phi (e.g., 0) and see if we can optimize centers.
        # Actually, let's optimize centers for a fixed phi, then optimize phi.
        
        pass # Placeholder

    # Let's implement a simple local search.
    # Variables: centers (11*2) and phi (1).
    # Objective: S(phi, centers) + penalty(overlap)
    
    def compute_S(centers, angles_inner, phi_outer):
        # centers: (N, 2)
        # angles_inner: (N,) in radians
        # phi_outer: scalar radians
        
        # Support function of outer hexagon side: a = S * sqrt(3)/2
        # Constraint: for each k in 0..5:
        #   max_i ( dot(c_i, d_k) + h_i(d_k) ) <= S * sqrt(3)/2
        # where d_k = (cos(phi_outer + k*pi/3), sin(phi_outer + k*pi/3))
        # and h_i(d_k) is the support function of the i-th inner hexagon in direction d_k.
        
        # Inner hexagon i has center c_i and rotation angle_i.
        # Its vertices are c_i + R(angle_i) * v_j.
        # h_i(d) = dot(c_i, d) + max_j dot(R(angle_i) v_j, d)
        #        = dot(c_i, d) + max_j dot(v_j, R(-angle_i) d)
        #        = dot(c_i, d) + h_unit(R(-angle_i) d)
        
        # h_unit(d) for d at angle theta is as defined before.
        
        max_req_a = 0.0
        for k in range(6):
            theta_d = phi_outer + k * np.pi / 3
            d = np.array([np.cos(theta_d), np.sin(theta_d)])
            
            max_val = -np.inf
            for i in range(len(centers)):
                c_i = centers[i]
                angle_i = angles_inner[i]
                
                # Rotate d by -angle_i to get effective angle for unit hexagon
                theta_eff = theta_d - angle_i
                h_i = h_theta(theta_eff)
                
                val = np.dot(c_i, d) + h_i
                if val > max_val:
                    max_val = val
            
            # Required apothem for this side
            req_a = max_val
            if req_a > max_req_a:
                max_req_a = req_a
        
        # S = 2 * max_req_a / sqrt(3)
        return 2.0 * max_req_a / np.sqrt(3)

    def check_overlap(centers, angles_inner):
        # Check if any two inner hexagons overlap.
        # Two hexagons i and j overlap if the distance between centers is less than the Minkowski radius in that direction.
        # Simplified: If we assume all angles are similar, we can check distance > 2 * apothem = sqrt(3).
        # But with rotation, it's tricky.
        # Let's use a conservative check: distance > 2.0 (circumradius * 2) ensures no overlap for any rotation?
        # No, if they are rotated 30 deg relative, they can be closer.
        # The minimum distance for non-overlap of two unit hexagons is sqrt(3) (side touching) if aligned.
        # If rotated, the minimum distance can be smaller? No, the Minkowski sum is larger.
        # The "forbidden" region is the Minkowski sum.
        # To be safe and fast, let's check if distance < 1.9 (approx) -> penalty.
        # Actually, let's just use a penalty based on the minimum possible distance.
        # The minimum distance between two disjoint unit hexagons is sqrt(3) (if aligned side-to-side).
        # If we enforce distance >= sqrt(3) + eps, it might be too restrictive if rotations allow tighter packing.
        # But for simplicity, let's start with this.
        
        penalty = 0.0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                # If d < sqrt(3), they might overlap.
                # Let's use a soft penalty.
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d) ** 2
        
        return penalty

    # Optimization
    # Start with the current program's layout.
    eps = 0.01
    centers_init = []
    angles_init = []
    
    # Center
    centers_init.append([0.0, 0.0])
    angles_init.append(np.deg2rad(30.0))
    
    # 6 neighbors
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers_init.append([x, y])
        angles_init.append(np.deg2rad(30.0))
        
    # 4 outer
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers_init.append([x, y])
        angles_init.append(np.deg2rad(30.0))
        
    centers = np.array(centers_init)
    angles = np.array(angles_init)
    phi_outer = np.deg2rad(30.0)
    
    # Local search
    step_size = 0.1
    best_S = compute_S(centers, angles, phi_outer)
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_phi = phi_outer
    
    # Simple coordinate descent / random jitter
    max_iter = 500
    for i in range(max_iter):
        if time.time() > deadline - 10.0:
            break
            
        # Jitter one center
        idx = np.random.randint(len(centers))
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_centers = centers.copy()
        new_centers[idx] += [dx, dy]
        
        # Jitter one angle (optional, maybe skip for speed)
        # Jitter phi
        new_phi = phi_outer + np.random.normal(0, step_size / 10.0)
        
        S_new = compute_S(new_centers, angles, new_phi)
        overlap_penalty = check_overlap(new_centers, angles)
        
        score_new = S_new + 10.0 * overlap_penalty
        score_old = best_S + 10.0 * check_overlap(centers, angles)
        
        if score_new < score_old:
            centers = new_centers
            phi_outer = new_phi
            best_S = S_new
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_phi = phi_outer
        else:
            # Annealing? Keep simple for now.
            pass
        
        # Reduce step size
        step_size *= 0.99

    # Final check
    final_S = compute_S(best_centers, best_angles, best_phi)
    final_overlap = check_overlap(best_centers, best_angles)
    
    if final_overlap > 1e-6:
        # Fallback to initial valid layout
        final_S = 4.04
        best_centers = centers_init
        best_angles = angles_init
        best_phi = np.deg2rad(30.0)
        
    # Construct inner list
    inner = []
    for i in range(len(best_centers)):
        x, y = best_centers[i]
        ang_deg = np.rad2deg(best_angles[i])
        inner.append([x, y, ang_deg])
        
    # Return
    # Outer center is (0,0)
    # Side is final_S
    # Angle is best_phi in degrees
    return inner, (0.0, 0.0), final_S, np.rad2deg(best_phi)

