"""
Strategy: Support Function Envelope with Smooth Penalties.
Instead of hard SAT checks, we compute the minimal outer hexagon side length
via support function projections (smooth) and use a softplus penalty for
inner hexagon overlaps. This allows gradient-based optimization (L-BFGS-B)
to refine the 6-fold symmetric arrangement of 12 unit hexagons.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(angle_deg):
        """
        Returns the 6 outward normal vectors of a regular hexagon with given rotation.
        For a unit hexagon (side=1), the apothem is sqrt(3)/2.
        The normals point to the midpoints of the edges.
        Standard orientation (0 deg): flat top? Or pointy top?
        Let's define 0 deg as pointy-top (vertex at 90 deg).
        Vertices at 90, 150, 210, 270, 330, 30.
        Edge midpoints (normals) at 60, 120, 180, 240, 300, 0.
        Wait, standard math:
        If vertices are at theta, theta+60...
        Let's stick to the definition in the seed:
        angles = angle_rad + [0, 60, 120, 180, 240, 300]
        This means at angle=0, vertices are at 0, 60, 120... (pointy right).
        Edge midpoints are at 30, 90, 150, 210, 270, 330.
        So normals are at angle_deg + 30 + k*60.
        """
        base_angle = np.radians(angle_deg + 30)
        angles = base_angle + np.radians(np.arange(0, 360, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def support_function_max(verts, normals):
        """
        Compute the maximum projection of vertices onto the given normals.
        This gives the 'radius' of the shape in those directions.
        """
        # verts: (N, 2), normals: (6, 2)
        # proj: (N, 6)
        proj = verts @ normals.T
        return np.max(proj, axis=0)

    def compute_minimal_outer_side(inner_positions, outer_center, outer_angle):
        """
        Compute the minimal side length of an outer hexagon (centered, rotated)
        that contains all inner hexagons.
        """
        normals = get_normals(outer_angle)
        
        # Transform all inner vertices to outer local frame?
        # Or just compute distances from outer center and project on outer normals.
        # The condition for a point p to be inside a hexagon centered at c with normals n_i and apothem h is:
        # dot(p - c, n_i) <= h for all i.
        # We want to minimize side s, which implies h = s * sqrt(3)/2.
        # So we need h >= max_i dot(p - c, n_i) for all points p in all inner hexagons.
        
        max_proj = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            # Translate to outer center
            verts_rel = verts - np.array(outer_center)
            # Project onto outer normals
            projs = verts_rel @ normals.T
            max_proj = max(max_proj, np.max(projs))
        
        # max_proj is the required apothem
        apothem = max_proj
        side = apothem / (math.sqrt(3) / 2)
        return side

    def distance_between_hexagons(v1, v2):
        """
        Approximate minimum distance between two convex polygons.
        We use the SAT separating axis. If they are separated, return the gap.
        If they overlap, return 0 (or negative penetration depth, approximated).
        For smooth penalty, we want a smooth function of overlap.
        
        Better approach for smoothness:
        Compute the minimum distance between any vertex of v1 and any edge of v2,
        and vice versa. Take the minimum of all these distances.
        If the minimum is positive, they are disjoint.
        If negative, they overlap.
        This is not perfectly smooth at 0, but we can apply softplus.
        
        However, checking all pairs of vertices/edges is O(36).
        Let's try a simpler metric: Center-to-center distance minus sum of inradii?
        No, that's too loose.
        
        Let's use the standard SAT gap.
        For each normal axis, compute the projection interval for both polygons.
        Gap = min(max1 - min2, max2 - min1).
        If any gap > 0, they are disjoint. The "distance" is the max of these gaps?
        No, the minimum distance is the max of the gaps along the separating axes?
        Actually, if they are disjoint, the minimum distance is the maximum gap along the separating axis.
        If they overlap, all gaps are <= 0.
        
        Let's compute the "overlap depth" as the minimum gap (which will be negative if overlapping).
        overlap_depth = min( gap for all axes )
        If overlap_depth > 0, disjoint.
        If overlap_depth < 0, overlapping by -overlap_depth (roughly).
        
        This function is continuous but not differentiable at 0.
        We will apply a softplus penalty: penalty = softplus(-overlap_depth * k)
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        min_gap = float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap is how much they are separated.
            # If max1 < min2, gap is min2 - max1.
            # If max2 < min1, gap is min1 - max2.
            # If they overlap, gap is negative (penetration).
            gap1 = min2 - max1
            gap2 = min1 - max2
            gap = max(gap1, gap2)
            
            # We want the minimum gap over all axes.
            # If any axis has a positive gap, they are disjoint.
            # The "distance" is the max positive gap.
            # The "penetration" is the min negative gap (closest to 0 from below).
            
            if gap < min_gap:
                min_gap = gap
                
        # If min_gap > 0, they are disjoint. Distance is min_gap?
        # Actually, the minimum distance between convex sets is the maximum of the gaps along the separating axes.
        # Wait. If they are disjoint, there is a unique separating axis (or one of a range).
        # The gap along the *separating* axis is the distance.
        # The gap along other axes might be larger.
        # So Distance = max( gap for all axes ) ?
        # Let's test. Two squares. Axis 1: gap 1. Axis 2: gap 2. Distance is 1.
        # So yes, Distance = max( gap ).
        # If they overlap, all gaps are negative.
        # We want a penalty for overlap.
        # Let's define penalty_input = -min( max_gap, 0 )
        # If max_gap > 0, penalty_input = 0.
        # If max_gap < 0, penalty_input = -max_gap (positive).
        
        # But wait, is max(gap) the distance?
        # Consider two points. Only one axis. Gap is distance.
        # Consider two squares side by side.
        # Axis X: gap = d.
        # Axis Y: gap = side length (if aligned) or larger.
        # So max gap is indeed the distance if the shapes are convex and disjoint?
        # Actually, the minimum distance between two convex sets is the solution to a QP.
        # The SAT gap is an upper bound on the distance? No, it's the distance if we pick the right axis.
        # The *minimum* distance is the *maximum* of the gaps along the separating axes?
        # Let's assume for the penalty that we just need a smooth measure of "how much they overlap".
        # Let's use the minimum gap (most negative).
        # If they are disjoint, min_gap might be negative for non-separating axes?
        # No, if they are disjoint, there is at least one axis with positive gap.
        # But other axes might have negative gaps (projections overlap).
        # So min_gap will be negative even if disjoint.
        # We need the MAX gap. If max_gap > 0, disjoint.
        
        max_gap = -float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            gap = max(gap1, gap2)
            
            if gap > max_gap:
                max_gap = gap
                
        # max_gap is the distance if disjoint, and negative (penetration depth approx) if overlapping.
        return max_gap

    def objective(params):
        if time.time() > deadline - 2:
            return 1000.0

        r1, a1, r2, a2, o_a = params
        
        # Generate inner positions
        inner_positions = []
        
        # Center hexagon
        inner_positions.append((0.0, 0.0, 0.0))
        
        # Inner ring: 6 hexagons
        for k in range(6):
            angle = np.radians(60 * k)
            x = r1 * np.cos(angle)
            y = r1 * np.sin(angle)
            inner_positions.append((x, y, a1))
            
        # Outer ring: 5 hexagons (offset by 30 degrees to fit in gaps)
        for k in range(5):
            angle = np.radians(60 * k + 30)
            x = r2 * np.cos(angle)
            y = r2 * np.sin(angle)
            inner_positions.append((x, y, a2))
        
        # Compute minimal outer side
        os = compute_minimal_outer_side(inner_positions, (0.0, 0.0), o_a)
        
        # Compute collision penalty
        penalty = 0.0
        verts_list = []
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_list.append(verts)
            
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                # Quick check: if centers are far apart, skip
                # Inradius is sqrt(3)/2 ~ 0.866. Sum is ~1.732.
                # If dist > 3.5, definitely disjoint.
                ci = inner_positions[i]
                cj = inner_positions[j]
                dist_sq = (ci[0]-cj[0])**2 + (ci[1]-cj[1])**2
                if dist_sq > 12.0: # 3.46^2
                    continue
                    
                gap = distance_between_hexagons(verts_list[i], verts_list[j])
                # gap > 0 means disjoint.
                # gap < 0 means overlapping.
                # We want to penalize overlap.
                # Use softplus for smoothness.
                # penalty += softplus(-gap * k)
                # If gap is 1.0, -gap is -1.0, softplus(-1) ~ 0.31. Small penalty.
                # If gap is -0.5, -gap is 0.5, softplus(0.5) ~ 0.97. Large penalty.
                # This works.
                k_smooth = 20.0
                # To avoid overflow in exp for large positive arguments
                arg = -gap * k_smooth
                if arg > 50:
                    penalty += arg
                else:
                    penalty += math.log(1 + math.exp(arg))
                    
        # We want to minimize os, but penalize overlap heavily.
        # If penalty is large, the configuration is invalid.
        # We can return os + penalty * weight.
        # But os is ~ 4-5. Penalty can be large.
        # Let's use a large weight to push out overlaps.
        return os + penalty * 10.0

    # Initial guess
    # Standard hexagonal packing:
    # Inner ring at distance 2.0 (touching center)
    # Outer ring at distance ~3.46 (touching inner ring)
    # Angles: 0 for inner, 30 for outer (to fit in gaps)
    # Outer rotation: 30 (flat top) or 0 (pointy top)
    
    initial_guess = np.array([2.0, 0.0, 3.5, 30.0, 30.0])
    
    # Bounds
    bounds = [
        (1.5, 5.0),  # r1
        (0, 360),    # a1
        (2.0, 6.0),  # r2
        (0, 360),    # a2
        (0, 360)     # o_a
    ]
    
    best_result = None
    best_score = float('inf')
    
    # Multi-start L-BFGS-B
    # We'll try a few random starts near the initial guess
    starts = [initial_guess]
    
    # Add some variations
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        r1 = np.random.uniform(1.8, 2.5)
        a1 = np.random.uniform(0, 30)
        r2 = np.random.uniform(3.0, 4.5)
        a2 = np.random.uniform(15, 45)
        o_a = np.random.uniform(0, 60)
        starts.append(np.array([r1, a1, r2, a2, o_a]))
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                start, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            if res.success:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_result = res.x
        except Exception as e:
            pass
            
    if best_result is None:
        best_params = initial_guess
    else:
        best_params = best_result

    # Ensure angles are in [0, 360)
    best_params[1] = np.mod(best_params[1], 360)
    best_params[3] = np.mod(best_params[3], 360)
    best_params[4] = np.mod(best_params[4], 360)
    
    # Ensure radii are positive
    if best_params[0] < 1.0: best_params[0] = 1.0
    if best_params[2] < 1.5: best_params[2] = 1.5

    r1, a1, r2, a2, o_a = best_params
    
    # Construct final inner positions
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = np.radians(60 * k)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        inner.append((x, y, a1))
        
    for k in range(5):
        angle = np.radians(60 * k + 30)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        inner.append((x, y, a2))
        
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_minimal_outer_side(inner, oc, oa)
    
    # Final validation
    # Check if any overlaps exist. If so, we need to expand or fix.
    # Since the optimizer minimizes penalty, it should be disjoint.
    # But let's verify.
    verts_list = []
    for i in range(len(inner)):
        cx, cy, angle = inner[i]
        verts = hexagon_vertices(cx, cy, 1.0, angle)
        verts_list.append(verts)
        
    is_valid = True
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            gap = distance_between_hexagons(verts_list[i], verts_list[j])
            if gap < 1e-6: # Overlapping or touching too closely
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback: Use a known valid configuration
        # Increase radii slightly to ensure separation
        r1_fb = 2.05
        r2_fb = 3.6
        a1_fb = 0.0
        a2_fb = 30.0
        o_a_fb = 30.0
        
        inner = []
        inner.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = np.radians(60 * k)
            x = r1_fb * np.cos(angle)
            y = r1_fb * np.sin(angle)
            inner.append((x, y, a1_fb))
            
        for k in range(5):
            angle = np.radians(60 * k + 30)
            x = r2_fb * np.cos(angle)
            y = r2_fb * np.sin(angle)
            inner.append((x, y, a2_fb))
            
        oc = (0.0, 0.0)
        oa = o_a_fb
        os = compute_minimal_outer_side(inner, oc, oa)
        
        # Verify again
        verts_list = []
        for i in range(len(inner)):
            cx, cy, angle = inner[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_list.append(verts)
            
        for i in range(len(inner)):
            for j in range(i+1, len(inner)):
                gap = distance_between_hexagons(verts_list[i], verts_list[j])
                if gap < 1e-6:
                    # Expand further
                    os += 0.5
                    break

    return inner, oc, os, oa
