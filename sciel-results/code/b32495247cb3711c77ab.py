import numpy as np
import time
import math
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 12
    
    def hex_vertices(cx, cy, side, angle_rad):
        """
        Returns vertices of a regular hexagon.
        angle_rad=0 means flat-top (vertex at 0 deg is rightmost? No, standard math: 0 is East).
        Let's define angle 0 as 'pointy-top' or 'flat-top'?
        Standard regular hexagon with side s.
        If angle=0, vertices are at 0, 60, 120... degrees.
        Vertex 0: (s, 0).
        This is 'pointy' on the right.
        """
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_penetration(v1, v2):
        """
        Computes the minimum penetration depth between two convex polygons.
        Returns 0 if disjoint, >0 if overlapping.
        """
        min_overlap = float('inf')
        
        # Get axes from edges of v1 and v2
        axes = []
        for v in (v1, v2):
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal /= norm
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # If separated, no overlap on this axis
            if max1 < min2 or max2 < min1:
                return 0.0
            
            # Overlap amount
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def get_cost(positions, angles, lambda_pen=10.0):
        """
        Calculate total cost: Outer Side Length + Penalty for overlaps.
        """
        # 1. Calculate Outer Side Length
        all_verts = []
        for i in range(N):
            v = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Find min enclosing hexagon side
        # The width of a set of points in direction theta is max(dot) - min(dot).
        # For a regular hexagon of side S, the width in direction theta is S * sqrt(3) * |cos(theta - theta_hex)|?
        # Actually, for a regular hexagon with side S, the distance between parallel sides (apothem*2) is S*sqrt(3).
        # The width varies. The maximum width is 2*S. The minimum width is S*sqrt(3).
        # To find the min S such that the hexagon contains the points:
        # We need to find S and orientation phi such that for all 6 directions, the projection fits.
        # Let's use a fast approximation first: check 6 orientations relative to the points' principal axes.
        # Or simply: minimize over phi in [0, pi/6].
        
        def calc_side(phi):
            # Normals of the outer hexagon
            angles_norm = phi + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            # For a regular hexagon, max_width = 2 * S * cos(delta) where delta is angle between normal and apothem?
            # Actually, if we define the hexagon by its 6 normals, the width in direction n_i is w_i.
            # The side length S is related to the width in the direction of the normal to the side.
            # For a regular hexagon, the width in the direction of a normal (perpendicular to a side) is S * sqrt(3).
            # The width in the direction of a vertex (diagonal) is 2*S.
            # The function width(phi) is periodic.
            # If we align the outer hexagon such that its normals are at angles phi + k*pi/3,
            # the constraint is that the projection of the points onto these normals must be <= S*sqrt(3)/2 from center?
            # Let's use the property: S = max_width / sqrt(3) is valid ONLY if the max_width is measured along the normal to a side.
            # However, the "width" of the point cloud in direction theta is w(theta).
            # The minimal S is min_phi max_theta (w(theta) / f(theta, phi))?
            # Simpler: The condition for containment is that for each of the 6 half-planes defined by the hexagon, the points must be inside.
            # Distance from center to side is apothem a = S * sqrt(3) / 2.
            # So for a given center C and orientation phi, we need:
            # max_i dot(P_i - C, n_i) <= a  and  min_i dot(P_i - C, n_i) >= -a
            # This implies max_i dot(P_i, n_i) - min_i dot(P_i, n_i) <= 2a = S*sqrt(3).
            # So S = (max_i dot(P_i, n_i) - min_i dot(P_i, n_i)) / sqrt(3) for the direction that gives the max width.
            # Wait, this formula S = max_width / sqrt(3) assumes the width is measured perpendicular to a side.
            # If we rotate the hexagon, the "width" of the point cloud in the 6 normal directions changes.
            # We want to minimize S(phi) = max_{k=0..5} ( dot(P, n_k(phi)).max - dot(P, n_k(phi)).min ) / sqrt(3).
            
            return max_width / np.sqrt(3)
        
        # Optimize phi in [0, pi/6] due to symmetry
        # Using a few samples is fast, minimize_scalar is slower but accurate.
        # Since we call this in SA, we need it fast.
        # Let's sample 12 angles.
        min_side = float('inf')
        for phi in np.linspace(0, np.pi/6, 13):
            s = calc_side(phi)
            if s < min_side:
                min_side = s
        
        # 2. Calculate Overlap Penalty
        penalty = 0.0
        for i in range(N):
            vi = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, N):
                vj = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                pen = sat_penetration(vi, vj)
                if pen > 0:
                    penalty += pen * pen
        
        return min_side + lambda_pen * penalty

    def is_valid(positions, angles, tol=1e-5):
        for i in range(N):
            vi = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, N):
                vj = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                if sat_penetration(vi, vj) > tol:
                    return False
        return True

    def get_final_side(positions, angles):
        """Precise calculation of the minimum outer side length."""
        all_verts = []
        for i in range(N):
            v = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        def calc_side_precise(phi):
            angles_norm = phi + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            return max_width / np.sqrt(3)
        
        res = minimize_scalar(calc_side_precise, bounds=(0, np.pi/6), method='bounded', options={'xatol': 1e-5})
        return res.fun, res.x

    # --- Initialization: Honeycomb ---
    # 1 center, 6 in ring 1, 5 in ring 2 (gaps)
    # Distance between centers of touching hexagons is 2.0
    d = 2.0
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    positions[0] = [0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        ang = np.radians(60 * k)
        positions[k+1] = [d * np.cos(ang), d * np.sin(ang)]
        
    # Ring 2: 5 hexagons in the "valleys" of ring 1
    # Distance from center to ring 2 centers: d * sqrt(3) ?
    # In a hex grid, neighbors of (0,0) are at dist 2.
    # Next neighbors (corners of the hexagon formed by ring 1) are at dist 2*sqrt(3).
    # Let's place them at angles 30, 90, 150, 210, 270.
    dist2 = d * np.sqrt(3)
    gap_angles = [30, 90, 150, 210, 270]
    for k in range(5):
        ang = np.radians(gap_angles[k])
        positions[k+7] = [dist2 * np.cos(ang), dist2 * np.sin(ang)]

    # Verify initial validity
    if not is_valid(positions, angles):
        # If not valid, spread them out slightly
        d = 2.01
        positions[0] = [0.0, 0.0]
        for k in range(6):
            ang = np.radians(60 * k)
            positions[k+1] = [d * np.cos(ang), d * np.sin(ang)]
        dist2 = d * np.sqrt(3)
        for k in range(5):
            ang = np.radians(gap_angles[k])
            positions[k+7] = [dist2 * np.cos(ang), dist2 * np.sin(ang)]
            
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_cost = get_cost(best_positions, best_angles)
    best_valid_side = float('inf')
    best_valid_pos = best_positions.copy()
    best_valid_ang = best_angles.copy()
    
    # Check if initial is valid
    if is_valid(best_positions, best_angles):
        side, _ = get_final_side(best_positions, best_angles)
        best_valid_side = side
        best_valid_pos = best_positions.copy()
        best_valid_ang = best_angles.copy()
        best_cost = side # Cost is just side if valid (penalty 0)

    # --- Simulated Annealing ---
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    step_size = 0.2
    rng = np.random.default_rng(42)
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_cost = best_cost
    
    max_iter = 20000
    iter_count = 0
    
    while T > T_min:
        if time.time() > deadline - 10:
            break
            
        # Select a random hexagon to move
        i = rng.integers(N)
        
        # Random perturbation
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        dtheta = rng.normal(0, 0.1) # Small angle change
        
        old_x, old_y = current_positions[i]
        old_theta = current_angles[i]
        
        current_positions[i, 0] += dx
        current_positions[i, 1] += dy
        current_angles[i] += dtheta
        
        new_cost = get_cost(current_positions, current_angles)
        
        # Acceptance
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
                
                # Check validity periodically to update best valid side
                if iter_count % 100 == 0:
                    if is_valid(best_positions, best_angles):
                        side, _ = get_final_side(best_positions, best_angles)
                        if side < best_valid_side:
                            best_valid_side = side
                            best_valid_pos = best_positions.copy()
                            best_valid_ang = best_angles.copy()
        else:
            # Reject
            current_positions[i, 0] = old_x
            current_positions[i, 1] = old_y
            current_angles[i] = old_theta
            
        T *= cooling
        iter_count += 1

    # --- Final Refinement ---
    # Use the best valid solution found
    if best_valid_side < float('inf'):
        final_pos = best_valid_pos
        final_ang = best_valid_ang
    else:
        # Fallback: try to fix the best cost solution if it's close to valid
        final_pos = best_positions
        final_ang = best_angles
        if not is_valid(final_pos, final_ang):
             # Last resort: expand slightly
             # This might result in a larger side, but valid
             for i in range(N):
                 if i == 0: continue
                 # Move away from center
                 vec = final_pos[i]
                 norm = np.linalg.norm(vec)
                 if norm > 1e-6:
                     final_pos[i] += 0.1 * (vec / norm)
        
        if not is_valid(final_pos, final_ang):
            # If still invalid, use the initial honeycomb which we know is valid (or nearly)
            # Re-initialize to safe honeycomb
            d = 2.01
            final_pos = np.zeros((N, 2))
            final_ang = np.zeros(N)
            final_pos[0] = [0.0, 0.0]
            for k in range(6):
                ang = np.radians(60 * k)
                final_pos[k+1] = [d * np.cos(ang), d * np.sin(ang)]
            dist2 = d * np.sqrt(3)
            for k in range(5):
                ang = np.radians(gap_angles[k])
                final_pos[k+7] = [dist2 * np.cos(ang), dist2 * np.sin(ang)]
            if not is_valid(final_pos, final_ang):
                # Increase d until valid
                d = 2.0
                while not is_valid(final_pos, final_ang) and d < 3.0:
                    d += 0.01
                    final_pos = np.zeros((N, 2))
                    final_pos[0] = [0.0, 0.0]
                    for k in range(6):
                        ang = np.radians(60 * k)
                        final_pos[k+1] = [d * np.cos(ang), d * np.sin(ang)]
                    dist2 = d * np.sqrt(3)
                    for k in range(5):
                        ang = np.radians(gap_angles[k])
                        final_pos[k+7] = [dist2 * np.cos(ang), dist2 * np.sin(ang)]

    # Calculate final outer side and center
    outer_side, outer_angle_rad = get_final_side(final_pos, final_ang)
    
    # Calculate Center
    # The center of the minimal enclosing hexagon is the average of the midpoints of the projections?
    # Or simply, the centroid of the points is a good approximation, but we want the center of the hexagon.
    # The hexagon is defined by 6 half-planes.
    # Let's find the center that minimizes the required side length?
    # Actually, for a given orientation, the center that minimizes the max distance to the sides is the midpoint of the min/max projections.
    
    all_verts = []
    for i in range(N):
        v = hex_vertices(final_pos[i, 0], final_pos[i, 1], 1.0, final_ang[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    angles_norm = outer_angle_rad + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
    dots = all_verts @ normals.T
    mins = dots.min(axis=0)
    maxs = dots.max(axis=0)
    mids = (mins + maxs) / 2.0
    
    # Solve for center C such that dot(C, n_i) = mid_i
    # C = (normals.T)^+ mids
    c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
    outer_center = tuple(c)
    outer_angle_deg = np.degrees(outer_angle_rad)
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_ang[i])] for i in range(N)]
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
