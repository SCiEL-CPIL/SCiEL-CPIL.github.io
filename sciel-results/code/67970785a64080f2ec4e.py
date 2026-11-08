"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin for 120s limit
    
    # Hexagon geometry constants
    # Unit hexagon: side 1.
    # Distance between parallel sides (width) = sqrt(3)
    # Distance between opposite vertices (diameter) = 2
    # To ensure disjointness of two unit hexagons with same orientation,
    # their centers must be at least 2 * inradius = sqrt(3) apart?
    # No. If aligned, the distance between centers must be at least 
    # 2 * (distance from center to edge) = 2 * (sqrt(3)/2) = sqrt(3) 
    # ONLY if the line connecting centers is perpendicular to the edges.
    # In the worst case (rotated 30 deg), the "width" is 2.
    # Actually, for two identical regular hexagons to be disjoint,
    # the minimum distance between centers is sqrt(3) if they are aligned 
    # such that a flat side faces the other. 
    # Wait, the inradius is sqrt(3)/2. If they touch side-to-side, 
    # distance is 2 * sqrt(3)/2 = sqrt(3).
    # If they touch vertex-to-vertex, distance is 2.
    # The minimum distance for disjointness is actually sqrt(3) 
    # (when aligned). If rotated, the required distance increases up to 2.
    # To be safe and simple, we assume all inner hexagons have the SAME 
    # orientation (angle 0) as the outer one. This simplifies the 
    # repulsion logic to a fixed minimum center-distance of sqrt(3).
    # If we allowed rotation, we'd need to check vertices, which is expensive.
    # Given N=11 is small, we might try a few rotations later, but let's 
    # start with aligned hexagons.
    
    MIN_CENTER_DIST = np.sqrt(3)
    
    # Objective: Minimize outer side length S.
    # Constraint: All inner hexagons (side 1) must be inside outer hexagon (side S).
    # And inner hexagons must be disjoint.
    
    # For a regular hexagon of side S centered at origin:
    # The distance from center to any edge is S * sqrt(3)/2.
    # A point (x,y) is inside if:
    # |x| <= S
    # |x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2  (approx, depends on orientation)
    # Actually, the condition for a point (x,y) to be inside a regular hexagon 
    # of side S, flat-top orientation (vertices at 0, 60, ...):
    # The hexagon is defined by |x| <= S, |y| <= S*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= S
    # Wait, let's use the standard "flat-top" hexagon where the top and bottom 
    # are flat.
    # Vertices at: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Inequalities:
    # 1. |x| <= S
    # 2. |x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2  --> |x + y*sqrt(3)| <= S * sqrt(3)
    # 3. |-x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2 --> |-x + y*sqrt(3)| <= S * sqrt(3)
    
    # A unit hexagon (side 1) centered at (cx, cy) with same orientation 
    # fits inside outer hexagon (side S) centered at (0,0) if:
    # The "expanded" unit hexagon (Minkowski sum with outer boundary offset)
    # Actually, simpler: The unit hexagon must be contained.
    # This is equivalent to the center (cx, cy) being inside a "shrunk" 
    # outer hexagon. The shrinkage is the inradius of the unit hexagon 
    # along the normal of each edge.
    # Since orientations are aligned, the unit hexagon has inradius r_in = sqrt(3)/2.
    # The outer hexagon has inradius R_in = S * sqrt(3)/2.
    # The center (cx, cy) must be at distance >= r_in from each edge of the outer hexagon.
    # This is equivalent to (cx, cy) being inside a hexagon of side S - 1?
    # Let's check. If we shrink a hexagon of side S by distance d from each edge,
    # the new side length is S - 2*d / sqrt(3)?
    # Inradius scales linearly with side length.
    # New inradius = S*sqrt(3)/2 - sqrt(3)/2 = (S-1)*sqrt(3)/2.
    # So yes, the centers must lie within a regular hexagon of side S-1.
    
    # So the problem reduces to:
    # Place N points (centers) such that:
    # 1. Pairwise distance >= sqrt(3)
    # 2. All points lie inside a regular hexagon of side L = S - 1.
    # Minimize S = L + 1. So minimize L.
    
    # We will search for L.
    # Start with a large L and shrink it, using simulated annealing / repulsion.
    
    def is_inside_hexagon(points, L):
        # points: (N, 2) array
        # Hexagon of side L, flat-top, centered at 0.
        # Inequalities for point (x,y):
        # |x| <= L
        # |x + y*sqrt(3)| <= L * sqrt(3)
        # |-x + y*sqrt(3)| <= L * sqrt(3)
        sqrt3 = np.sqrt(3)
        abs_x = np.abs(points[:, 0])
        abs_x_plus_y_sqrt3 = np.abs(points[:, 0] + points[:, 1] * sqrt3)
        abs_x_minus_y_sqrt3 = np.abs(points[:, 0] - points[:, 1] * sqrt3)
        
        return np.all(abs_x <= L) and np.all(abs_x_plus_y_sqrt3 <= L * sqrt3) and np.all(abs_x_minus_y_sqrt3 <= L * sqrt3)

    def compute_violations(points, L):
        # Returns a vector of forces/penalties.
        # 1. Boundary violation: Push points inward if outside.
        # 2. Collision violation: Push points apart if too close.
        
        N_pts = len(points)
        forces = np.zeros_like(points)
        
        # Boundary forces
        sqrt3 = np.sqrt(3)
        for i in range(N_pts):
            x, y = points[i]
            # Check 3 constraints
            # 1. x > L or x < -L
            if x > L:
                forces[i, 0] -= (x - L)
            elif x < -L:
                forces[i, 0] += (L - x)
            
            # 2. x + y*sqrt3 > L*sqrt3 or < -L*sqrt3
            val = x + y * sqrt3
            limit = L * sqrt3
            if val > limit:
                # Gradient of (x + y*sqrt3) is (1, sqrt3). Normalize? 
                # We just want to push back.
                forces[i, 0] -= (val - limit)
                forces[i, 1] -= (val - limit) * sqrt3
            elif val < -limit:
                forces[i, 0] += (limit - val)
                forces[i, 1] += (limit - val) * sqrt3
                
            # 3. -x + y*sqrt3 > L*sqrt3 or < -L*sqrt3
            val2 = -x + y * sqrt3
            if val2 > limit:
                forces[i, 0] += (val2 - limit)
                forces[i, 1] += (val2 - limit) * sqrt3
            elif val2 < -limit:
                forces[i, 0] -= (limit - val2)
                forces[i, 1] -= (limit - val2) * sqrt3

        # Collision forces
        for i in range(N_pts):
            for j in range(i + 1, N_pts):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < MIN_CENTER_DIST**2:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-9 else 1e-9
                    # Push apart
                    overlap = MIN_CENTER_DIST - dist
                    if dist < 1e-9:
                        # Random push
                        angle = np.random.rand() * 2 * np.pi
                        fx = np.cos(angle) * overlap
                        fy = np.sin(angle) * overlap
                    else:
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    
        return forces

    # Initialize points
    # Start with a hexagonal lattice pattern
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1 (6 hexagons)
    for k in range(6):
        angle = k * np.pi / 3
        # Distance for lattice is 2? No, for unit hexagons side-by-side, 
        # center distance is 2? 
        # Wait, inradius is sqrt(3)/2. If they touch side to side, 
        # distance is 2 * inradius = sqrt(3).
        # But in a hexagonal lattice of hexagons, the distance between 
        # centers of adjacent hexagons is 2 * side_length? 
        # No. A regular hexagon of side 1 has distance between opposite 
        # vertices 2. Distance between opposite sides sqrt(3).
        # If you pack them in a honeycomb, the centers form a triangular 
        # lattice. The distance between nearest neighbors is 2 * inradius = sqrt(3).
        # Let's verify. 
        # Hexagon 1 at (0,0). 
        # Hexagon 2 to the right. Its left side touches Hex 1's right side.
        # Hex 1 right side is at x = sqrt(3)/2.
        # Hex 2 left side is at x = cx2 - sqrt(3)/2.
        # So cx2 - sqrt(3)/2 = sqrt(3)/2 => cx2 = sqrt(3).
        # So yes, distance is sqrt(3).
        
        # However, if we use a triangular lattice with spacing sqrt(3), 
        # do they overlap?
        # Distance sqrt(3) is the minimum. So they touch.
        # So initial positions:
        r = np.sqrt(3)
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 2 (need 4 more to reach 11)
    # Total 11. We have 7. Need 4 more.
    # Place them in the "corners" of the hexagonal cluster?
    # Or just scatter them randomly nearby to let the optimizer fix it.
    # Better: Add them to the next ring positions.
    # Ring 2 has 12 positions. We pick 4.
    # To minimize area, we should probably fill the cluster compactly.
    # A cluster of 19 hexagons is a hexagon of side 2 (in terms of hex count).
    # We only have 11.
    # Let's just add 4 random points in a circle of radius 2*sqrt(3) and 
    # let the optimizer move them in.
    for _ in range(4):
        angle = np.random.rand() * 2 * np.pi
        radius = 2.5 * np.sqrt(3)
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
    points = np.array(points)
    
    # Annealing loop
    # We want to find the minimum L.
    # Strategy: 
    # 1. Start with a large L.
    # 2. Run optimization to minimize boundary violations and collisions.
    # 3. If valid, shrink L.
    # 4. If invalid, increase L slightly and re-optimize.
    
    # Or: 
    # Simulated Annealing on L?
    # Let's try: 
    # Start L = 5.0 (generous).
    # Optimize positions for this L.
    # Check if valid.
    # If valid, try smaller L.
    
    best_L = None
    best_points = None
    
    # Initial guess for L
    # Area of 11 unit hexagons = 11 * (3*sqrt(3)/2) * 1^2 ≈ 11 * 2.598 ≈ 28.58
    # Area of outer hexagon side L is (3*sqrt(3)/2) * L^2
    # So L^2 >= 28.58 / 2.598 = 11.
    # L >= sqrt(11) ≈ 3.31.
    # So S = L + 1 >= 4.31.
    # Let's start L at 5.0.
    
    current_L = 5.0
    temp = 1.0
    learning_rate = 0.1
    
    # We will iterate.
    # For each step:
    # 1. Compute forces for current L.
    # 2. Update points.
    # 3. Check validity.
    # 4. If valid, try to decrease L.
    
    # To be efficient, we'll do a "shrink and solve" loop.
    
    # First, stabilize positions for a large L.
    for step in range(2000):
        if time.time() > deadline:
            break
        forces = compute_violations(points, current_L)
        # Add noise
        noise = np.random.randn(*points.shape) * temp
        points += learning_rate * (forces + noise)
        temp *= 0.999
        
        # Check validity
        if is_inside_hexagon(points, current_L):
            # Check collisions
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d2 = np.sum((points[i]-points[j])**2)
                    if d2 < MIN_CENTER_DIST**2:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                if best_L is None or current_L < best_L:
                    best_L = current_L
                    best_points = points.copy()
                
                # Try to shrink L
                # How much?
                # If we shrink too much, we lose the valid state.
                # Let's shrink by 0.01
                current_L -= 0.01
                # Reset learning rate slightly?
                learning_rate = 0.05
                temp = 0.1
            else:
                # Invalid, increase L
                current_L += 0.02
                learning_rate = 0.1
                temp = 0.5
        else:
            # Outside boundary, increase L
            current_L += 0.05
            learning_rate = 0.1
            temp = 1.0

    # If we found a best_L, use it.
    # Otherwise, use the last valid state or a default.
    
    if best_L is not None:
        L_final = best_L
        pts_final = best_points
    else:
        # Fallback: Return a valid, albeit non-optimal, configuration
        # Just place them in a line.
        # Distance sqrt(3) apart.
        pts_final = np.array([[k * np.sqrt(3), 0.0] for k in range(N)])
        # Calculate required L for this line
        # Max x = (N-1)*sqrt(3) = 10*sqrt(3) ≈ 17.32
        # L must be at least max(|x|, ...)
        # For a line along x-axis, the hexagon must cover x from -17.32 to 17.32?
        # No, centered at 0. So x from -8.66 to 8.66?
        # Wait, k from 0 to 10. Center at 5*sqrt(3).
        # Let's center the line.
        pts_final = np.array([(k - 5) * np.sqrt(3) for k in range(N)]).reshape(-1, 1)
        pts_final = np.hstack([pts_final, np.zeros((N, 1))])
        # Max x = 5*sqrt(3) ≈ 8.66
        # L must be >= 8.66?
        # Check constraint |x| <= L. So L >= 8.66.
        # Check other constraints.
        L_final = 9.0
        
    S_final = L_final + 1.0
    
    # Construct return value
    # inner: list of [x, y, angle_degrees]
    # center: [0, 0]
    # side: S_final
    # angle: 0.0
    
    inner = []
    for p in pts_final:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(S_final), 0.0
# EVOLVE-BLOCK-END
