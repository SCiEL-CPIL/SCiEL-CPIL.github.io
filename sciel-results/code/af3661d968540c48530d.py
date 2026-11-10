# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses linear programming with the Minkowski difference (dodecagon) for non-overlap
    and containment constraints.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    def elapsed():
        return time.time() - start_time

    def hex_vertices(cx, cy, d, angle_deg):
        """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a (degrees)."""
        a = np.radians(angle_deg)
        angles = a + np.arange(6) * np.pi / 3.0
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    # For unit hexagons, the Minkowski difference H1 - H2 is a dodecagon.
    # Two unit hexagons (with same orientation) are non-overlapping iff
    # their center difference is outside the interior of the Minkowski difference.
    # The Minkowski difference of two unit hexagons with same orientation is a regular
    # dodecagon with circumradius 2 (i.e., the hexagon side * 2).
    #
    # Actually, let me think more carefully. Two convex polygons P and Q are disjoint iff
    # there exists a separating axis. For two identical regular hexagons with the same
    # orientation, the Minkowski difference P - Q is a centrally symmetric convex polygon.
    # The two hexagons are disjoint iff their center difference is NOT in the interior of P - Q.
    #
    # The Minkowski difference of two unit regular hexagons (same orientation) is a dodecagon.
    # Its support function in direction u is h_H(u) + h_H(-u) = 2*h_H(u) where h_H is the
    # support function of the unit hexagon.
    #
    # For a unit regular hexagon with flat-top orientation (angle 0), the support function
    # in direction (cos θ, sin θ) is:
    # h(θ) = max over vertices of dot(vertex, direction)
    #
    # The 6 vertices of a unit hexagon (angle 0, centered at origin) are:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    #
    # The support function has 6 linear pieces. The Minkowski difference has 12 linear pieces
    # (since it's centrally symmetric, 6 unique directions).
    #
    # The key insight: two unit hexagons with the SAME orientation are disjoint iff for ALL
    # 6 edge normals of the hexagon, the projection of the center difference onto that normal
    # has magnitude >= the sum of the half-widths... no, that's not quite right.
    #
    # Let me use the SAT (Separating Axis Theorem) approach directly.
    # Two convex polygons are disjoint iff there exists a separating axis (edge normal of
    # either polygon) such that their projections on that axis are disjoint.
    #
    # For two identical unit hexagons with the same orientation, the 12 edge normals reduce
    # to 6 unique directions (since opposite edges have opposite normals, and the hexagon is
    # centrally symmetric). Actually, a hexagon has 6 edges, so 6 edge normals. But since
    # both hexagons have the same orientation, we only need to check 6 axes.
    #
    # For a unit regular hexagon with angle 0 (flat-top), the edge normals point in directions:
    # 0°, 60°, 120°, 180°, 240°, 300° — wait, let me compute properly.
    #
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # Edge from vertex i to vertex i+1 has normal pointing outward.
    #
    # Actually, for a regular hexagon with vertices at angles 0, 60, ..., 300,
    # the outward normals of the edges point at angles 30, 90, 150, 210, 270, 330 degrees.
    #
    # The width of a unit hexagon in direction n (unit vector) is the difference between
    # max and min of dot(v, n) over vertices v. For a regular hexagon, this equals 2 * apothem
    # if n is perpendicular to an edge, or 2 * circumradius * cos(30°) if n points to a vertex.
    #
    # For a unit regular hexagon: circumradius = 1, apothem = sqrt(3)/2.
    # Width in edge-normal direction = 2 * apothem = sqrt(3) ≈ 1.732
    # Width in vertex direction = 2 * 1 = 2
    #
    # Two hexagons are disjoint iff for some edge normal direction n of the hexagon:
    # |dot(c1 - c2, n)| >= width_in_direction_n
    #
    # Wait, that's not right either. The SAT says: two convex bodies A and B are disjoint iff
    # there exists a direction n such that max_{a in A} dot(a,n) < min_{b in B} dot(b,n)
    # or vice versa.
    #
    # For two unit hexagons with centers c1 and c2 and same orientation:
    # max_{a in H1} dot(a,n) = dot(c1, n) + h_H(n)
    # min_{b in H2} dot(b,n) = dot(c2, n) - h_H(n)
    #
    # They are disjoint in direction n iff:
    # dot(c1, n) + h_H(n) < dot(c2, n) - h_H(n)  OR  dot(c2, n) + h_H(n) < dot(c1, n) - h_H(n)
    # i.e., |dot(c1 - c2, n)| > 2 * h_H(n)
    #
    # So the non-overlap constraint is: for AT LEAST ONE of the 6 edge normal directions,
    # |dot(c_i - c_j, n_k)| >= 2 * h_H(n_k)
    #
    # This is a disjunctive constraint, which makes LP harder. However, we can use the fact
    # that the Minkowski difference H - H is a convex polygon, and two hexagons overlap iff
    # c_i - c_j is in the interior of H - H.
    #
    # H - H for a unit hexagon is a dodecagon. The constraint "c_i - c_j is NOT in the interior
    # of the dodecagon" is equivalent to: for at least one of the 12 edge normals of the dodecagon,
    # dot(c_i - c_j, n) >= h_dodecagon(n) or dot(c_i - c_j, n) <= -h_dodecagon(n).
    #
    # This is still disjunctive. For LP, we'd need to enumerate which separating axis is active
    # for each pair, which is combinatorial.
    #
    # Alternative approach: Use a different formulation. Since we want to minimize s, and the
    # problem is small (12 hexagons), we can try a heuristic approach:
    #
    # 1. Start with a known good packing configuration
    # 2. Use local optimization (Nelder-Mead or similar) to minimize s
    #
    # Or: Use the fact that for hexagonal packings, the optimal arrangement often has
    # hexagons in a hexagonal lattice pattern.
    #
    # Let me try a different approach: formulate it as a nonlinear optimization problem
    # using scipy.optimize.minimize with SLSQP.
    
    from scipy.optimize import minimize
    
    def compute_outer_s(inner_positions, inner_angles, outer_angle_deg):
        """Compute the minimum outer hexagon side length s such that all inner hexagons fit."""
        # The outer hexagon is centered at the centroid of inner hexagons (or we can optimize its center too)
        # For now, let's center it at the origin and allow rotation.
        # 
        # A point p is inside the outer hexagon (centered at origin, side s, angle a) iff
        # for all 6 edge normals n_k: dot(p, n_k) <= s * apothem_factor
        #
        # For a regular hexagon with side s and angle a, the vertices are at distance s from center.
        # The apothem (distance from center to edge) is s * sqrt(3)/2.
        #
        # The 6 edge normals point at angles a + 30° + k*60° for k=0..5.
        # For each normal direction n_k, the constraint is dot(p, n_k) <= s * sqrt(3)/2.
        
        centroid = np.mean(inner_positions, axis=0)
        
        # Shift so centroid is at origin
        shifted = inner_positions - centroid
        
        # For each inner hexagon, compute all 6 vertices
        # The outer hexagon must contain all vertices of all inner hexagons
        
        # The minimum s is determined by the vertex that requires the largest s.
        # For a vertex v (relative to outer center) and outer hexagon angle a:
        # s must satisfy: for all k, dot(v, n_k(a)) <= s * sqrt(3)/2
        # So s >= max_k dot(v, n_k(a)) / (sqrt(3)/2) = 2/sqrt(3) * max_k dot(v, n_k(a))
        
        # But we also need to consider that the hexagon vertices are at specific positions.
        # Actually, the condition for a point to be inside a regular hexagon with circumradius R
        # (which equals side length s) centered at origin with angle a is:
        # The hexagon is the intersection of 6 half-planes. The distance from center to each edge
        # is R * sqrt(3)/2 = s * sqrt(3)/2.
        
        a_rad = np.radians(outer_angle_deg)
        # Edge normal directions
        normal_angles = a_rad + np.pi/6 + np.arange(6) * np.pi/3  # 30°, 90°, 150°, 210°, 270°, 330° offset
        
        max_required_s = 0.0
        for i in range(12):
            verts = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, inner_angles[i])
            for v in verts:
                v_shifted = v - centroid
                for ka in normal_angles:
                    proj = v_shifted[0] * np.cos(ka) + v_shifted[1] * np.sin(ka)
                    s_needed = proj / (np.sqrt(3)/2)
                    if s_needed > max_required_s:
                        max_required_s = s_needed
        
        # Also need s >= sqrt(12) ≈ 3.464
        return max(max_required_s, np.sqrt(12))
    
    def objective_and_constraints(params):
        """
        params: [x_0, y_0, ..., x_11, y_11, outer_angle_deg]
        Returns the objective value (s) and checks constraints.
        """
        inner_positions = params[:24].reshape(12, 2)
        outer_angle_deg = params[24]
        inner_angles = np.zeros(12)  # All inner hexagons have angle 0 for simplicity
        
        # Check non-overlap using SAT
        for i in range(12):
            for j in range(i+1, 12):
                c1 = inner_positions[i]
                c2 = inner_positions[j]
                diff = c1 - c2
                
                # For two unit hexagons with same orientation (angle 0),
                # edge normals are at angles 30°, 90°, 150°, 210°, 270°, 330°
                normal_angles = np.pi/6 + np.arange(6) * np.pi/3
                
                separated = False
                for ka in normal_angles:
                    n = np.array([np.cos(ka), np.sin(ka)])
                    proj = np.dot(diff, n)
                    # Width of unit hexagon in this direction
                    # For edge normal direction: width = sqrt(3)
                    # For vertex direction: width = 2
                    # The 6 directions 30,90,...,330 are edge normals, so width = sqrt(3)
                    if abs(proj) >= np.sqrt(3) - 1e-10:
                        separated = True
                        break
                
                if not separated:
                    # They overlap! Add a large penalty
                    return 1e6
        
        s = compute_outer_s(inner_positions, inner_angles, outer_angle_deg)
        return s
    
    # Initial guess: hexagonal lattice arrangement
    # 12 hexagons in a hexagonal pattern: center + 6 around + 5 more
    # Let's use a known configuration
    
    # Hexagonal lattice positions for 12 hexagons:
    # Layer 0: (0,0)
    # Layer 1: 6 hexagons at distance sqrt(3) in 6 directions
    # Layer 2: 5 more hexagons
    
    # For unit hexagons in a hexagonal lattice, adjacent centers are sqrt(3) apart.
    # Let's place them:
    init_positions = []
    # Center
    init_positions.append([0, 0])
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        init_positions.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    # Ring 2: need 5 more
    # Place at distance 2*sqrt(3) in alternating directions
    for k in range(5):
        angle = (k * 2 + 1) * np.pi / 6  # offset angles
        # Actually, let me just use specific positions
        pass
    
    # Simpler initial guess from the seed program
    init_positions = [
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4],
    ]
    
    init_params = np.array(init_positions).flatten()
    init_params = np.append(init_params, 0.0)  # outer angle = 0
    
    # Use Nelder-Mead optimization
    # This is a non-smooth problem due to the max in compute_outer_s,
    # but Nelder-Mead can handle it.
    
    best_params = init_params.copy()
    best_s = compute_outer_s(np.array(init_positions), np.zeros(12), 0.0)
    
    # Try a few random restarts
    np.random.seed(42)
    for restart in range(5):
        if elapsed() > DEADLINE - 60:
            break
        
        # Perturb the initial guess
        if restart == 0:
            params = init_params.copy()
        else:
            params = init_params + np.random.randn(len(init_params)) * 0.5
            params[24] = np.random.uniform(0, 60)  # random outer angle
        
        result = minimize(
            lambda p: objective_and_constraints(p),
            params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if result.fun < best_s and result.fun < 1e5:
            best_s = result.fun
            best_params = result.x
        
        if elapsed() > DEADLINE - 30:
            break
    
    # Extract the best solution
    inner_positions = best_params[:24].reshape(12, 2)
    outer_angle_deg = best_params[24]
    inner_angles = np.zeros(12)
    
    # Compute final s
    s = compute_outer_s(inner_positions, inner_angles, outer_angle_deg)
    
    # Verify non-overlap
    for i in range(12):
        for j in range(i+1, 12):
            c1 = inner_positions[i]
            c2 = inner_positions[j]
            diff = c1 - c2
            normal_angles = np.pi/6 + np.arange(6) * np.pi/3
            separated = False
            for ka in normal_angles:
                n = np.array([np.cos(ka), np.sin(ka)])
                proj = np.dot(diff, n)
                if abs(proj) >= np.sqrt(3) - 1e-8:
                    separated = True
                    break
            if not separated:
                # Fallback to seed solution
                inner_hex_data = np.array(
                    [
                        [0, 0, 0],
                        [-2.5, 0, 0],
                        [2.5, 0, 0],
                        [-1.25, 2.17, 0],
                        [1.25, 2.17, 0],
                        [-1.25, -2.17, 0],
                        [1.25, -2.17, 0],
                        [-3.75, 2.17, 0],
                        [3.75, 2.17, 0],
                        [-3.75, -2.17, 0],
                        [3.75, -2.17, 0],
                        [0, -4, 0],
                    ]
                )
                outer_hex_data = np.array([0, 0, 0])
                return inner_hex_data, outer_hex_data, 8.0
    
    # Build output arrays
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    outer_hex_data = np.array([0, 0, outer_angle_deg])
    
    # Ensure s >= sqrt(12)
    s = max(s, np.sqrt(12))
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
