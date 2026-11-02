"""
Strategy: Decoupled Simulated Annealing with Analytic Enclosing Hexagon.
Instead of optimizing the outer hexagon parameters alongside the inner ones, 
we treat the outer hexagon as the minimal container for the current inner 
configuration. This reduces the dimensionality and allows the SA to focus 
on packing the inner hexagons tightly. The outer side length is computed 
by scanning the rotation angle of the outer hexagon to find the minimum 
size that contains all inner hexagons.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(v1, v2, tol=1e-9):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        Returns True if disjoint, False if overlapping.
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
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
        return False

    def get_minimal_enclosing_hexagon(inner_positions):
        """
        Computes the minimal side length and optimal rotation/center 
        for a regular hexagon containing all inner hexagons.
        inner_positions: list of (x, y, angle)
        Returns: (side_length, center_x, center_y, angle_deg)
        """
        # Collect all vertices of all inner hexagons
        all_vertices = []
        for x, y, a in inner_positions:
            verts = hexagon_vertices(x, y, 1.0, a)
            all_vertices.extend(verts)
        
        if not all_vertices:
            return 1.0, 0.0, 0.0, 0.0
            
        all_vertices = np.array(all_vertices)
        
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 0.0
        
        # Scan rotation angles. 
        # Due to symmetry, we only need to check 0 to 60 degrees.
        # Fine grid for accuracy.
        angles = np.arange(0, 60.01, 1.0)
        
        for theta_deg in angles:
            theta = np.radians(theta_deg)
            # Normal vectors for the 6 sides of the outer hexagon
            # Side normals are at 30, 90, 150, 210, 270, 330 + theta
            normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + theta_deg)
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Project all vertices onto these normals
            # projections[i, j] = dot(vertex_i, normal_j)
            projections = all_vertices @ normals.T
            
            # For a valid enclosing hexagon centered at C with side s:
            # For each normal j, the extent of projections must be <= 2 * apothem = s * sqrt(3)
            # The width of the set of points along normal j is max(proj_j) - min(proj_j)
            # We need s * sqrt(3) >= width_j for all j.
            # So s >= max_j(width_j) / sqrt(3)
            
            widths = np.max(projections, axis=0) - np.min(projections, axis=0)
            required_side = np.max(widths) / np.sqrt(3)
            
            if required_side < best_side:
                best_side = required_side
                best_angle = theta_deg
                
                # Compute center
                # The center C must be such that for each normal j:
                # min_proj_j + C.dot(n_j) >= -s*sqrt(3)/2  => C.dot(n_j) >= -s*sqrt(3)/2 - min_proj_j
                # max_proj_j + C.dot(n_j) <= s*sqrt(3)/2   => C.dot(n_j) <= s*sqrt(3)/2 - max_proj_j
                # Let L_j = -s*sqrt(3)/2 - min_proj_j
                # Let U_j = s*sqrt(3)/2 - max_proj_j
                # We need to find C such that L_j <= C.dot(n_j) <= U_j for all j.
                # This is a feasibility problem. 
                # Note that U_j - L_j = s*sqrt(3) - (max - min) = s*sqrt(3) - width_j.
                # Since s is chosen as max(width)/sqrt(3), at least one constraint is tight (width_max = s*sqrt(3)).
                
                # A simple heuristic for center: average of the "midpoints" of the projection ranges?
                # Actually, the center of the minimal enclosing hexagon is the centroid of the "intersection of strips".
                # For a regular hexagon, if we know the side and angle, the center is determined by the bounding box in the rotated frame?
                # Let's try: Center is the average of the 6 "supporting lines" midpoints?
                # Midpoint of projection range j: M_j = (max + min) / 2
                # We want C such that C.dot(n_j) = M_j? 
                # If we set C.dot(n_j) = M_j for all j, this is an overdetermined system (6 eqs, 2 unknowns).
                # However, due to symmetry of the hexagon normals, sum(n_j) = 0.
                # Let's solve least squares for C: N C = M
                # Where N is 6x2 matrix of normals, M is 6x1 vector of midpoints.
                
                midpoints = (np.max(projections, axis=0) + np.min(projections, axis=0)) / 2.0
                # Solve N C = M
                # N.T N C = N.T M
                A = normals.T @ normals
                b = normals.T @ midpoints
                try:
                    center = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    center = np.zeros(2)
                
                best_center = (center[0], center[1])
                
        return best_side, best_center[0], best_center[1], best_angle

    def is_valid(inner_positions):
        """Check if inner hexagons are disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                v1 = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
                v2 = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    # Initial Configuration: Hexagonal Lattice
    # 1 center, 6 ring, 5 second ring? 
    # Let's start with a compact cluster.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3) ~ 1.732
    # But if we rotate them, they can pack tighter?
    # Standard lattice:
    # Center (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) * 1.15 (slightly separated to avoid initial overlap)
    # Actually, let's place them with a small gap to ensure validity.
    
    d = math.sqrt(3) * 1.05 # Slightly more than touching distance
    
    init_pos = []
    # Center
    init_pos.append([0.0, 0.0, 0.0])
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        init_pos.append([d * np.cos(angle), d * np.sin(angle), 0.0])
        
    # Ring 2 (partial to reach 12)
    # Positions in ring 2 of hex lattice are at distance 2*d or d*sqrt(3)?
    # Let's just add 5 more in a ring further out or fill gaps.
    # A 12-hexagon cluster: 
    # Center + 6 + 5.
    # The 5 can be placed in the "corners" of the second ring.
    # Angles: 30, 90, 150, 210, 270?
    # Distance for ring 2 corners is 2 * d * cos(30) = d * sqrt(3) ~ 3.0
    # Let's place them at distance 2.0 * d?
    # Actually, simpler: just use a grid-like arrangement that is known to be valid.
    # 4x3 grid was used before. Let's stick to a valid starting point.
    
    # Let's restart with the 4x3 grid from the parent, which is guaranteed valid if spaced correctly.
    # Spacing: x: 2.0, y: 1.8 (approx)
    # To be safe, let's use a larger spacing and let SA tighten it.
    
    init_pos = []
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    for y in y_coords:
        for x in x_coords:
            init_pos.append([x, y, 0.0])
            
    current_inner = [tuple(p) for p in init_pos]
    
    # Verify initial validity
    if not is_valid(current_inner):
        # Expand if needed
        scale = 1.1
        new_inner = []
        for p in current_inner:
            new_inner.append((p[0]*scale, p[1]*scale, p[2]))
        current_inner = new_inner
        if not is_valid(current_inner):
             # Fallback to very large spacing
             new_inner = []
             for p in current_inner:
                 new_inner.append((p[0]*1.5, p[1]*1.5, p[2]))
             current_inner = new_inner

    # Calculate initial cost
    side, cx, cy, ang = get_minimal_enclosing_hexagon(current_inner)
    current_cost = side
    best_inner = [tuple(p) for p in current_inner]
    best_side, best_cx, best_cy, best_ang = side, cx, cy, ang
    best_cost = side

    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.995
    
    steps = 0
    max_steps = 100000
    
    while T > T_min and steps < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        # Perturb one random hexagon
        i = np.random.randint(N)
        dx = np.random.normal(0, 0.5)
        dy = np.random.normal(0, 0.5)
        da = np.random.normal(0, 10.0)
        
        new_inner = [tuple(p) for p in current_inner]
        old_x, old_y, old_a = new_inner[i]
        new_inner[i] = (old_x + dx, old_y + dy, old_a + da)
        
        # Check validity (disjointness)
        if not is_valid(new_inner):
            # Reject immediately to save time on enclosing hex calc
            T *= alpha
            steps += 1
            continue
            
        # Calculate new cost
        new_side, new_cx, new_cy, new_ang = get_minimal_enclosing_hexagon(new_inner)
        new_cost = new_side
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / T):
            current_inner = new_inner
            current_cost = new_cost
            side, cx, cy, ang = new_side, new_cx, new_cy, new_ang
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_inner = [tuple(p) for p in current_inner]
                best_side, best_cx, best_cy, best_ang = side, cx, cy, ang
        else:
            # Revert
            pass
        # Note: If we rejected, current_inner remains unchanged.
        # Wait, if we accepted, current_inner is updated.
        # If we rejected, we should not update current_inner.
        # The logic above handles this: current_inner is only assigned if accepted.
        
        T *= alpha
        steps += 1

    # Final Refinement: Local search on best found
    # Try small perturbations to tighten further
    for _ in range(500):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            for dx in [-0.05, 0.05]:
                for dy in [-0.05, 0.05]:
                    new_inner = [tuple(p) for p in best_inner]
                    old_x, old_y, old_a = new_inner[i]
                    new_inner[i] = (old_x + dx, old_y + dy, old_a)
                    if not is_valid(new_inner):
                        continue
                    s, cx2, cy2, a2 = get_minimal_enclosing_hexagon(new_inner)
                    if s < best_cost - 1e-6:
                        best_cost = s
                        best_inner = new_inner
                        best_side, best_cx, best_cy, best_ang = s, cx2, cy2, a2
                        improved = True
        if not improved:
            break

    return best_inner, (best_cx, best_cy), best_side, best_ang

# EVOLVE-BLOCK-END
