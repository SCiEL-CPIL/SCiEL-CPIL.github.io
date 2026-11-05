"""Program for packing 11 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place 11 unit regular hexagons (side length 1) inside a regular hexagon,
    minimizing the outer side length.
    
    Strategy:
    1. Define the vertices of a unit regular hexagon.
    2. Use a local optimization to place the 11 hexagons.
    3. For each configuration, compute the minimal enclosing regular hexagon side length.
    4. Minimize this side length.
    """
    
    # Vertices of a unit regular hexagon (side length 1) centered at origin
    # Angle of vertices: 0, 60, 120, 180, 240, 300 degrees
    # For a regular hexagon with side length s, the distance from center to vertex is s
    # So for side length 1, vertices are at distance 1 from center
    hex_verts = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    
    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon centered at (cx, cy) with rotation angle_deg."""
        angle = np.radians(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        return hex_verts @ R.T + np.array([cx, cy])
    
    def hexagon_contains_point(px, py, hex_center, hex_angle_deg, hex_side):
        """
        Check if point (px, py) is inside or on the boundary of a regular hexagon
        with center hex_center, rotation hex_angle_deg, and side length hex_side.
        """
        # Transform point to hexagon's local coordinate system
        angle = np.radians(hex_angle_deg)
        R_inv = np.array([[np.cos(-angle), -np.sin(-angle)],
                          [np.sin(-angle), np.cos(-angle)]])
        local_point = R_inv @ (np.array([px, py]) - np.array(hex_center))
        
        # In local coords, a regular hexagon with side length s has:
        # - 3 pairs of parallel sides
        # - The hexagon is defined by |x| <= s, |x/2 + sqrt(3)/2*y| <= s*sqrt(3)/2, |-x/2 + sqrt(3)/2*y| <= s*sqrt(3)/2
        # Actually, let's use the half-widths approach
        # For a regular hexagon with side length s, the apothem (distance from center to side) is s*sqrt(3)/2
        # The hexagon can be defined by 3 inequalities:
        # |dot(p, n_i)| <= apothem for i=0,1,2 where n_i are unit normals to 3 pairs of opposite sides
        
        apothem = hex_side * np.sqrt(3) / 2
        
        # Normals to the 3 pairs of sides (in local coords, for a hexagon with flat top/bottom or pointy top)
        # For our hexagon defined with vertices at angles 0, 60, ..., 300:
        # The sides are between consecutive vertices
        # Side 0: from vertex 0 to vertex 1, normal points outward
        # Let's compute the normals properly
        
        # Actually, simpler: a regular hexagon with side length s centered at origin with the given orientation
        # The distance from center to each side (apothem) is s * sqrt(3) / 2
        # The 3 outward normals (in local frame) are at angles 30, 90, 150 degrees (perpendicular to sides)
        # Wait, let me think again.
        
        # Vertices at angles 0, 60, 120, 180, 240, 300
        # Side between vertex 0 (angle 0) and vertex 1 (angle 60): midpoint at angle 30, normal outward at angle 30
        # Side between vertex 1 (angle 60) and vertex 2 (angle 120): midpoint at angle 90, normal outward at angle 90
        # Side between vertex 2 (angle 120) and vertex 3 (angle 180): midpoint at angle 150, normal outward at angle 150
        # And the opposite sides have normals at 210, 270, 330
        
        normals = np.array([
            [np.cos(np.radians(30)), np.sin(np.radians(30))],
            [np.cos(np.radians(90)), np.sin(np.radians(90))],
            [np.cos(np.radians(150)), np.sin(np.radians(150))]
        ])
        
        for n in normals:
            if abs(np.dot(local_point, n)) > apothem + 1e-9:
                return False
        return True
    
    def min_enclosing_hexagon_side(points, hex_angle_deg):
        """
        Find the minimum side length of a regular hexagon with rotation hex_angle_deg
        that contains all given points.
        
        For a regular hexagon with rotation theta, the side length s is determined by
        the maximum over all points of the required s to contain that point.
        
        For a point p, the minimum s such that p is inside a hexagon of side s with rotation theta
        centered at c is:
        s = max_i |dot(p - c, n_i)| / (sqrt(3)/2)
        
        where n_i are the 3 outward normals of the hexagon.
        
        To minimize s, we also need to choose the center c. For a fixed set of points and fixed
        hexagon orientation, the optimal center and minimal s can be found by noting that
        s = max_i (|dot(p_j - c, n_i)|) / (sqrt(3)/2) for all j, i.
        
        This is equivalent to finding the minimum s such that there exists c with
        |dot(p_j - c, n_i)| <= s * sqrt(3)/2 for all j, i.
        
        This is a linear programming problem in variables c_x, c_y, s.
        """
        angle = np.radians(hex_angle_deg)
        normals = np.array([
            [np.cos(angle + np.radians(30)), np.sin(angle + np.radians(30))],
            [np.cos(angle + np.radians(90)), np.sin(angle + np.radians(90))],
            [np.cos(angle + np.radians(150)), np.sin(angle + np.radians(150))]
        ])
        
        # We want to minimize s subject to:
        # |dot(p_j - c, n_i)| <= s * sqrt(3)/2 for all j, i
        # This is equivalent to:
        # dot(p_j, n_i) - dot(c, n_i) <= s * sqrt(3)/2
        # -dot(p_j, n_i) + dot(c, n_i) <= s * sqrt(3)/2
        
        # Variables: c_x, c_y, s
        # This is a linear program. Let's use scipy's linprog.
        
        n_points = len(points)
        # Constraints: for each point j and normal i (and both signs):
        # dot(p_j, n_i) - dot(c, n_i) - s * sqrt(3)/2 <= 0
        # -dot(p_j, n_i) + dot(c, n_i) - s * sqrt(3)/2 <= 0
        
        # A matrix with rows for each constraint, variables are [c_x, c_y, s]
        A = []
        b = []
        
        for j in range(n_points):
            for i in range(3):
                # dot(p_j, n_i) - dot(c, n_i) - s * sqrt(3)/2 <= 0
                # -n_i[0]*c_x - n_i[1]*c_y - sqrt(3)/2 * s <= -dot(p_j, n_i)
                A.append([-normals[i][0], -normals[i][1], -np.sqrt(3)/2])
                b.append(-np.dot(points[j], normals[i]))
                
                # -dot(p_j, n_i) + dot(c, n_i) - s * sqrt(3)/2 <= 0
                # n_i[0]*c_x + n_i[1]*c_y - sqrt(3)/2 * s <= dot(p_j, n_i)
                A.append([normals[i][0], normals[i][1], -np.sqrt(3)/2])
                b.append(np.dot(points[j], normals[i]))
        
        A = np.array(A)
        b = np.array(b)
        
        # Minimize s, i.e., c = [0, 0, 1]
        c_obj = np.array([0, 0, 1])
        
        # s must be non-negative
        bounds = [(None, None), (None, None), (0, None)]
        
        from scipy.optimize import linprog
        result = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if result.success:
            return result.x[2]
        else:
            # Fallback: use a large value
            return 100.0
    
    def total_outer_side_length(positions, angles, outer_center, outer_angle):
        """
        Compute the side length of the minimal enclosing regular hexagon.
        positions: list of (x, y) centers of inner hexagons
        angles: list of rotation angles of inner hexagons
        outer_center: (x, y) center of outer hexagon
        outer_angle: rotation angle of outer hexagon
        """
        # Get all vertices of all inner hexagons
        all_points = []
        for idx in range(len(positions)):
            verts = get_hex_vertices(positions[idx][0], positions[idx][1], angles[idx])
            all_points.extend(verts.tolist())
        
        all_points = np.array(all_points)
        
        # Find minimal enclosing hexagon with given outer_center and outer_angle
        # The minimal side length is determined by the maximum "projection" of all points
        # relative to the outer hexagon center
        
        angle = np.radians(outer_angle)
        normals = np.array([
            [np.cos(angle + np.radians(30)), np.sin(angle + np.radians(30))],
            [np.cos(angle + np.radians(90)), np.sin(angle + np.radians(90))],
            [np.cos(angle + np.radians(150)), np.sin(angle + np.radians(150))]
        ])
        
        # For the outer hexagon centered at outer_center with side length s,
        # a point p is inside if |dot(p - outer_center, n_i)| <= s * sqrt(3)/2 for all i
        # So s >= max_j,i |dot(p_j - outer_center, n_i)| / (sqrt(3)/2)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        for p in all_points:
            for n in normals:
                proj = abs(np.dot(p - oc, n))
                if proj > max_proj:
                    max_proj = proj
        
        s = max_proj / (np.sqrt(3) / 2)
        return s
    
    # Initial configuration: hexagonal packing
    # 11 hexagons: 1 in center, 6 around it, 4 more?
    # Let's use a hexagonal number pattern: 1 + 6 + 4 = 11? No, hex numbers are 1, 7, 19...
    # Let's just place them in a grid-like hexagonal arrangement
    
    # Distance between centers of adjacent unit hexagons in a hexagonal packing:
    # For flat-top hexagons, horizontal spacing is 3/2 * side = 1.5
    # Vertical spacing is sqrt(3) * side = sqrt(3)
    # For pointy-top, it's different.
    
    # Let's use a simple initial guess and let optimization handle it
    # Start with a linear arrangement or a small cluster
    
    # Hexagonal cluster: center at (0,0), 6 neighbors at distance sqrt(3) (apothem * 2? No)
    # Two adjacent unit hexagons: the distance between centers is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # if they share a side. But they don't have to share sides.
    
    # Let's start with a reasonable initial guess
    initial_positions = []
    initial_angles = []
    
    # Center hexagon
    initial_positions.append((0.0, 0.0))
    initial_angles.append(0.0)
    
    # 6 neighbors at distance sqrt(3) from center
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        initial_positions.append((x, y))
        initial_angles.append(0.0)
    
    # 4 more hexagons further out
    # Place them at distance 2*sqrt(3) from center
    for k in range(4):
        angle = np.radians(60 * k + 30)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        initial_positions.append((x, y))
        initial_angles.append(0.0)
    
    # Now optimize
    # Variables: for each hexagon, (x, y, angle), plus outer center (ox, oy) and outer angle
    # Total variables: 11*3 + 2 + 1 = 36
    
    def objective(params):
        # Extract positions and angles
        positions = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            positions.append((x, y))
            angles.append(a)
        
        outer_center = (params[33], params[34])
        outer_angle = params[35]
        
        # Check for overlaps (penalty)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Check if hexagons i and j overlap
                # Simple distance check: if centers are too close, they might overlap
                # Minimum distance between centers of two non-overlapping unit hexagons is sqrt(3)
                # (when they share a side)
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3) - 1e-6:
                    # Penalty for overlap
                    overlap = np.sqrt(3) - dist
                    penalty += overlap * 100.0
        
        side_length = total_outer_side_length(positions, angles, outer_center, outer_angle)
        
        return side_length + penalty
    
    # Initial parameter vector
    x0 = []
    for i in range(N):
        x0.append(initial_positions[i][0])
        x0.append(initial_positions[i][1])
        x0.append(initial_angles[i])
    x0.append(0.0)  # outer center x
    x0.append(0.0)  # outer center y
    x0.append(0.0)  # outer angle
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Extract best solution
    best_params = result.x
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a = best_params[3*i + 2]
        inner.append([x, y, a])
    
    outer_center = (best_params[33], best_params[34])
    outer_angle = best_params[35]
    
    # Compute the actual side length
    positions = [(inner[i][0], inner[i][1]) for i in range(N)]
    angles = [inner[i][2] for i in range(N)]
    side_length = total_outer_side_length(positions, angles, outer_center, outer_angle)
    
    return inner, outer_center, side_length, outer_angle

# EVOLVE-BLOCK-END
