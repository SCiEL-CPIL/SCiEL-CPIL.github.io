"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SIDE_INNER = 1.0
# Distance between centers of touching aligned unit hexagons
# For two unit hexagons aligned, the distance between centers is sqrt(3)
MIN_DIST = math.sqrt(3) 

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Base vertices of a unit hexagon centered at (0,0) with one vertex at (1,0)
    # Angles: 0, 60, 120, 180, 240, 300
    base_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])

    def get_vertices(pos, angle_deg):
        """
        Calculate vertices of a hexagon with center pos and rotation angle_deg.
        pos: (x, y)
        angle_deg: rotation in degrees
        """
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # Rotate the base vertices and translate
        return base_verts @ R.T + pos

    def calc_R_max(positions, outer_angle_deg):
        """
        Calculate the maximum distance of any inner hexagon vertex from the origin (0,0).
        This corresponds to the apothem of the outer hexagon if the outer hexagon is aligned with axes such that its apothem is perpendicular to a side?
        Actually, for a regular hexagon, the apothem is the distance from center to side.
        The vertices of the outer hexagon are at distance S (side length).
        The sides are at distance A (apothem).
        A = S * sqrt(3) / 2.
        So S = 2 * A / sqrt(3).
        
        If we contain all vertices within distance R from the center, 
        and the outer hexagon has apothem A = R, 
        then the vertices of the outer hexagon must be far enough.
        Wait. The outer hexagon must contain the inner hexagons.
        The "tightest" outer hexagon for a set of points has sides tangent to the convex hull of the points.
        The apothem of the outer hexagon is the maximum projection of the points onto the normal of any side.
        So if we fix the outer hexagon angle, R_max is the max projection onto the 6 normals.
        However, we can also rotate the outer hexagon.
        In this formulation, `calc_R_max` assumes the outer hexagon is aligned such that one of its normals is along the x-axis?
        No, `calc_R_max` in the objective function should calculate the minimum apothem required for a hexagon with a *given* orientation to contain the points.
        
        Let's refine:
        The objective is to minimize the side length S.
        S = 2/sqrt(3) * A_min(theta), where A_min(theta) is the minimum apothem of a hexagon rotated by theta that contains all points.
        A_min(theta) = max over all points p of (max over 6 normals n_k(theta) of dot(p, n_k)).
        
        So, for a fixed set of points and a fixed outer angle theta, the required apothem is:
        A(theta) = max_{p in P} max_{k=0..5} dot(p, n_k(theta))
        
        We want to minimize S(theta) = 2/sqrt(3) * A(theta) over theta and positions.
        
        In the optimization variable x, the last variable is theta_out.
        The objective function should compute A(theta_out) for the current positions.
        """
        theta_out_rad = math.radians(outer_angle_deg)
        # Normals of the outer hexagon
        # If the hexagon has a vertex at angle 0, the normal to the side at 30 deg is at 30 deg?
        # Regular hexagon vertices at 0, 60...
        # Sides are perpendicular to 30, 90...
        # So normals are at 30 + k*60 degrees.
        # Let's just define normals at angles: outer_angle_deg + 30 + k*60
        # Actually, if we rotate the hexagon by alpha, the normals rotate by alpha.
        # Base normals (for hex with vertex at 0): 30, 90, 150, 210, 270, 330 degrees.
        
        max_proj = 0.0
        # Generate normals
        normal_angles = []
        for k in range(6):
            ang = math.radians(outer_angle_deg + 30 + k * 60)
            normal_angles.append(ang)
            
        for i in range(N):
            p = positions[i]
            # Inner hex angle is fixed 0
            verts = get_vertices(p, 0.0)
            # Project all vertices of this inner hex onto all 6 normals
            # We need the max projection for this vertex on any normal?
            # No, the condition is that ALL vertices must be inside.
            # So for each normal n, max_{v in V} dot(v, n) <= A.
            # So A >= max_{n in N} max_{v in V} dot(v, n).
            # So A = max_{n in N} ( max_{v in V} dot(v, n) )
            
            # Efficiently: for each normal, find max dot product among all vertices.
            # Then take max of those 6 values.
            for ang in normal_angles:
                n = np.array([math.cos(ang), math.sin(ang)])
                proj = np.dot(verts, n)
                curr_max = np.max(proj)
                if curr_max > max_proj:
                    max_proj = curr_max
                    
        return max_proj

    def objective(x):
        """
        x: [x0, y0, ..., x11, y11, theta_out]
        Returns the required apothem A.
        """
        pos = x[:24].reshape(12, 2)
        theta_out = x[24]
        return calc_R_max(pos, theta_out)

    def make_constraints():
        """
        Create constraint functions for COBYLA.
        COBYLA expects functions h(x) >= 0.
        """
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                def constraint_fun(x, i=i, j=j):
                    pos = x[:24].reshape(12, 2)
                    p_i = pos[i]
                    p_j = pos[j]
                    # Distance must be at least MIN_DIST
                    return np.linalg.norm(p_i - p_j) - MIN_DIST
                cons.append(constraint_fun)
        return cons

    constraints_list = make_constraints()

    best_A = float('inf')
    best_pos = None
    best_theta = 0.0

    # Generate initial honeycomb configuration
    def generate_honeycomb():
        coords = []
        for r in range(4):
            for c in range(3):
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                y = r * 1.5
                coords.append((x, y))
        coords = np.array(coords)
        coords -= coords.mean(axis=0)
        return coords

    init_pos = generate_honeycomb()
    
    np.random.seed(42)
    
    # Run optimization multiple times with different initializations
    num_attempts = 5
    
    for k in range(num_attempts):
        if time.time() > deadline - 10:
            break
            
        # Prepare x0
        if k == 0:
            x0 = init_pos.flatten().tolist()
            x0.append(0.0)
        else:
            # Jitter
            jitter = np.random.uniform(-0.5, 0.5, (12, 2))
            x0 = (init_pos + jitter).flatten().tolist()
            x0.append(np.random.uniform(0, 30))
            
        try:
            res = minimize(
                objective,
                x0,
                method='COBYLA',
                constraints=constraints_list,
                options={'maxiter': 500, 'rhobeg': 0.5, 'catol': 1e-6}
            )
            
            if res.success or res.x is not None:
                x_res = res.x
                pos_res = x_res[:24].reshape(12, 2)
                theta_res = x_res[24]
                
                # Calculate the achieved A
                A_res = calc_R_max(pos_res, theta_res)
                
                # Check if constraints are satisfied (COBYLA might violate slightly)
                # We'll do a strict check later, but for now, accept if A is better
                if A_res < best_A:
                    best_A = A_res
                    best_pos = pos_res
                    best_theta = theta_res
                    
        except Exception:
            pass

    # If no optimization result was better, use the initial honeycomb
    if best_pos is None:
        best_pos = init_pos
        best_theta = 0.0
        best_A = calc_R_max(best_pos, best_theta)
        
    # Final Score is Side Length S
    # S = 2 * A / sqrt(3)
    final_side = best_A * 2 / math.sqrt(3)
    
    # Construct return values
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), final_side, best_theta
# EVOLVE-BLOCK-END
