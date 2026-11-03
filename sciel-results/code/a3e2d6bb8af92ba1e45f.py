import math
import time
import numpy as np
from scipy.optimize import fsolve, minimize

N = 12
sqrt3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / sqrt3

    def check_disjoint(poly1, poly2, eps=1e-9):
        """
        Fast check for disjointness (interiors do not overlap).
        Returns True if disjoint, False if overlapping.
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 + eps < min_x2 or max_x2 + eps < min_x1 or max_y1 + eps < min_y2 or max_y2 + eps < min_y1:
            return True

        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2 (strictly inside)
        for pt in poly1:
            if point_in_poly(pt, poly2):
                return False
        
        # Check vertices of v2 in v1
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return False
        
        # Check edge intersections (proper intersection, not just touching)
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) - (b[1]-a[1]) * (c[0]-a[0])
            
            d1 = ccw(p3, p4, p1)
            d2 = ccw(p3, p4, p2)
            d3 = ccw(p1, p2, p3)
            d4 = ccw(p1, p2, p4)
            
            # Proper intersection if signs differ
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        
        return True

    # Strategy:
    # 1. Define a specific "Contact Graph" topology for 12 hexagons.
    #    A common dense packing for 12 items is a central hexagon surrounded by 6,
    #    and then another ring of 5 filling the gaps? Or a 3-4-5 strip?
    #    Let's try a "Honeycomb Cluster" of 12.
    #    Center: (0,0).
    #    Ring 1: 6 hexes at distance sqrt(3) from center.
    #    Ring 2: 5 hexes filling the "valleys" between Ring 1 hexes?
    #    Actually, a 12-hexagon cluster often looks like a larger hexagon shape.
    #    A regular hexagon of side 2 has 7 hexes. Side 3 has 19.
    #    So 12 is irregular.
    
    # Let's use a specific known good configuration as the initial guess for fsolve.
    # We will define the contacts explicitly.
    
    # Initial guess: Honeycomb lattice points
    # v1 = (0, sqrt(3)), v2 = (1.5, sqrt(3)/2)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)

    best_score = float('inf')
    best_positions = init_positions.copy()
    best_angles = init_angles.copy()

    # Define the contact graph.
    # We assume a rigid structure where certain pairs are in contact.
    # For a generic dense packing, we can assume all "close" pairs are in contact.
    # To make fsolve work, we need exactly 2N equations for 2N variables (positions).
    # If we fix the global rotation and translation (3 DOF removed), we have 2N-3 variables.
    # But we also have angles. Let's fix angles to 0 for the initial solve, then optimize.
    
    # Let's try to solve for positions assuming all angles are 0.
    # We pick a set of contacts that forms a rigid graph.
    # A simple rigid graph for 12 nodes is a triangulated lattice subset.
    
    # Let's define the contacts based on the initial lattice positions.
    # Any pair with distance < sqrt(3) + eps is considered a contact.
    contacts = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(init_positions[i] - init_positions[j])
            if d < sqrt3 + 0.1:
                contacts.append((i, j))
                
    # If the graph is not rigid (too few contacts), fsolve might fail or drift.
    # A standard honeycomb cluster of 12 usually has enough contacts.
    # Let's ensure we have at least 2N-3 contacts.
    # If not, we add the closest pairs until we have enough.
    if len(contacts) < 2*N - 3:
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                if (i,j) not in contacts and (j,i) not in contacts:
                    d = np.linalg.norm(init_positions[i] - init_positions[j])
                    dists.append((d, i, j))
        dists.sort()
        for d, i, j in dists:
            if len(contacts) >= 2*N - 3:
                break
            contacts.append((i, j))

    def equations(params):
        positions = params.reshape(N, 2)
        res = []
        for i, j in contacts:
            # Distance between centers of two touching unit hexagons (aligned) is sqrt(3)
            # Wait, if they are aligned (angle 0), and touching flat-to-flat, distance is sqrt(3).
            # If they are touching vertex-to-vertex, distance is 2.
            # In a honeycomb, they touch flat-to-flat.
            # So distance should be sqrt(3).
            diff = positions[i] - positions[j]
            d = np.linalg.norm(diff)
            res.append(d - sqrt3)
        return res

    # Solve for positions
    x0 = init_positions.flatten()
    try:
        if time.time() < deadline - 20:
            sol = fsolve(equations, x0, full_output=True)
            positions_sol, info, ier, msg = sol
            if ier == 1:
                new_positions = positions_sol.reshape(N, 2)
                new_angles = np.zeros(N)
                
                # Verify validity
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, 0.0) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if score < best_score:
                        best_score = score
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
    except Exception:
        pass

    # Now, refine with optimization allowing angles to change.
    # This allows "interlocking" if beneficial (though for regular hexes, aligned is usually best).
    # We use SLSQP with overlap penalty to ensure validity.
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if penalty > 1.0:
                    break
                if not check_disjoint(polys[i], polys[j]):
                    penalty += 10.0
            if penalty > 1.0:
                break
        
        if penalty > 0:
            return penalty + calc_min_outer_side(positions, angles, 0.0)
            
        return calc_min_outer_side(positions, angles, 0.0)

    x0 = np.concatenate([best_positions.flatten(), best_angles])
    
    try:
        if time.time() < deadline - 10:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_score = res.fun
                    best_positions = new_positions
                    best_angles = new_angles
    except Exception:
        pass

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
