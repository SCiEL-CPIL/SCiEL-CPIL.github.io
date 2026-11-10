import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import math

def hexagon_packing_12():
    start_time = time.time()
    timeout = 580.0  # Safety margin for 600s limit

    n_inner = 12
    inner_side = 1.0
    min_dist = np.sqrt(3)  # Distance between centers of touching unit hexagons (aligned)

    # Helper to compute hexagon vertices
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append([x, y])
        return np.array(verts)

    def check_validity(inner_hex_data, outer_hex_data, s):
        # Check finite values
        if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
            return False
        if s < math.sqrt(12) - 1e-9:
            return False
        if not np.isfinite(s):
            return False
        
        # Check outer vertices distinct
        outer_cx, outer_cy, outer_angle = outer_hex_data
        outer_verts = get_hex_verts(outer_cx, outer_cy, s, outer_angle)
        for i in range(6):
            for j in range(i+1, 6):
                if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                    return False

        # Shift all centers by minus outer center
        shift = np.array([outer_cx, outer_cy])
        inner_centers = inner_hex_data[:, :2] - shift
        inner_angles = inner_hex_data[:, 2]
        
        outer_verts_shifted = outer_verts - shift

        # Check edge lengths of outer hexagon
        for i in range(6):
            v1 = outer_verts_shifted[i]
            v2 = outer_verts_shifted[(i+1)%6]
            length = np.linalg.norm(v2 - v1)
            if abs(length - s) > 1e-8:
                return False
            # Actually, the task says "a unit edge whose computed length differs from 1 by more than 1e-8"
            # Wait, "a unit edge" refers to the inner hexagons?
            # "after shifting all centers by minus the outer center, a unit edge whose computed length differs from 1 by more than 1e-8"
            # This implies the inner hexagons must have side length 1.
            # But the input is "rows (x, y, angle_deg) for hexagons of side 1".
            # So the side length is fixed at 1. The check is likely redundant or verifies the geometry.
            # Let's check inner hexagon edges.

        for i in range(n_inner):
            cx, cy, angle = inner_centers[i][0], inner_centers[i][1], inner_angles[i]
            verts = get_hex_verts(cx, cy, inner_side, angle)
            for j in range(6):
                v1 = verts[j]
                v2 = verts[(j+1)%6]
                length = np.linalg.norm(v2 - v1)
                if abs(length - 1.0) > 1e-8:
                    return False

        # Separation constraints: Separating Axis Theorem (SAT)
        # Two convex polygons are separated if there exists an axis such that their projections do not overlap.
        # The axes are the normals to the edges of either polygon.
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                ci = inner_centers[i]
                cj = inner_centers[j]
                ai = inner_angles[i]
                aj = inner_angles[j]
                
                vi = get_hex_verts(ci[0], ci[1], inner_side, ai)
                vj = get_hex_verts(cj[0], cj[1], inner_side, aj)
                
                # Get normals for polygon i
                normals = []
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k+1)%6]
                    edge = p2 - p1
                    # Normal is perpendicular to edge
                    n = np.array([-edge[1], edge[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                
                # Get normals for polygon j
                for k in range(6):
                    p1 = vj[k]
                    p2 = vj[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                
                # Check if any axis separates them
                separated = False
                for n in normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    # If intervals do not overlap
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                
                if not separated:
                    return False

        # Containment: Inner vertices must be inside outer hexagon
        # For every outer edge (p1, p2) in vertex order, cross(p2 - p1, v - p1) must be >= -1e-6
        # This assumes counter-clockwise vertex order.
        # get_hex_verts produces CCW order?
        # i=0: angle 0 -> (1, 0)
        # i=1: angle 60 -> (0.5, sqrt(3)/2)
        # This is CCW.
        
        for i in range(n_inner):
            cx, cy, angle = inner_centers[i][0], inner_centers[i][1], inner_angles[i]
            verts = get_hex_verts(cx, cy, inner_side, angle)
            for v in verts:
                for k in range(6):
                    p1 = outer_verts_shifted[k]
                    p2 = outer_verts_shifted[(k+1)%6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < -1e-6:
                        return False
        
        return True

    # Strategy 1: Deterministic known packing
    def known_valid_packing():
        # Honeycomb cluster: 1 center, 6 ring 1, 5 ring 2
        # Centers:
        centers = []
        # Center
        centers.append([0.0, 0.0])
        
        # Ring 1: distance sqrt(3)
        d1 = math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            centers.append([d1 * math.cos(angle), d1 * math.sin(angle)])
        
        # Ring 2: 5 hexagons. In a hexagonal packing, ring 2 has 12 positions.
        # We need 5 more. To minimize bounding box, we should pick 5 that are "outer" but symmetric.
        # The 12 positions in ring 2 are at distance 3.0 and 2*sqrt(3)?
        # Let's calculate ring 2 positions.
        # Ring 2 is at distance 3.0 (for vertices) or 2*sqrt(3) (for edge-midpoints)?
        # Hexagonal lattice vectors: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
        # Ring 1: ±u, ±v, ±(u-v)
        # Ring 2: ±2u, ±2v, ±2(u-v), ±(u+v), ±(u+2v), ±(2u+v)? No.
        # The 12 neighbors of the 6 ring-1 hexagons.
        # Let's list coordinates.
        # u = (s, 0), v = (s/2, s*sqrt(3)/2) where s = sqrt(3)
        # s = 1.732
        # u = (1.732, 0)
        # v = (0.866, 1.5)
        # u-v = (0.866, -1.5)
        
        # Ring 1:
        # (s, 0)
        # (s/2, 1.5s)
        # (-s/2, 1.5s)
        # (-s, 0)
        # (-s/2, -1.5s)
        # (s/2, -1.5s)
        
        # Ring 2 candidates (distance from origin):
        # 2u = (2s, 0) -> dist 2s = 2*sqrt(3) ~ 3.46
        # 2v = (s, 3s) -> dist 2*sqrt(3)
        # 2(u-v) = (s, -3s)
        # u+v = (1.5s, 1.5s) -> dist 1.5*sqrt(2)*s? No. sqrt((1.5s)^2 + (1.5s)^2) = 1.5s*sqrt(2) ~ 3.67
        # Wait, u+v is a neighbor of u and v.
        # Let's check distance from origin to u+v.
        # u=(s,0), v=(s/2, s*sqrt(3)/2)
        # u+v = (1.5s, s*sqrt(3)/2)
        # dist = sqrt(2.25s^2 + 0.75s^2) = sqrt(3s^2) = s*sqrt(3) = 3.
        # So "corner" hexagons in ring 2 are at distance 3.0.
        # "Side" hexagons in ring 2 (like 2u) are at distance 2*sqrt(3) ~ 3.46.
        
        # To minimize the bounding hexagon, we should prefer the hexagons at distance 3.0?
        # Or maybe a mix?
        # The 6 corner positions are at distance 3.0.
        # The 6 side positions are at distance 2*sqrt(3).
        # 3.0 < 3.46. So the corners are closer to the center!
        # This is counter-intuitive. Let's re-verify.
        # Distance from origin to (1.5s, s*sqrt(3)/2).
        # s = sqrt(3).
        # x = 1.5*sqrt(3) = 2.598
        # y = sqrt(3)*sqrt(3)/2 = 3/2 = 1.5
        # dist = sqrt(2.598^2 + 1.5^2) = sqrt(6.75 + 2.25) = sqrt(9) = 3.
        # Distance to 2u = (2s, 0) = (2*sqrt(3), 0).
        # dist = 2*sqrt(3) = 3.464.
        
        # So the 6 "corner" hexagons are at distance 3.0.
        # The 6 "side" hexagons are at distance 3.464.
        
        # We need 5 more hexagons.
        # If we pick 5 of the 6 corners, we break symmetry.
        # If we pick 3 corners and 2 sides?
        # Let's try to pick 5 corners and see if we can optimize.
        # Or maybe 6 corners minus 1?
        
        # Let's define the 6 corner positions (angles 30, 90, 150, 210, 270, 330? No).
        # u+v is at angle atan2(s*sqrt(3)/2, 1.5s) = atan2(1.5, 2.598) = 30 degrees.
        # So corners are at 30, 90, 150, 210, 270, 330.
        
        # Let's pick 5 of them. Which one to omit?
        # Maybe omit one to allow the outer hexagon to be smaller?
        # Or maybe pick a different set?
        
        # Let's start with 5 corners: 30, 90, 150, 210, 270. (Omit 330).
        # This is asymmetric.
        
        # Alternative: 3 corners and 2 sides?
        
        # Let's try a full optimization from a good start.
        # Start with 1 center, 6 ring 1, and 5 random ring 2 positions.
        
        corners = []
        for i in range(6):
            angle = math.radians(30 + 60 * i)
            corners.append([3.0 * math.cos(angle), 3.0 * math.sin(angle)])
        
        sides = []
        for i in range(6):
            angle = math.radians(0 + 60 * i)
            sides.append([2*math.sqrt(3) * math.cos(angle), 2*math.sqrt(3) * math.sin(angle)])
        
        # Combine center + ring 1 + 5 corners
        # We can try all combinations of 5 out of 6 corners (6 ways)
        # And maybe some with sides.
        
        best_s = float('inf')
        best_inner = None
        
        # Try 5 corners
        for omit in range(6):
            ring2 = [c for i, c in enumerate(corners) if i != omit]
            all_centers = [centers[0]] + centers[1:7] + ring2
            # All at angle 0
            inner_data = np.array([[c[0], c[1], 0.0] for c in all_centers])
            # Estimate s
            # Max radial extent of inner hexagons
            max_r = 0
            for c in all_centers:
                r = np.linalg.norm(c)
                # Max extent is r + sqrt(3)/2 (if edge points out) or r + 1 (if vertex points out)
                # With angle 0, vertices at 0, 60...
                # If c is at angle 30, vertex at 30 is radial. Extent r+1.
                # If c is at angle 0, edge normal at 0 is radial? No, normal at 0 is perpendicular to edge.
                # Edge at 0 is horizontal? No.
                # Vertex at 0 is (1,0). Edge between 330 and 30?
                # Edge normal at 30?
                # Let's just compute the required s for a given angle.
                # We can try to optimize s for this fixed configuration by rotating outer hex.
                # But simpler: just compute the max distance from origin to any vertex of any inner hex.
                # This gives a lower bound on s if outer is centered at origin and angle 0?
                # No, outer hex can be rotated.
                # The minimal s to contain a set of points in a hexagon is related to the maximum projection onto the 6 normals.
                
                # Let's just run a quick optimization on s and outer_angle for this fixed inner config.
                # Variables: s, outer_angle
                # Objective: s
                # Constraints: All inner vertices inside outer hex.
                
                def obj(vars):
                    return vars[0]
                
                def cons(vars):
                    s_val = vars[0]
                    o_ang = vars[1]
                    o_verts = get_hex_verts(0, 0, s_val, o_ang)
                    c_cons = []
                    for i in range(n_inner):
                        cx, cy, ang = inner_data[i]
                        verts = get_hex_verts(cx, cy, 1.0, ang)
                        for v in verts:
                            for k in range(6):
                                p1 = o_verts[k]
                                p2 = o_verts[(k+1)%6]
                                edge = p2 - p1
                                vec = v - p1
                                cross = edge[0]*vec[1] - edge[1]*vec[0]
                                c_cons.append(cross + 1e-6)
                    return c_cons
                
                try:
                    res = minimize(obj, np.array([4.0, 0.0]), method='SLSQP', 
                                   constraints=[{'type': 'ineq', 'fun': cons}],
                                   bounds=[(3.0, 6.0), (0.0, 60.0)],
                                   options={'maxiter': 100})
                    if res.success and res.fun < best_s:
                        best_s = res.fun
                        best_inner = inner_data.copy()
                        best_outer = np.array([0.0, 0.0, res.x[1]])
                except:
                    pass
        
        if best_inner is not None:
            return best_inner, best_outer, best_s
        else:
            # Fallback
            inner_data = np.array([[0,0,0]]*12)
            return inner_data, np.array([0,0,0]), 10.0

    # Run known packing
    inner, outer, s = known_valid_packing()
    if check_validity(inner, outer, s):
        return inner, outer, s

    # Strategy 2: Optimization
    # Variables: 24 centers + 12 angles + 1 s + 1 outer_angle = 38 variables
    # This is a lot.
    # Let's fix angles to 0 for now to reduce dimensionality.
    # 24 centers + 1 s + 1 outer_angle = 26 variables.
    
    def objective(vars):
        return vars[-2] # s is second to last?
        # Let's define order: x1,y1...x12,y12, s, outer_angle
        return vars[-2]

    def constraints(vars):
        centers = vars[:24].reshape(12, 2)
        s = vars[-2]
        o_ang = vars[-1]
        
        cons = []
        
        # Separation
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                cons.append(dist - min_dist + 1e-6)
        
        # Containment
        o_verts = get_hex_verts(0, 0, s, o_ang)
        for i in range(12):
            cx, cy = centers[i]
            # Angle 0
            verts = get_hex_verts(cx, cy, 1.0, 0.0)
            for v in verts:
                for k in range(6):
                    p1 = o_verts[k]
                    p2 = o_verts[(k+1)%6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    cons.append(cross + 1e-6)
        
        return cons

    x0 = np.zeros(26)
    # Initial guess
    x0[:24] = known_valid_packing()[0][:, :2].flatten()
    x0[-2] = known_valid_packing()[2]
    x0[-1] = known_valid_packing()[1][2]
    
    # Optimize
    try:
        res = minimize(objective, x0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': constraints}],
                       bounds=[(None, None)] * 24 + [(3.0, 10.0), (0.0, 60.0)],
                       options={'maxiter': 500, 'ftol': 1e-8})
        
        if res.success:
            centers = res.x[:24].reshape(12, 2)
            s = res.x[-2]
            o_ang = res.x[-1]
            inner_data = np.column_stack((centers, np.zeros(12)))
            outer_data = np.array([0.0, 0.0, o_ang])
            
            if check_validity(inner_data, outer_data, s):
                if s < known_valid_packing()[2]:
                    return inner_data, outer_data, s
    except:
        pass

    return inner, outer, s

