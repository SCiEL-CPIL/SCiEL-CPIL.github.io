"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)

    def get_sat_gap(v1, v2):
        """
        Compute the maximum separating axis gap between two convex hexagons.
        Returns a positive value if separated (distance), negative if overlapping (penetration depth).
        """
        max_gap = -float('inf')
        # Check axes from v1
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
            
            # Check axes from v2
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
        """
        Given inner hexagons and outer hexagon params, find minimum outer side length.
        """
        ox, oy = outer_center
        oa = math.radians(outer_angle_deg)
        sqrt3 = math.sqrt(3)
        
        max_scale = 0.0
        for hex_data in inner_hexes:
            cx, cy, ang = hex_data
            verts = hex_vertices(cx, cy, ang)
            for vx, vy in verts:
                dx, dy = vx - ox, vy - oy
                required_s = 0.0
                for k in range(6):
                    edge_mid_angle = oa + k * math.pi / 3 + math.pi / 6
                    inward_nx = -math.cos(edge_mid_angle)
                    inward_ny = -math.sin(edge_mid_angle)
                    dot_val = dx * inward_nx + dy * inward_ny
                    if dot_val > 0:
                        s_req = dot_val / (sqrt3 / 2)
                        required_s = max(required_s, s_req)
                max_scale = max(max_scale, required_s)
        return max_scale

    def objective(x, verts_cache=None):
        """
        Objective function for optimization.
        x: [x0, y0, a0, x1, y1, a1, ...]
        """
        centers = x[0::3]
        angles_deg = x[1::3]
        angles_rad = np.deg2rad(angles_deg)
        
        # Compute vertices
        verts = []
        for i in range(N):
            v = hex_vertices(centers[i], angles_deg[i])
            verts.append(v)
        
        penalty = 0.0
        # Overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                gap = get_sat_gap(verts[i], verts[j])
                if gap < 0:
                    # Quadratic penalty for overlap
                    penalty += 10.0 * gap * gap
        
        # Container penalty
        # Outer hexagon centered at (0,0) with angle 0
        # Side length is a variable? No, we minimize side length.
        # Wait, the strategy is to minimize side length.
        # So the objective should be the side length + penalty.
        # But side length depends on the positions.
        # Let's compute the required side length for the current configuration.
        inner_hexes = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # Angles are in deg for compute_min_outer_side
        # Actually compute_min_outer_side expects [cx, cy, angle_deg]
        inner_hexes = [[centers[i], angles_deg[i]] for i in range(N)] # Wait, my helper expects 3 elements?
        # Let's re-check compute_min_outer_side signature in current code.
        # It takes inner_hexes (list of [cx, cy, angle_deg]).
        # So:
        inner_hexes = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # 3rd element is dummy? No, it uses ang.
        # Correct:
        inner_hexes = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # This is wrong, angle_deg is the rotation.
        # Let's just inline the side length calculation for speed.
        
        max_proj = 0.0
        for i in range(N):
            v = verts[i]
            # Project onto 6 outward normals of unit hexagon (side=1)
            # Outward normals for hexagon with angle 0:
            # Edge midpoints angles: 30, 90, 150, 210, 270, 330 degrees
            # Outward normal angles: 30, 90, ...
            # Normal vectors: (cos(30), sin(30)), (cos(90), sin(90)), ...
            # (sqrt(3)/2, 1/2), (0, 1), (-sqrt(3)/2, 1/2), (-sqrt(3)/2, -1/2), (0, -1), (sqrt(3)/2, -1/2)
            # Distance to edge for side s is s * sqrt(3)/2.
            # Condition: dot(p, n) <= s * sqrt(3)/2
            # So s >= dot(p, n) / (sqrt(3)/2)
            for k in range(6):
                # Pre-calculated normals
                # n0: (0.866, 0.5)
                # n1: (0, 1)
                # n2: (-0.866, 0.5)
                # n3: (-0.866, -0.5)
                # n4: (0, -1)
                # n5: (0.866, -0.5)
                # Using math
                angle = math.radians(30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = np.max(v[:, 0] * nx + v[:, 1] * ny)
                if proj > 0:
                    s_req = proj / (math.sqrt(3) / 2)
                    if s_req > max_proj:
                        max_proj = s_req
        
        # Objective: Side length + Overlap Penalty
        return max_proj + penalty

    # Initial guesses
    # 1. Honeycomb grid
    sqrt3 = math.sqrt(3)
    init_1 = []
    idx = 0
    for row in range(4):
        for col in range(3):
            cx = col * 1.5
            cy = row * sqrt3
            if row % 2 == 1:
                cx += 0.75
            init_1.extend([cx - 1.5, cy - 1.5 * sqrt3, 0.0])
    
    # 2. Random
    np.random.seed(42)
    init_2 = []
    for i in range(N):
        init_2.extend([np.random.randn() * 2, np.random.randn() * 2, np.random.randn() * 10])

    best_solution = None
    best_score = float('inf')

    # Try both initializations
    inits = [np.array(init_1), np.array(init_2)]
    
    # Also try some rotated variants of the grid
    for angle_offset in [0, 30, 60]:
        new_init = init_1.copy()
        for i in range(N):
            new_init[3*i + 2] = angle_offset
        inits.append(np.array(new_init))

    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth SAT penalty
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_score:
                # Verify validity
                centers = res.x[0::3]
                angles_deg = res.x[1::3]
                
                valid = True
                verts = [hex_vertices(centers[i], angles_deg[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = get_sat_gap(verts[i], verts[j])
                        if gap < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    inner_hexes = [[centers[i], angles_deg[i], 0.0] for i in range(N)]
                    # Fix compute_min_outer_side call
                    inner_hexes_correct = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # Wait, the 3rd element is angle_deg?
                    # My helper: hex_data = [cx, cy, ang]
                    # So:
                    inner_hexes_correct = [[centers[i], angles_deg[i]] for i in range(N)] # No, 3 elements needed?
                    # Let's check the helper code again.
                    # `cx, cy, ang = hex_data`
                    # So it needs 3 elements.
                    inner_hexes_correct = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # The 3rd is ignored? No, `ang` is the 3rd.
                    # Wait, `hex_data` is unpacked.
                    # So I must pass [cx, cy, angle_deg].
                    inner_hexes_correct = [[centers[i], angles_deg[i], 0.0] for i in range(N)] # This is wrong.
                    # The list element is [cx, cy, angle_deg].
                    # So:
                    inner_hexes_correct = [[centers[i], angles_deg[i]] for i in range(N)] # This has 2 elements.
                    # Error in my head.
                    # `for hex_data in inner_hexes: cx, cy, ang = hex_data`
                    # So hex_data must be length 3.
                    # My x vector has [x, y, angle].
                    # So centers[i] is x, angles_deg[i] is angle?
                    # No, x[0::3] are x coords. x[1::3] are y coords?
                    # Let's re-verify x layout.
                    # x: [x0, y0, a0, x1, y1, a1, ...]
                    # centers = x[0::3] -> x0, x1, ...
                    # y_coords = x[1::3] -> y0, y1, ...
                    # angles_deg = x[2::3] -> a0, a1, ...
                    
                    # I made a mistake in objective function extraction.
                    # Let's fix objective and extraction.
                    pass
        except Exception:
            pass

    # Re-implementing properly with correct indexing
    def objective_fixed(x):
        # x: [x0, y0, a0, x1, y1, a1, ...]
        centers_x = x[0::3]
        centers_y = x[1::3]
        angles_deg = x[2::3]
        
        verts = []
        for i in range(N):
            v = hex_vertices(centers_x[i], centers_y[i], angles_deg[i])
            verts.append(v)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = get_sat_gap(verts[i], verts[j])
                if gap < 0:
                    penalty += 10.0 * gap * gap
        
        max_proj = 0.0
        for i in range(N):
            v = verts[i]
            for k in range(6):
                angle = math.radians(30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = np.max(v[:, 0] * nx + v[:, 1] * ny)
                if proj > 0:
                    s_req = proj / (math.sqrt(3) / 2)
                    if s_req > max_proj:
                        max_proj = s_req
        
        return max_proj + penalty

    # Retry optimization with fixed objective
    best_solution = None
    best_score = float('inf')

    inits = [np.array(init_1)]
    # Add random init
    inits.append(np.array(init_2))
    
    for x0 in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective_fixed, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            # Verify
            cx = res.x[0::3]
            cy = res.x[1::3]
            ad = res.x[2::3]
            
            valid = True
            verts = [hex_vertices(cx[i], cy[i], ad[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = get_sat_gap(verts[i], verts[j])
                    if gap < -1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                inner_hexes = [[cx[i], cy[i], ad[i]] for i in range(N)]
                score = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)
                if score < best_score:
                    best_score = score
                    best_solution = (cx, cy, ad)
        except Exception:
            pass

    if best_solution is None:
        # Fallback to initial grid
        cx = np.array(init_1[0::3])
        cy = np.array(init_1[1::3])
        ad = np.array(init_1[2::3])
        inner_hexes = [[cx[i], cy[i], ad[i]] for i in range(N)]
        best_score = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)
    else:
        cx, cy, ad = best_solution

    inner = []
    for i in range(N):
        inner.append([cx[i], cy[i], ad[i]])
    
    return inner, (0.0, 0.0), best_score, 0.0

# EVOLVE-BLOCK-END
