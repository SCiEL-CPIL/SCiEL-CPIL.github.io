import math
import time
import numpy as np
import random

N = 12
DEADLINE = 145.0  # Safety margin
SQRT3 = math.sqrt(3.0)

def construct():
    t0 = time.time()
    
    # --- Geometry Helpers ---
    
    def get_hex_vertices(cx, cy, ang_deg):
        """Returns (6, 2) array of vertices for a unit hexagon."""
        a = np.deg2rad(ang_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))

    def calc_outer_side(vertices, phi_deg):
        """
        Calculate the side length of the minimal regular hexagon 
        with rotation phi_deg that contains all points in vertices (N*6, 2).
        """
        phi = np.deg2rad(phi_deg)
        # Face normals of the outer hexagon
        # A regular hexagon has 6 faces. The normals are at 30, 90, 150, 210, 270, 330 + phi
        # We only need 3 unique directions (opposite normals give the min)
        angs = phi + np.pi / 6 + np.array([0.0, np.pi / 3, 2.0 * np.pi / 3])
        dirs = np.column_stack((np.cos(angs), np.sin(angs)))
        
        # Project all vertices onto these 3 directions
        proj = vertices @ dirs.T  # (N*6, 3)
        pmax = proj.max(axis=0)
        pmin = proj.min(axis=0)
        ranges = pmax - pmin
        
        # The side length s is related to the range R by R = s * sqrt(3)
        # So s = R / sqrt(3)
        side = float(np.max(ranges) / SQRT3)
        
        # Find center (optional for cost, but good for return)
        # mids = (pmax + pmin) / 2
        # c, _, _, _ = np.linalg.lstsq(dirs, mids, rcond=None)
        
        return side

    def check_overlap_penalty(v1, v2, dist):
        """
        Calculate overlap penalty between two hexagons.
        v1, v2: (6, 2) arrays of vertices.
        dist: distance between centers.
        Returns 0 if disjoint, else squared penetration depth.
        """
        if dist >= 2.0:
            return 0.0
        
        # Get normals for SAT
        # Normals of hex 1
        n1 = v1[1] - v1[0]
        n1 = np.array([-n1[1], n1[0]])
        n1 /= np.linalg.norm(n1)
        
        # Actually, we need all 6 normals for each.
        # It's faster to precompute the normals for the unit hexagon 
        # and rotate them.
        
        # Let's use the vertices to get normals.
        # For a convex polygon, the normals are perpendicular to the edges.
        
        # To speed up, we can just check the 12 potential separating axes.
        # Axes: 6 from v1, 6 from v2.
        
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                axes.append(normal / norm)
                
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                axes.append(normal / norm)
                
        min_penetration = float('inf')
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_penetration:
                min_penetration = overlap
                
        if min_penetration < 0:
            return 0.0
            
        return min_penetration ** 2

    # --- State Management ---
    
    # State: [x1, y1, a1, ..., x12, y12, a12, phi]
    # 12 * 3 + 1 = 37 parameters
    
    def init_state():
        # 4x3 grid
        xs = []
        ys = []
        angles = []
        
        # Grid parameters
        dx = SQRT3
        dy = 1.5
        
        # 4 columns, 3 rows
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                xs.append(x)
                ys.append(y)
                angles.append(0.0)
        
        # Center the grid
        cx = np.mean(xs)
        cy = np.mean(ys)
        for i in range(12):
            xs[i] -= cx
            ys[i] -= cy
            
        state = []
        for i in range(12):
            state.extend([xs[i], ys[i], angles[i]])
        state.append(0.0) # phi
        
        return np.array(state)

    def get_cost(state, lambda_penalty=10.0):
        if time.time() - t0 > DEADLINE:
            return float('inf')
            
        # Unpack
        verts = []
        centers = []
        for i in range(12):
            x = state[i*3]
            y = state[i*3+1]
            a = state[i*3+2]
            centers.append((x, y))
            verts.append(get_hex_vertices(x, y, a))
            
        all_verts = np.vstack(verts)
        
        phi = state[36]
        
        # Box cost
        side = calc_outer_side(all_verts, phi)
        
        # Overlap cost
        overlap = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.hypot(dx, dy)
                if dist < 2.0:
                    overlap += check_overlap_penalty(verts[i], verts[j], dist)
                    
        return side + lambda_penalty * overlap

    # --- Simulated Annealing ---
    
    best_state = init_state()
    best_cost = get_cost(best_state)
    
    current_state = best_state.copy()
    current_cost = best_cost
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    lambda_pen = 10.0
    
    while time.time() - t0 < DEADLINE:
        # Generate neighbor
        new_state = current_state.copy()
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Move a hexagon center
            idx = random.randint(0, 11)
            new_state[idx*3] += random.gauss(0, 0.5)
            new_state[idx*3+1] += random.gauss(0, 0.5)
        elif move_type < 0.8:
            # Rotate a hexagon
            idx = random.randint(0, 11)
            new_state[idx*3+2] += random.gauss(0, 10.0)
        else:
            # Rotate outer hexagon
            new_state[36] += random.gauss(0, 5.0)
            
        new_cost = get_cost(new_state, lambda_pen)
        
        # Acceptance
        if new_cost < current_cost:
            current_state = new_state
            current_cost = new_cost
            if new_cost < best_cost:
                best_state = new_state
                best_cost = new_cost
        else:
            delta = new_cost - current_cost
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_state = new_state
                    current_cost = new_cost
                    
        # Cool down
        T *= alpha
        
        # Adaptive lambda
        # If current solution has no overlap, we can reduce lambda to focus on size
        # If it has overlap, increase lambda
        # This is a heuristic.
        
    # --- Final Optimization ---
    # Use Nelder-Mead on the best state found by SA
    from scipy.optimize import minimize
    
    def opt_obj(p):
        return get_cost(p, lambda_pen=1000.0) # High penalty for final check
        
    try:
        res = minimize(opt_obj, best_state, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_cost:
            best_state = res.x
            best_cost = res.fun
    except Exception:
        pass
        
    # Extract result
    inner = []
    for i in range(12):
        inner.append([float(best_state[i*3]), float(best_state[i*3+1]), float(best_state[i*3+2])])
        
    phi = float(best_state[36])
    
    # Calculate final outer center
    # We need to find the center of the outer hexagon.
    # Re-calculate projections
    verts = []
    for i in range(12):
        verts.append(get_hex_vertices(inner[i][0], inner[i][1], inner[i][2]))
    all_verts = np.vstack(verts)
    
    phi_rad = np.deg2rad(phi)
    angs = phi_rad + np.pi / 6 + np.array([0.0, np.pi / 3, 2.0 * np.pi / 3])
    dirs = np.column_stack((np.cos(angs), np.sin(angs)))
    proj = all_verts @ dirs.T
    pmax = proj.max(axis=0)
    pmin = proj.min(axis=0)
    mids = (pmax + pmin) / 2.0
    c, _, _, _ = np.linalg.lstsq(dirs, mids, rcond=None)
    
    outer_center = (float(c[0]), float(c[1]))
    side = float(np.max(pmax - pmin) / SQRT3)
    
    return inner, outer_center, side, phi

# EVOLVE-BLOCK-END
