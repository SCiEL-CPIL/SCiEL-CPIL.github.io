"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        # Transform vertices to container's local coordinate system
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                # Translate
                lx = v[0] - ocx
                ly = v[1] - ocy
                # Rotate by -oangle to align container axes
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                # Project onto 6 normals
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """
        Compute the outer side length S and penalty for overlaps.
        params layout:
        [0:3]   outer center x, y, angle
        [3:3+3N] inner hexagons (cx, cy, angle)
        """
        ocx = params[0]
        ocy = params[1]
        oangle = params[2]
        
        vertices_list = []
        for i in range(N):
            cx = params[3 + 3 * i]
            cy = params[3 + 3 * i + 1]
            angle = params[3 + 3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0: # Only check SAT if potentially overlapping
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        # Strong penalty to push apart
                        penalty += 1000.0 * (2.0 - dist) ** 2
        
        return S + penalty

    def get_initial_params():
        p = np.zeros(3 + 3 * N)
        # Outer
        p[0:3] = [0, 0, 0]
        
        # Inner
        idx = 0
        # Center
        p[3 + 3*idx : 3 + 3*idx + 3] = [0, 0, 0]
        idx += 1
        # Ring 1
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        # Ring 2 (4 corners of the hexagon shape)
        angles_r2 = [0, 120, 240, 180]
        for deg in angles_r2:
            theta = np.radians(deg)
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        return p

    best_params = get_initial_params()
    best_score = objective(best_params)
    
    # Multi-start optimization
    np.random.seed(42)
    for i in range(20):
        p = best_params.copy()
        if i > 0:
            # Perturb inner positions and rotations, keep outer near 0
            noise_inner = np.random.normal(0, 0.5, size=3 * N)
            noise_outer = np.random.normal(0, 0.1, size=3)
            p[0:3] += noise_outer
            p[3:] += noise_inner
            
        res = minimize(objective, p, method='Powell', 
                      options={'maxiter': 1000, 'maxfev': 5000})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    # Extract results
    ocx, ocy, oangle = best_params[0], best_params[1], best_params[2]
    inner = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Recalculate S for the best params to be safe
    vertices_list = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
        
    S_final = compute_S_for_container(vertices_list, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S_final, np.degrees(oangle)
# EVOLVE-BLOCK-END
