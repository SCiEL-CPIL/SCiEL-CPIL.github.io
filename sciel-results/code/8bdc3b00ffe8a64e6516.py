"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def sat_overlap(v1, v2, eps=1e-9):
        """
        Check if two convex polygons overlap using Separating Axis Theorem.
        v1, v2: (6,2) arrays of vertices.
        Returns True if they overlap (interior intersection).
        """
        polys = [v1, v2]
        for poly in polys:
            for i in range(6):
                p1 = poly[i]
                p2 = poly[(i + 1) % 6]
                edge = p2 - p1
                # Normal to edge
                normal = np.array([-edge[1], edge[0]])
                nlen = np.linalg.norm(normal)
                if nlen < eps:
                    continue
                normal /= nlen
                
                # Project both polygons onto normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                # Check for gap
                if max1 < min2 + eps or max2 < min1 + eps:
                    return False  # Separating axis found, no overlap
        
        return True  # No separating axis, they overlap

    def compute_min_enclosing_hex_side(all_verts, ocx, ocy, oangle):
        """
        Compute the side length S of the outer hexagon (centered at ocx,ocy, rotated by oangle)
        that just encloses all vertices in all_verts.
        """
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        
        # Transform all vertices to outer hexagon's local frame
        lx = all_verts[:, 0] - ocx
        ly = all_verts[:, 1] - ocy
        rx = lx * cos_a + ly * sin_a
        ry = -lx * sin_a + ly * cos_a
        
        # The 6 normals for a regular hexagon (pointing outward from center to edge midpoints)
        # Edge normals are at angles pi/6 + k*pi/3 for k=0..5
        theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
        cos_tn = np.cos(theta_n)
        sin_tn = np.sin(theta_n)
        
        # Project all points onto each normal
        # proj[i, k] = rx_i * cos_tn_k + ry_i * sin_tn_k
        proj = rx[:, np.newaxis] * cos_tn[np.newaxis, :] + ry[:, np.newaxis] * sin_tn[np.newaxis, :]
        
        # For each normal direction, find max absolute projection
        max_proj = np.abs(proj).max(axis=0)
        
        # The apothem (inradius) of the outer hexagon must be at least max(max_proj)
        apothem = max_proj.max()
        
        # Side length S = apothem / (sqrt(3)/2)
        S = apothem / (np.sqrt(3) / 2)
        
        return S

    def objective(params):
        """
        params: [dx_0, dy_0, ang_0, dx_1, dy_1, ang_1, ..., ocx, ocy, oangle]
        Total: 3*N + 3
        """
        n = N
        # Extract inner hexagon params
        dx = params[0:3*n:3]
        dy = params[1:3*n:3]
        angles = params[2:3*n:3]
        
        # Extract outer hexagon params
        ocx = params[3*n]
        ocy = params[3*n + 1]
        oangle = params[3*n + 2]
        
        # Compute positions
        cx = lattice_positions[:, 0] + dx
        cy = lattice_positions[:, 1] + dy
        
        # Compute vertices for all inner hexagons
        vertices_list = []
        for i in range(n):
            verts = hex_vertices(cx[i], cy[i], angles[i])
            vertices_list.append(verts)
        
        # Stack all vertices
        all_verts = np.vstack(vertices_list)
        
        # Compute outer hexagon side length
        S = compute_min_enclosing_hex_side(all_verts, ocx, ocy, oangle)
        
        # Compute overlap penalty using SAT
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Quick bounding box check
                dist = np.linalg.norm([cx[i] - cx[j], cy[i] - cy[j]])
                if dist < 2.5:
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        # Compute penetration depth approximation
                        # Use average of negative distances from SAT
                        v1, v2 = vertices_list[i], vertices_list[j]
                        min_pen = float('inf')
                        for poly in [v1, v2]:
                            for k in range(6):
                                p1 = poly[k]
                                p2 = poly[(k + 1) % 6]
                                edge = p2 - p1
                                nlen = np.linalg.norm(edge)
                                if nlen < 1e-10:
                                    continue
                                normal = np.array([-edge[1], edge[0]]) / nlen
                                
                                proj1 = np.dot(v1, normal)
                                proj2 = np.dot(v2, normal)
                                
                                min1, max1 = proj1.min(), proj1.max()
                                min2, max2 = proj2.min(), proj2.max()
                                
                                overlap = min(max1, max2) - max(min1, min2)
                                if overlap > 0:
                                    min_pen = min(min_pen, overlap)
                        
                        if min_pen < float('inf'):
                            penalty += min_pen ** 2
        
        return S + 50.0 * penalty

    def refine_container(vertices_list, ocx_init, ocy_init, oangle_init):
        """Refine the outer hexagon parameters."""
        def cont_obj(params_c):
            ocx, ocy, oangle = params_c
            all_verts = np.vstack(vertices_list)
            return compute_min_enclosing_hex_side(all_verts, ocx, ocy, oangle)
        
        res = minimize(cont_obj, [ocx_init, ocy_init, oangle_init], method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-10})
        return res.x, res.fun

    # Standard hexagonal lattice positions for 11 hexagons
    # Center at (0,0), 6 neighbors at distance sqrt(3), 4 more at distance 2*sqrt(3)
    lattice_positions = [(0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        lattice_positions.append((cx, cy))
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        lattice_positions.append((cx, cy))
    
    lattice_positions = np.array(lattice_positions)
    
    # Initial parameters: zero displacements, zero angles, outer at origin with 0 rotation
    init_params = np.zeros(3 * N + 3)
    
    # Multi-start optimization
    np.random.seed(42)
    best_params = init_params.copy()
    best_score = objective(init_params)
    
    # Try several random restarts
    for start in range(12):
        if time.time() > DEADLINE:
            break
            
        if start == 0:
            p = init_params.copy()
        else:
            # Random perturbations
            p = init_params.copy()
            p[0:3*N:3] += np.random.normal(0, 0.1, N)  # dx
            p[1:3*N:3] += np.random.normal(0, 0.1, N)  # dy
            p[2:3*N:3] += np.random.uniform(-0.3, 0.3, N)  # angles
            p[3*N] += np.random.normal(0, 0.05)  # ocx
            p[3*N+1] += np.random.normal(0, 0.05)  # ocy
            p[3*N+2] += np.random.uniform(-0.1, 0.1)  # oangle
        
        res = minimize(objective, p, method='Nelder-Mead',
                      options={'maxiter': 2000, 'fatol': 1e-8, 'xatol': 1e-8, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Extract final positions
    n = N
    dx = best_params[0:3*n:3]
    dy = best_params[1:3*n:3]
    angles = best_params[2:3*n:3]
    ocx = best_params[3*n]
    ocy = best_params[3*n + 1]
    oangle = best_params[3*n + 2]
    
    cx = lattice_positions[:, 0] + dx
    cy = lattice_positions[:, 1] + dy
    
    inner = []
    vertices_list = []
    for i in range(n):
        inner.append([cx[i], cy[i], np.degrees(angles[i])])
        vertices_list.append(hex_vertices(cx[i], cy[i], angles[i]))
    
    # Final refinement of outer container
    if time.time() < DEADLINE - 5:
        cont_params, S = refine_container(vertices_list, ocx, ocy, oangle)
        ocx, ocy, oangle = cont_params
        oangle_deg = np.degrees(oangle)
        
        # Recompute S with final parameters
        all_verts = np.vstack(vertices_list)
        S = compute_min_enclosing_hex_side(all_verts, ocx, ocy, oangle)
    else:
        oangle_deg = np.degrees(oangle)
        all_verts = np.vstack(vertices_list)
        S = compute_min_enclosing_hex_side(all_verts, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S, oangle_deg
# EVOLVE-BLOCK-END
