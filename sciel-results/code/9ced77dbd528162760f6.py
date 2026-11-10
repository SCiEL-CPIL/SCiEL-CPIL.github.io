import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590
    
    n = 12
    a = 1.0
    inradius = a * np.sqrt(3) / 2
    d = np.sqrt(3)  # distance between adjacent hex centers
    e1 = np.array([d, 0.0])
    e2 = np.array([d/2, d * np.sqrt(3)/2])
    
    def get_hex_vertices(center, angle_deg):
        angle = np.radians(angle_deg)
        thetas = angle + np.arange(6) * np.pi / 3
        return center + a * np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    def compute_all_vertices(centers, angles):
        verts = []
        for c, ang in zip(centers, angles):
            verts.append(get_hex_vertices(c, ang))
        return np.vstack(verts)
    
    def min_hex_side_for_angle(verts, phi):
        normals = np.column_stack([
            np.cos(phi + k * np.pi/3),
            np.sin(phi + k * np.pi/3)
        ] for k in range(6))
        projs = verts @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        spans = max_projs - min_projs
        return np.max(spans) / np.sqrt(3)
    
    def objective(params):
        centers = params[:2*n].reshape(n, 2)
        angles = params[2*n:]
        verts = compute_all_vertices(centers, angles)
        
        # Optimize over phi
        result = minimize_scalar(lambda p: min_hex_side_for_angle(verts, p),
                                bounds=(0, np.pi/3), method='bounded',
                                options={'xatol': 1e-8})
        return result.fun
    
    def check_separation(centers, angles, tol=1e-6):
        """Check that no two inner hexagons overlap using SAT"""
        for i in range(n):
            for j in range(i+1, n):
                verts_i = get_hex_vertices(centers[i], angles[i])
                verts_j = get_hex_vertices(centers[j], angles[j])
                
                # Get edge normals for hex i
                for k in range(6):
                    v1 = verts_i[k]
                    v2 = verts_i[(k+1) % 6]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    
                    if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                        return True  # Separated along this axis
                # If we get here, check hex j's normals too (redundant for convex shapes but safe)
                # Actually for two convex polygons, checking all normals from both is sufficient
                # But for hexagons, checking normals from one is enough if they're parallel-sided
                # Let's also check j's normals to be safe
                found_separation = False
                for k in range(6):
                    v1 = verts_j[k]
                    v2 = verts_j[(k+1) % 6]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    
                    if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                        found_separation = True
                        break
                if not found_separation:
                    return False
        return True
    
    # Initial guess
    init_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (1, 1), (-1, -1),
        (2, 0), (-2, 0), (0, 2)
    ]
    init_centers = np.array([i * e1 + j * e2 for i, j in init_coords])
    init_angles = np.zeros(n)
    x0 = np.concatenate([init_centers.flatten(), init_angles])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if r == 0:
            x = x0.copy()
        else:
            scale = 0.2 * (1 + 0.05 * r)
            x = x0 + np.random.randn(3*n) * scale
        
        try:
            result = minimize(objective, x, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    centers = best_x[:2*n].reshape(n, 2)
    angles = best_x[2*n:]
    
    # Find optimal outer hex
    verts = compute_all_vertices(centers, angles)
    phi_result = minimize_scalar(lambda p: min_hex_side_for_angle(verts, p),
                                bounds=(0, np.pi/3), method='bounded',
                                options={'xatol': 1e-10})
    best_phi = phi_result.x
    s = phi_result.fun
    
    # Compute center
    normals = np.column_stack([
        np.cos(best_phi + k * np.pi/3),
        np.sin(best_phi + k * np.pi/3)
    ] for k in range(6))
    projs = verts @ normals.T
    max_projs = np.max(projs, axis=0)
    min_projs = np.min(projs, axis=0)
    A = normals[:3]
    b = (max_projs[:3] + min_projs[:3]) / 2
    center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([center[0], center[1], np.degrees(best_phi)])
    
    return inner_hex_data, outer_hex_data, s
