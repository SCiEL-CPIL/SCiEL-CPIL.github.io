# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_rigid_cluster_centers():
    """
    Returns the Cartesian coordinates of 12 unit hexagons in a fixed, valid tiling.
    Uses axial coordinates for a compact cluster.
    """
    # Axial coordinates (q, r)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (-2, 1), (-2, 0)
    ]
    
    centers = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        centers.append([x, y])
    
    centers = np.array(centers)
    # Center the cluster at origin
    centers = centers - centers.mean(axis=0)
    return centers

def _compute_s_rigid(theta, dx, dy, phi):
    """
    Compute the minimum side length s of the outer hexagon (centered at 0,0, angle phi)
    containing the rigid cluster rotated by theta and shifted by (dx, dy).
    """
    centers = _get_rigid_cluster_centers()
    
    # Rotate centers
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_centers = centers @ rot_mat.T
    
    # Shift centers
    shifted_centers = rotated_centers + np.array([dx, dy])
    
    # Get all vertices of the 12 hexagons
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Rotate base vertices by theta
    rotated_base_verts = base_verts @ rot_mat.T
    
    # Transform base vertices to each center
    all_verts = shifted_centers[:, np.newaxis, :] + rotated_base_verts[np.newaxis, :, :]
    all_verts = all_verts.reshape(72, 2)
    
    # Compute normals of outer hexagon
    phi_rad = np.radians(phi)
    normal_angles = phi_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Compute max dot product for each normal
    dots = all_verts @ normals.T
    max_dots = dots.max(axis=0)
    
    s = np.max(max_dots) / (np.sqrt(3)/2)
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    def objective(params):
        theta, dx, dy, phi = params
        return _compute_s_rigid(theta, dx, dy, phi)
    
    best_params = np.array([0.0, 0.0, 0.0, 0.0])
    best_s = objective(best_params)
    
    # Restart with different initial angles
    for theta_init in np.linspace(0, np.pi/3, 7):
        for phi_init in np.linspace(0, np.pi/3, 7):
            if time.time() > deadline - 10:
                break
            
            x0 = np.array([theta_init, 0.0, 0.0, phi_init])
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < best_s:
                    best_s = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if time.time() > deadline - 10:
            break

    theta, dx, dy, phi = best_params
    
    centers = _get_rigid_cluster_centers()
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_centers = centers @ rot_mat.T
    shifted_centers = rotated_centers + np.array([dx, dy])
    
    angle_deg = np.degrees(theta)
    inner = np.column_stack([shifted_centers, np.full(12, angle_deg)])
    
    outer = np.array([0.0, 0.0, phi])
    
    s = best_s * 1.00001
    
    return inner, outer, s

# EVOLVE-BLOCK-END
