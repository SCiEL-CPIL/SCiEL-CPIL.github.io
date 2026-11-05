import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 148.0

    # Define the 3-5-3 cluster of 11 hexagons
    # Axial coordinates (q, r)
    # Row r=1: q=-1, 0, 1
    # Row r=0: q=-2, -1, 0, 1, 2
    # Row r=-1: q=-1, 0, 1
    
    axial_coords = [
        (-1, 1), (0, 1), (1, 1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, -1), (0, -1), (1, -1)
    ]
    
    # Convert axial to Cartesian
    # x = 1.5 * q + sqrt(3)/2 * r
    # y = sqrt(3) * r
    centers = np.array([
        [1.5 * q + np.sqrt(3)/2 * r, np.sqrt(3) * r]
        for q, r in axial_coords
    ])
    
    n_hex = len(centers)
    
    # Base vertices of a unit hexagon (side 1)
    # Pointy-top orientation
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])
    
    def get_all_vertices(theta_inner):
        """
        Calculate all vertices of the 11 hexagons.
        theta_inner: rotation of the inner hexagons.
        """
        c = np.cos(theta_inner)
        s = np.sin(theta_inner)
        rot_inner = np.array([[c, -s], [s, c]])
        rotated_verts = base_vertices @ rot_inner.T
        
        all_vertices = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            all_vertices[i*6:(i+1)*6] = rotated_verts + centers[i]
        return all_vertices

    def min_side_for_theta(theta_inner):
        """
        Find minimum outer side length over all outer rotations.
        """
        verts = get_all_vertices(theta_inner)
        
        def side_for_outer(theta_outer):
            # Normals of the outer hexagon sides
            # The outer hexagon is regular.
            # Its sides are at angles theta_outer + 30 + k*60.
            # The normal to a side is at angle theta_outer + 30 + k*60.
            normals = np.array([
                [np.cos(np.deg2rad(30 + k*60) + theta_outer),
                 np.sin(np.deg2rad(30 + k*60) + theta_outer)]
                for k in range(6)
            ])
            dots = verts @ normals.T
            max_dot = np.max(dots)
            # Side length S = max_dot / (sqrt(3)/2)
            return max_dot / (np.sqrt(3) / 2)
        
        # Sample outer angles
        best_s = float('inf')
        best_t = 0.0
        for i in range(12):
            t = i * np.pi / 6
            s = side_for_outer(t)
            if s < best_s:
                best_s = s
                best_t = t
        
        # Refine
        res = minimize(side_for_outer, best_t, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
        if res.fun < best_s:
            best_s = res.fun
            best_t = res.x[0]
            
        return best_s, best_t

    # Optimize theta_inner
    # The cluster is symmetric, so theta_inner and theta_inner + 60 are equivalent.
    # We can search in [0, pi/3].
    
    best_score = float('inf')
    best_params = None
    best_outer = 0.0
    
    # Multi-start for theta_inner
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
        if i == 0:
            x_start = np.array([0.0])
        else:
            x_start = np.array([i * np.pi / (3 * n_starts)])
            
        try:
            # Objective: min side length
            def obj(params):
                s, _ = min_side_for_theta(params[0])
                return s
            
            res = minimize(obj, x_start, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                _, best_outer = min_side_for_theta(best_params[0])
        except Exception:
            pass

    if best_params is None:
        # Fallback
        return _fallback_solution()

    theta_inner = best_params[0]
    
    # Final precise evaluation
    s, t = min_side_for_theta(theta_inner)
    best_score = s
    best_outer = t
    
    # Construct inner list
    c = np.cos(theta_inner)
    s_val = np.sin(theta_inner)
    rot_inner = np.array([[c, -s_val], [s_val, c]])
    rotated_verts = base_vertices @ rot_inner.T # Not needed for centers
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(theta_inner)])
        
    side = best_score + 1e-9
    return inner, (0.0, 0.0), side, np.rad2deg(best_outer)

def _fallback_solution():
    # 3-5-3 cluster, no rotation
    axial_coords = [
        (-1, 1), (0, 1), (1, 1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, -1), (0, -1), (1, -1)
    ]
    centers = np.array([
        [1.5 * q + np.sqrt(3)/2 * r, np.sqrt(3) * r]
        for q, r in axial_coords
    ])
    inner = [[x, y, 0.0] for x, y in centers]
    # Side length approx 4
    side = 4.0 + 1e-9
    return inner, (0.0, 0.0), side, 0.0

if __name__ == "__main__":
    construct()
