import numpy as np
from scipy.optimize import minimize
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 580  # seconds, safety margin
    
    def get_hex_vertices(cx, cy, d, angle_deg):
        angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
        return np.column_stack([
            cx + d * np.cos(angles),
            cy + d * np.sin(angles)
        ])
    
    def get_edge_normals(vertices):
        n = len(vertices)
        normals = np.zeros((n, 2))
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-15:
                normal /= norm
            normals[i] = normal
        return normals
    
    def separation_gap(verts1, verts2):
        normals1 = get_edge_normals(verts1)
        normals2 = get_edge_normals(verts2)
        all_normals = np.vstack([normals1, normals2])
        
        max_gap = -np.inf
        for normal in all_normals:
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 <= min2:
                gap = min2 - max1
            elif max2 <= min1:
                gap = min1 - max2
            else:
                gap = min(max1 - max2, max2 - max1)
            
            if gap > max_gap:
                max_gap = gap
        
        return max_gap
    
    def containment_gap(inner_verts, outer_verts):
        n_outer = len(outer_verts)
        min_gap = np.inf
        
        for i in range(n_outer):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % n_outer]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-15:
                normal /= norm
            
            for v in inner_verts:
                val = np.dot(v - p1, normal)
                if val < min_gap:
                    min_gap = val
        
        return min_gap
    
    def objective(params):
        inner_params = params[:36].reshape(12, 3)
        outer_x, outer_y, outer_angle = params[36:39]
        s = params[39]
        
        if s < 1.0:
            return 1e10
        
        outer_verts = get_hex_vertices(outer_x, outer_y, s, outer_angle)
        
        total_violation = 0.0
        
        for i in range(12):
            vi, yi, ai = inner_params[i]
            verts_i = get_hex_vertices(vi, yi, 1.0, ai)
            
            for j in range(i + 1, 12):
                vj, yj, aj = inner_params[j]
                verts_j = get_hex_vertices(vj, yj, 1.0, aj)
                
                gap = separation_gap(verts_i, verts_j)
                violation = max(0.0, 1e-6 - gap)
                total_violation += violation
            
            gap = containment_gap(verts_i, outer_verts)
            violation = max(0.0, -1e-6 - gap)
            total_violation += violation
        
        return s + 100.0 * total_violation
    
    sqrt3 = np.sqrt(3.0)
    
    def make_grid_initial():
        inner = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3
                y = row * 1.5
                if row % 2 == 1:
                    x += sqrt3 / 2.0
                inner.append([x, y, 0.0])
        inner = np.array(inner)
        inner[:, 0] -= np.mean(inner[:, 0])
        inner[:, 1] -= np.mean(inner[:, 1])
        s_init = np.max(np.linalg.norm(inner, axis=1)) + 2.0
        return np.concatenate([inner.flatten(), [0.0, 0.0, 0.0], [s_init]])
    
    def make_cluster_initial():
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = np.radians(60.0 * k)
            inner.append([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle), 0.0])
        for k in range(5):
            angle = np.radians(60.0 * k + 30.0)
            inner.append([2.0 * sqrt3 * np.cos(angle), 2.0 * sqrt3 * np.sin(angle), 0.0])
        inner = np.array(inner)
        inner[:, 0] -= np.mean(inner[:, 0])
        inner[:, 1] -= np.mean(inner[:, 1])
        s_init = np.max(np.linalg.norm(inner, axis=1)) + 2.0
        return np.concatenate([inner.flatten(), [0.0, 0.0, 0.0], [s_init]])
    
    best_params = None
    best_score = np.inf
    
    initial_guesses = [make_grid_initial(), make_cluster_initial()]
    
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        if restart < len(initial_guesses):
            x0 = initial_guesses[restart].copy()
        else:
            x0 = initial_guesses[0].copy()
            x0[:36] += np.random.randn(36) * 0.15
            x0[39] *= (1.0 + np.random.rand() * 0.15)
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                options={'maxiter': 5000, 'ftol': 1e-14}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is None:
        best_params = initial_guesses[0]
    
    inner_params = best_params[:36].reshape(12, 3)
    outer_x, outer_y, outer_angle = best_params[36:39]
    s = max(best_params[39], np.sqrt(12.0) + 1e-6)
    
    return inner_params, np.array([outer_x, outer_y, outer_angle]), s
