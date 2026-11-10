import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 570  # 570s limit, keep margin
    
    n = 12
    sqrt3 = np.sqrt(3)
    
    def hex_vertices(cx, cy, angle_deg, d=1.0):
        """Get vertices of a regular hexagon."""
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([
            cx + d * np.cos(angles),
            cy + d * np.sin(angles)
        ])
    
    def sat_separation(v1, v2, tol=1e-6):
        """
        Check if two convex polygons are separated using SAT.
        Returns (is_separated, min_gap) where min_gap is negative if overlapping.
        """
        min_gap = float('inf')
        for verts in [v1, v2]:
            for i in range(len(verts)):
                p1 = verts[i]
                p2 = verts[(i + 1) % len(verts)]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-15:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if gap > min_gap:
                    min_gap = gap
                if gap > tol:
                    return True, gap
        
        return False, min_gap
    
    def get_all_vertices(params):
        """Get all vertices for all inner hexagons."""
        vertices = []
        for i in range(n):
            cx, cy, angle = params[i * 3], params[i * 3 + 1], params[i * 3 + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices.append(verts)
        return vertices
    
    def compute_min_s(params):
        """Compute minimum outer hexagon side length."""
        vertices = get_all_vertices(params)
        max_s = 0.0
        for verts in vertices:
            for v in verts:
                x, y = v
                proj0 = abs(x)
                proj60 = abs(x * 0.5 + y * (sqrt3 / 2))
                proj120 = abs(x * (-0.5) + y * (sqrt3 / 2))
                s_req = max(proj0, proj60, proj120) * (2.0 / sqrt3)
                if s_req > max_s:
                    max_s = s_req
        return max_s
    
    def objective(params):
        """Minimize s + overlap penalty."""
        if time.time() > deadline - 10:
            return 1000.0
            
        vertices = get_all_vertices(params)
        s = compute_min_s(params)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sep, gap = sat_separation(vertices[i], vertices[j], tol=1e-8)
                if not sep:
                    # Quadratic penalty for overlap
                    penalty += max(0, -gap) ** 2 * 50.0
        
        return s + penalty
    
    def is_valid(params):
        """Check validity."""
        vertices = get_all_vertices(params)
        
        for i in range(n):
            for j in range(i + 1, n):
                sep, _ = sat_separation(vertices[i], vertices[j], tol=1e-6)
                if not sep:
                    return False
        
        s = compute_min_s(params)
        limit = (sqrt3 / 2) * s
        for verts in vertices:
            for v in verts:
                x, y = v
                if abs(x) > limit + 1e-6:
                    return False
                if abs(x * 0.5 + y * (sqrt3 / 2)) > limit + 1e-6:
                    return False
                if abs(x * (-0.5) + y * (sqrt3 / 2)) > limit + 1e-6:
                    return False
        
        return True
    
    def make_initial_config(seed_offset=0):
        """Create initial configuration with hexagonal lattice."""
        rng = np.random.RandomState(42 + seed_offset)
        params = np.zeros(n * 3)
        
        # Hexagonal lattice positions
        positions = [
            (0, 0),
            (sqrt3, 0), (-sqrt3, 0),
            (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
            (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
            (2*sqrt3, 0), (-2*sqrt3, 0),
            (sqrt3, 3), (-sqrt3, 3),
            (0, -3), (0, 3),
        ]
        
        for i, (x, y) in enumerate(positions):
            params[i*3] = x + rng.normal(0, 0.2)
            params[i*3+1] = y + rng.normal(0, 0.2)
            params[i*3+2] = rng.uniform(-30, 30)
        
        return params
    
    best_params = None
    best_s = float('inf')
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        params0 = make_initial_config(restart)
        
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                bounds=[(-6, 6)] * n + [(-180, 180)] * n,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_s:
                if is_valid(result.x):
                    s_val = compute_min_s(result.x)
                    if s_val < best_s:
                        best_s = s_val
                        best_params = result.x.copy()
                        
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        positions = [
            (0, 0), (sqrt3, 0), (-sqrt3, 0),
            (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
            (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
            (2*sqrt3, 0), (-2*sqrt3, 0),
            (sqrt3, 3), (-sqrt3, 3),
            (0, -3), (0, 3),
        ]
        inner_hex_data = np.array([[x, y, 0] for x, y in positions])
        s_final = compute_min_s(inner_hex_data.reshape(n * 3)) + 1e-5
        return inner_hex_data, np.array([0.0, 0.0, 0.0]), s_final
    
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = best_params[i*3:i*3+3]
    
    s_final = compute_min_s(best_params) + 1e-5
    return inner_hex_data, np.array([0.0, 0.0, 0.0]), s_final
