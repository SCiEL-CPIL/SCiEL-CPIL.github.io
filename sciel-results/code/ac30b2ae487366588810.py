"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Pre-compute vertices for unit regular hexagon
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        """Get vertices of a regular hexagon with given center, rotation, and side length."""
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + side * (base_vertices @ rot.T)
    
    def polygon_area(vertices):
        """Shoelace formula for polygon area."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0
    
    def hexagon_containment_penalty(inner_verts, outer_verts):
        """
        Penalty for inner hexagon not being contained in outer hexagon.
        Uses distance from inner vertices to outer edges.
        """
        penalty = 0.0
        n_out = len(outer_verts)
        for i in range(6):
            # For each vertex of inner hex, check if it's inside outer
            for v in inner_verts:
                for j in range(n_out):
                    v1 = outer_verts[j]
                    v2 = outer_verts[(j + 1) % n_out]
                    # Edge from v1 to v2
                    edge = v2 - v1
                    # Normal pointing inward (perpendicular to edge)
                    normal = np.array([-edge[1], edge[0]])
                    # Normalize
                    norm = np.linalg.norm(normal)
                    if norm > 1e-10:
                        normal = normal / norm
                    # Check if normal points inward
                    # Use a point known to be inside (centroid of outer)
                    outer_centroid = np.mean(outer_verts, axis=0)
                    if np.dot(normal, outer_centroid - v1) < 0:
                        normal = -normal
                    # Distance from v to the line, positive if outside
                    dist = np.dot(v - v1, normal)
                    if dist < 0:
                        penalty += dist ** 2
        return penalty
    
    def min_distance_penalty(inner_verts_list, outer_verts):
        """
        Penalty for inner hexagons overlapping each other.
        Approximate by checking vertex-edge distances between pairs.
        """
        penalty = 0.0
        n = len(inner_verts_list)
        for i in range(n):
            for j in range(i + 1, n):
                vi = inner_verts_list[i]
                vj = inner_verts_list[j]
                # Check if any vertex of vi is inside vj or vice versa
                # Simplified: check if vertices are too close
                for a in vi:
                    for b in vj:
                        d = np.linalg.norm(a - b)
                        if d < 0.1:  # Threshold for potential overlap
                            penalty += (0.1 - d) ** 2
        return penalty
    
    def objective(params):
        """
        Objective: minimize outer side length + penalties for constraints.
        params: [cx, cy, angle, s, x1, y1, a1, ..., x12, y12, a12]
        """
        cx, cy, angle, s = params[0], params[1], params[2], params[3]
        
        if s < 0.5:
            return 1e10
        
        outer_verts = get_hex_vertices(np.array([cx, cy]), angle, s)
        
        penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            idx = 4 + i * 3
            xi, yi, ai = params[idx], params[idx + 1], params[idx + 2]
            inner_verts = get_hex_vertices(np.array([xi, yi]), ai, 1.0)
            inner_verts_list.append(inner_verts)
            
            # Containment penalty
            penalty += 10.0 * hexagon_containment_penalty(inner_verts, outer_verts)
        
        # Overlap penalty between inner hexagons
        penalty += 10.0 * min_distance_penalty(inner_verts_list, outer_verts)
        
        return s + penalty
    
    best_result = None
    best_score = float('inf')
    
    # Multiple random restarts
    np.random.seed(42)
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize
        # Outer hexagon: center at origin, angle 0, side length ~3-4
        cx, cy = 0.0, 0.0
        angle = 0.0
        s = 3.5 + np.random.random() * 0.5
        
        params = [cx, cy, angle, s]
        
        # Inner hexagons: place in a grid-like pattern
        for i in range(N):
            # Spread out in a hexagonal pattern
            row = i // 4
            col = i % 4
            xi = (col - 1.5) * 2.2
            yi = (row - 1.0) * 1.9
            ai = 0.0
            params.extend([xi, yi, ai])
        
        # Optimize
        try:
            res = minimize(
                objective, 
                np.array(params), 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success or res.fun < best_score:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
                    
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback: return a simple valid configuration
        inner = []
        for i in range(N):
            row = i // 4
            col = i % 4
            inner.append([(col - 1.5) * 2.2, (row - 1.0) * 1.9, 0.0])
        return inner, (0.0, 0.0), 4.0, 0.0
    
    # Extract result
    cx, cy, angle, s = best_result[0], best_result[1], best_result[2], best_result[3]
    
    inner = []
    for i in range(N):
        idx = 4 + i * 3
        xi, yi, ai = best_result[idx], best_result[idx + 1], best_result[idx + 2]
        inner.append([xi, yi, ai])
    
    return inner, (cx, cy), s, angle
# EVOLVE-BLOCK-END
