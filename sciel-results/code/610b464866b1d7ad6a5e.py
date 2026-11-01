"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def hexagon_in_hexagon(inner_verts, outer_center, outer_side, outer_angle):
        """Check if inner hexagon is inside outer hexagon using SAT"""
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        
        # Get all edges from both hexagons
        all_edges = []
        for i in range(6):
            all_edges.append(inner_verts[i] - inner_verts[(i+1) % 6])
            all_edges.append(outer_verts[i] - outer_verts[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) == 0:
                continue
            normal = normal / np.linalg.norm(normal)
            
            # Project all vertices
            inner_proj = np.dot(inner_verts, normal)
            outer_proj = np.dot(outer_verts, normal)
            
            inner_min, inner_max = np.min(inner_proj), np.max(inner_proj)
            outer_min, outer_max = np.min(outer_proj), np.max(outer_proj)
            
            # Check for separation
            if inner_max < outer_min or outer_max < inner_min:
                return False
        return True
    
    def hexagons_disjoint(h1, h2):
        """Check if two hexagons are disjoint using SAT"""
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) == 0:
                continue
            normal = normal / np.linalg.norm(normal)
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return True
        return False
    
    def is_valid(inner, center, side, angle):
        """Check if configuration is valid"""
        outer_center = np.array(center)
        outer_angle = angle
        
        for i in range(N):
            # Check inner hexagon is inside outer
            v_inner = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            if not hexagon_in_hexagon(v_inner, center, side, angle):
                return False
        
        # Check all inner hexagons are disjoint
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    return False
        
        return True
    
    def objective(params):
        """Objective: minimize outer side length"""
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            inner.append([x, y, angle])
        
        center = [params[3*N], params[3*N+1]]
        side = params[3*N+2]
        angle = params[3*N+3]
        
        if not is_valid(inner, center, side, angle):
            return 1e6
        
        return side
    
    # Initial guess: hexagonal packing
    def initial_guess():
        params = np.zeros(3*N + 4)
        # Place hexagons in a hexagonal pattern
        positions = []
        # Center
        positions.append((0, 0, 0))
        # Ring 1: 6 hexagons at distance 2
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        # Ring 2: 5 more to make 12
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        # Outer hexagon
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 8.0  # Initial guess for side
        params[3*N+3] = 0.0
        return params
    
    best_params = initial_guess()
    best_score = objective(best_params)
    
    # Try multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Perturb initial guess
        params = best_params.copy()
        if r > 0:
            noise = np.random.normal(0, 1.0, size=len(params))
            noise[3*N+2] = 0  # Don't perturb side
            params += noise
        
        # Optimize
        try:
            result = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    side = float(best_params[3*N+2])
    angle = float(best_params[3*N+3])
    
    # Final validation
    if not is_valid(inner, center, side, angle):
        # Return a valid fallback
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 100.0
        angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
