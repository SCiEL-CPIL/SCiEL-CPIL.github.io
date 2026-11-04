"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def sat_overlap_penalty(v1, v2, tol=1e-9):
        """
        Calculate a smooth penalty for overlap between two convex polygons.
        Returns 0 if disjoint, > 0 if overlapping.
        Uses the 12 edge normals from both polygons.
        """
        penalty = 0.0
        # Get all edge normals from both polygons
        # Poly 1 normals
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 + tol < min2 or max2 + tol < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
        
        # Poly 2 normals
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 + tol < min2 or max2 + tol < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
                
        return penalty

    def compute_S(vertices_list):
        """
        Compute the minimum side length S of the outer hexagon 
        centered at (0, 0) with rotation 0 to contain all vertices.
        """
        apothem = np.sqrt(3) / 2
        max_proj = 0.0
        
        # Normals for the container hexagon (fixed at 0)
        # Base normals at 30, 90, 150... degrees
        for k in range(6):
            theta_n = np.pi / 6 + k * np.pi / 3
            cos_n = np.cos(theta_n)
            sin_n = np.sin(theta_n)
            
            # Project all vertices of all inner hexagons
            # To be faster, we can project each hexagon's vertices
            max_rel = -np.inf
            min_rel = np.inf
            for verts in vertices_list:
                projs = verts[:, 0] * cos_n + verts[:, 1] * sin_n
                max_rel = max(max_rel, np.max(projs))
                min_rel = min(min_rel, np.min(projs))
            
            extent = max(abs(max_rel), abs(min_rel))
            if extent > max_proj:
                max_proj = extent
        
        return max_proj / apothem

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN]
        """
        if time.time() > deadline:
            return float('inf')
            
        vertices_list = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S(vertices_list)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick rejection based on center distance
                c1 = np.array([params[3*i], params[3*i+1]])
                c2 = np.array([params[3*j], params[3*j+1]])
                if np.linalg.norm(c1 - c2) > 2.5:
                    continue
                p = sat_overlap_penalty(vertices_list[i], vertices_list[j])
                penalty += p
        
        return S + 100.0 * penalty

    # Initialize with hexagonal lattice
    np.random.seed(42)
    params = np.zeros(3 * N)
    
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1
    r1 = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        cx = r1 * np.cos(theta)
        cy = r1 * np.sin(theta)
        idx = 1 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Ring 2 (4 more)
    # Generate 12 positions for ring 2
    r2 = 2 * np.sqrt(3)
    # Lattice positions for ring 2 are at angles 0, 30, 60... ? No.
    # In a hexagonal lattice, the 12 neighbors of ring 2 are at:
    # 4 "corners" (aligned with ring 1) and 8 "edges"
    # Let's just use 4 positions at 45, 135, 225, 315 degrees?
    # Or better, use the lattice positions.
    # Lattice vectors u = (sqrt(3), 0), v = (sqrt(3)/2, 3/2)
    # Ring 2 positions:
    # 2u, 2v, 2v-2u, -2u, -2v, 2u-2v, u+v, u-v, -u-v, -u+v, v-u, -v+u
    # Let's pick 4 symmetric ones: 2u, -2u, 2v, -2v?
    # 2u = (2sqrt(3), 0)
    # -2u = (-2sqrt(3), 0)
    # 2v = (sqrt(3), 3)
    # -2v = (-sqrt(3), -3)
    # These are 4 points.
    
    pos_ring2 = [
        (2*np.sqrt(3), 0),
        (-2*np.sqrt(3), 0),
        (np.sqrt(3), 3),
        (-np.sqrt(3), -3)
    ]
    
    for k in range(4):
        cx, cy = pos_ring2[k]
        idx = 7 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0

    current_params = params.copy()
    current_energy = objective(current_params)
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Optimization
    # Use Nelder-Mead with tight tolerances
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract results
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle))

    # Final S calculation
    S = compute_S(vertices_list)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
