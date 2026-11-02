"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagon_normals(vertices):
        """Returns 6 unit edge normals for a hexagon given its vertices."""
        normals = []
        for i in range(6):
            e = vertices[(i+1) % 6] - vertices[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                normals.append(n / nn)
        return normals

    def compute_overlap_penalty(h1, h2):
        """
        Computes a smooth penalty based on the sum of overlaps along all separating axes.
        Returns 0 if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        # Actually, max width of unit hex is 2*sqrt(3)/2 * 2 = sqrt(3) ~ 1.732? No.
        # Distance between opposite sides is sqrt(3). Diameter is 2.
        # So if dist > 2.0, they are definitely disjoint.
        if dist_sq > 4.0:
            return 0.0

        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        n1 = hexagon_normals(v1)
        n2 = hexagon_normals(v2)
        
        total_penalty = 0.0
        for n in n1 + n2:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap
            # Overlap exists if max1 > min2 and max2 > min1
            if max1 > min2 and max2 > min1:
                # Calculate overlap length
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > 0:
                    total_penalty += overlap
        
        return total_penalty

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        # The outer hexagon contains the inner ones.
        # The distance from center to a side is R = side * sqrt(3) / 2.
        # So side = 2 * R / sqrt(3).
        # We need to find the minimum R such that all points are inside.
        # A point p is inside a regular hexagon centered at origin with rotation angle and radius R (apothem)
        # if max_k ( dot(p, n_k) ) <= R, where n_k are the outward normals of the 6 sides.
        
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        # max_extent is the required apothem R
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Objective: minimize outer side length + penalty for overlaps."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness with smooth penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                penalty += compute_overlap_penalty(inner[i], inner[j])
        
        # Weighted penalty to ensure disjointness is prioritized
        # If penalty is large, the side length calculation might be invalid or large,
        # but we add a huge weight to force separation.
        if penalty > 1e-6:
             # Use a quadratic penalty to make it steeper
             return 1000.0 * penalty + compute_min_side_fast(inner, center, angle)
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def initial_guess():
        """Generate a valid initial honeycomb packing."""
        params = np.zeros(3*N + 3)
        
        # Honeycomb grid positions for 12 hexagons
        # Use axial coordinates for hex grid
        sqrt3 = np.sqrt(3)
        
        # Center at origin, then 6 neighbors, then 5 more in second ring
        positions = []
        # Center
        positions.append((0, 0, 0.0))
        # First ring: 6 hexagons
        # Distance between centers of touching unit hexagons is 2 * apothem = sqrt(3) * 2 ?
        # Apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3).
        # Wait, side length is 1. Apothem is sqrt(3)/2.
        # Center-to-center distance for touching is 2 * apothem = sqrt(3).
        # Let's verify.
        # Hexagon side 1. Width (flat-to-flat) is sqrt(3).
        # So distance is sqrt(3).
        for i in range(6):
            theta = i * np.pi / 3
            r = np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        # Second ring: pick 5 positions
        # Distance for second ring neighbors: 2 * sqrt(3)
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            r = 2 * np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    # Phase 1: Simulated Annealing for global search
    current_params = best_params.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    step_size = 0.3
    max_steps = 100000
    
    step_count = 0
    sa_deadline = deadline - 30.0  # Leave 30s for NM refinement
    
    while time.time() < sa_deadline and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb one or two random parameters for better exploration
        n_perturb = 1 if random.random() < 0.7 else 2
        for _ in range(n_perturb):
            idx = random.randint(0, 3*N + 2)
            current_step = step_size * (0.5 + 0.5 * T)
            new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        # Accept
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Phase 2: Nelder-Mead local refinement
    if time.time() < deadline - 5.0:
        def nm_objective(x):
            return objective(x)
        
        result = minimize(nm_objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun

        # If time remains, try another NM from a slightly perturbed start
        if time.time() < deadline - 5.0:
            perturbed = best_params.copy()
            for idx in range(len(perturbed)):
                perturbed[idx] += random.gauss(0, 0.05)
            p_score = objective(perturbed)
            if p_score < best_score:
                result2 = minimize(nm_objective, perturbed, method='Nelder-Mead', 
                                   options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if result2.fun < best_score:
                    best_params = result2.x
                    best_score = result2.fun

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if compute_overlap_penalty(inner[i], inner[j]) > 1e-6:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use a known valid configuration (hexagonal grid)
        inner = []
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((np.sqrt(3)*np.cos(theta), np.sqrt(3)*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((2*np.sqrt(3)*np.cos(theta), 2*np.sqrt(3)*np.sin(theta), 0))
        for pos in positions:
            inner.append([float(pos[0]), float(pos[1]), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
