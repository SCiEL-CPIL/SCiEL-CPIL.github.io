"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_points(points, axes):
        # points: (M, 2), axes: (K, 2)
        # returns (M, K) array of projections
        return points @ axes.T

    def sat_overlap_depth(verts1, verts2, axes1, axes2):
        """
        Returns the minimum translation depth to separate two convex polygons.
        Returns 0 if disjoint.
        """
        all_axes = np.array(axes1 + axes2)
        proj1 = project_points(verts1, all_axes)
        proj2 = project_points(verts2, all_axes)
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Overlap on each axis
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any overlap is <= 0, they are disjoint
        if np.any(overlaps <= 1e-9):
            return 0.0
            
        # The minimum overlap is the penetration depth
        return np.min(overlaps)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy) with side 'side' and rotation 'angle_deg'."""
        # Translate point to hexagon local coords
        local_x = px - cx
        local_y = py - cy
        
        # Rotate point by -angle
        rad = math.radians(-angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        lx = local_x * cos_a - local_y * sin_a
        ly = local_x * sin_a + local_y * cos_a
        
        # Hexagon vertices at angles 0, 60, ... with radius 'side' (circumradius)
        # For a regular hexagon, circumradius = side length.
        # The hexagon is defined by 6 inequalities.
        # Normal vectors are at 30, 90, ... degrees.
        # Distance from center to side (apothem) = side * sqrt(3)/2
        apothem = side * SQRT3 / 2.0
        
        # Check against 6 sides
        for i in range(6):
            normal_angle = math.radians(30 + i * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = lx * nx + ly * ny
            if proj > apothem:
                return False, proj - apothem
        return True, 0.0

    def containment_penalty(verts, cx, cy, side, angle_deg):
        """
        Returns sum of squared distances for vertices outside the outer hexagon.
        """
        penalty = 0.0
        for vx, vy in verts:
            inside, dist = point_in_hexagon(vx, vy, cx, cy, side, angle_deg)
            if not inside:
                penalty += dist * dist
        return penalty

    def evaluate(params):
        """
        params: [x1, y1, a1, ..., xN, yN, aN, X, Y, L, phi]
        Returns: Penalty if invalid, L if valid.
        """
        inner_params = params[:N*3]
        X = params[N*3]
        Y = params[N*3+1]
        L = params[N*3+2]
        phi = params[N*3+3]
        
        # Ensure L is positive and reasonably bounded
        if L < 1.0:
            return 1000.0 + (1.0 - L) * 100.0
            
        total_penalty = 0.0
        hex_verts = []
        hex_axes = []
        
        # Process inner hexagons
        for i in range(N):
            x = inner_params[i*3]
            y = inner_params[i*3+1]
            a = inner_params[i*3+2]
            
            # Normalize angle to [0, 360) for consistency? Not strictly needed for math.
            verts = get_hex_vertices(x, y, a)
            axes = get_hex_axes(a)
            hex_verts.append(verts)
            hex_axes.append(axes)
            
            # Check containment in outer hexagon
            total_penalty += containment_penalty(verts, X, Y, L, phi)
        
        # Check overlaps between inner hexagons
        for i in range(N):
            for j in range(i+1, N):
                depth = sat_overlap_depth(hex_verts[i], hex_verts[j], hex_axes[i], hex_axes[j])
                if depth > 0:
                    total_penalty += depth * depth
        
        if total_penalty > 1e-6:
            return L + 100.0 * total_penalty
        else:
            return L

    # Differential Evolution Implementation
    # Bounds for variables
    # Inner hexagons: x, y in [-10, 10], angle in [0, 360]
    # Outer: X, Y in [-10, 10], L in [2, 10], phi in [0, 360]
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 360))  # angle
    
    bounds.append((-10, 10)) # X
    bounds.append((-10, 10)) # Y
    bounds.append((2.0, 10.0)) # L
    bounds.append((0, 360)) # phi

    # Custom DE
    pop_size = 50
    F = 0.8 # Mutation factor
    CR = 0.9 # Crossover rate
    
    # Initialize population
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, len(bounds)))
    
    # Initial evaluation
    pop_fitness = np.array([evaluate(ind) for ind in pop])
    best_idx = np.argmin(pop_fitness)
    best_individual = pop[best_idx].copy()
    best_fitness = pop_fitness[best_idx]
    
    # Random restarts for global exploration
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Run DE for a fixed number of generations or time
        gen = 0
        max_gens = 200
        
        while gen < max_gens and time.time() < deadline - 10.0:
            gen += 1
            new_pop = pop.copy()
            
            for i in range(pop_size):
                if time.time() > deadline - 10.0:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = random.sample(range(pop_size), 3)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clamp to bounds
                for k in range(len(bounds)):
                    if mutant[k] < bounds[k][0]:
                        mutant[k] = bounds[k][0]
                    elif mutant[k] > bounds[k][1]:
                        mutant[k] = bounds[k][1]
                        
                # Crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, len(bounds)-1)
                for j in range(len(bounds)):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                        
                # Evaluate
                trial_fitness = evaluate(trial)
                
                # Selection
                if trial_fitness < pop_fitness[i]:
                    new_pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                        
            pop = new_pop
            
            # Convergence check (optional)
            if gen % 10 == 0:
                # Re-seed worst individuals if stagnating?
                if pop_fitness.min() == pop_fitness.max():
                    break

        # Local Refinement with Nelder-Mead on the best individual found so far
        if time.time() < deadline - 5.0:
            def objective_nm(p):
                return evaluate(p)
            
            try:
                res = minimize(objective_nm, best_individual, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res.fun < best_fitness:
                    best_fitness = res.fun
                    best_individual = res.x
            except Exception:
                pass

    # Extract final result
    x_final = best_individual[:N*3]
    X = best_individual[N*3]
    Y = best_individual[N*3+1]
    L = best_individual[N*3+2]
    phi = best_individual[N*3+3]
    
    # Construct return values
    inner_list = []
    for i in range(N):
        x = x_final[i*3]
        y = x_final[i*3+1]
        a = x_final[i*3+2]
        inner_list.append([x, y, a])
        
    # Verify validity one last time
    # If invalid, try to return a known valid configuration (e.g., grid)
    # For now, assume DE found a valid one. If not, the penalty would have been high, 
    # but we need to ensure we don't return an invalid packing.
    
    # Check validity
    hex_verts = []
    hex_axes = []
    for i in range(N):
        x = inner_list[i][0]
        y = inner_list[i][1]
        a = inner_list[i][2]
        hex_verts.append(get_hex_vertices(x, y, a))
        hex_axes.append(get_hex_axes(a))
        
    valid = True
    for i in range(N):
        # Containment
        for vx, vy in hex_verts[i]:
            inside, _ = point_in_hexagon(vx, vy, X, Y, L, phi)
            if not inside:
                valid = False
                break
        if not valid:
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                depth = sat_overlap_depth(hex_verts[i], hex_verts[j], hex_axes[i], hex_axes[j])
                if depth > 1e-6:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback to a simple valid configuration
        # Grid of 11 hexagons
        # 3-4-3-1 arrangement or similar
        # Let's use the seed's fallback logic or a simple grid
        # Simple grid:
        # Row 0: 3 hex
        # Row 1: 4 hex
        # Row 2: 3 hex
        # Row 3: 1 hex
        # This is not optimal but valid.
        
        # Better fallback: Use the initial best from DE even if slightly invalid? No, must be valid.
        # Let's construct a valid one manually.
        # 11 hexagons in a 4x4 grid cluster?
        # Let's just use the first few from a dense grid.
        
        coords = []
        # Centered at 0,0
        # (0,0)
        coords.append((0,0))
        # Neighbors
        coords.append((1,0))
        coords.append((-1,0))
        coords.append((0,1))
        coords.append((0,-1))
        coords.append((1,1))
        coords.append((-1,1))
        coords.append((1,-1))
        coords.append((-1,-1))
        coords.append((2,0))
        coords.append((-2,0))
        
        inner_fb = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            inner_fb.append([x, y, 0.0])
            
        # Calculate minimal enclosing hexagon for this fallback
        # Rotate outer to align with inner?
        # Just use a large enough hexagon
        # Max dist from origin:
        max_r = max(math.sqrt(x*x+y*y) for x,y,a in inner_fb)
        # Side length of hexagon with circumradius R is R.
        # But we need to enclose.
        # For a hexagon of side L, the distance from center to vertex is L.
        # Distance from center to edge is L*sqrt(3)/2.
        # So L must be at least max_r.
        L_fb = max_r + 0.1
        return (inner_fb, (0.0, 0.0), L_fb, 0.0)

    return (inner_list, (X, Y), L, phi)

# EVOLVE-BLOCK-END
