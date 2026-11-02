import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_hex_vertices(x, y, angle):
        """Returns list of 6 vertices of a unit regular hexagon."""
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3
            verts.append((x + math.cos(a), y + math.sin(a)))
        return verts

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects p3-p4."""
        # This is a fallback, SAT is preferred for convex polygons
        pass

    def hexagons_disjoint(verts1, verts2):
        """
        Check if two convex polygons (hexagons) are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if they intersect or touch.
        """
        edges = []
        # Add edges from hex 1
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % 6]
            # Normal to edge p1-p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal is (-dy, dx) or (dy, -dx). We just need the direction.
            nx = -dy
            ny = dx
            # Normalize
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-9:
                nx /= norm
                ny /= norm
            edges.append((nx, ny))
        
        # Add edges from hex 2
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i + 1) % 6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx = -dy
            ny = dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-9:
                nx /= norm
                ny /= norm
            edges.append((nx, ny))

        # Check projection on each axis
        for nx, ny in edges:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            
            for v in verts1:
                proj = v[0]*nx + v[1]*ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            for v in verts2:
                proj = v[0]*nx + v[1]*ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            
            # If there is a gap, they are disjoint
            if max1 < min2 or max2 < min1:
                return True
        
        return False

    def all_disjoint(positions, angles):
        """Check if all hexagons are pairwise disjoint."""
        verts_list = []
        for i in range(N):
            verts_list.append(get_hex_vertices(positions[i][0], positions[i][1], angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    def min_outer_side(positions, angles, alpha):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, rotation alpha)
        that contains all inner hexagons.
        """
        outer_normals = [alpha + k * math.pi / 3 for k in range(6)]
        max_dot = 0.0
        
        for i in range(N):
            verts = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
            for v in verts:
                for n_angle in outer_normals:
                    dot = v[0] * math.cos(n_angle) + v[1] * math.sin(n_angle)
                    if dot > max_dot:
                        max_dot = dot
        
        return 2.0 * max_dot / math.sqrt(3)

    def objective(params):
        """
        Objective function for optimization.
        params: [x1, y1, a1, ..., xN, yN, aN, alpha]
        """
        positions = []
        angles = []
        for i in range(N):
            positions.append((params[i*3], params[i*3+1]))
            angles.append(params[i*3+2])
        alpha = params[3*N]
        
        # Check disjointness
        if not all_disjoint(positions, angles):
            # Penalty for overlap
            return 1000.0
        
        # Calculate outer side length
        return min_outer_side(positions, angles, alpha)

    # --- Initial Guess: Hexagonal Lattice ---
    # Axial coordinates for 11 hexagons
    hex_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-2, 0)
    ]
    
    init_positions = []
    for q, r in hex_coords:
        # Flat-top hexagon axial to cartesian
        x = 1.5 * q
        y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
        init_positions.append((x, y))
        
    init_angles = [0.0] * N
    init_alpha = 0.0
    
    x0 = []
    for p, a in zip(init_positions, init_angles):
        x0.append(p[0])
        x0.append(p[1])
        x0.append(a)
    x0.append(init_alpha)
    x0 = np.array(x0)
    
    best_result = None
    best_value = objective(x0)
    
    # --- Optimization 1: Nelder-Mead ---
    if time.time() < deadline - 30:
        try:
            # Use a tighter tolerance
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_value:
                best_value = res.fun
                best_result = res.x
        except Exception:
            pass

    # --- Optimization 2: Differential Evolution (Global Search) ---
    # Only if time permits and we haven't found a very good solution
    if time.time() < deadline - 40 and best_value > 4.5: # 4.5 is a rough estimate of optimal
        bounds = []
        for i in range(N):
            bounds.append((-5, 5))
            bounds.append((-5, 5))
            bounds.append((0, 2 * math.pi))
        bounds.append((0, 2 * math.pi))
        
        try:
            # DE is slow, use limited maxiter
            res_de = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-5, seed=42, polish=False)
            if res_de.fun < best_value:
                best_value = res_de.fun
                best_result = res_de.x
        except Exception:
            pass

    # --- Final Refinement: Nelder-Mead on best result ---
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(objective, best_result, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            if res_final.fun < best_value:
                best_value = res_final.fun
                best_result = res_final.x
        except Exception:
            pass

    # --- Construct Output ---
    if best_result is not None and best_value < 100.0: # Valid result found
        inner = []
        for i in range(N):
            x = best_result[i * 3]
            y = best_result[i * 3 + 1]
            a = best_result[i * 3 + 2]
            a_deg = math.degrees(a) % 360.0
            inner.append([x, y, a_deg])
        
        outer_center = (0.0, 0.0)
        outer_side = best_value
        outer_angle_deg = math.degrees(best_result[3 * N]) % 360.0
        
        # Final validity check before returning
        # Re-extract positions and angles for check
        pos = [(inner[i][0], inner[i][1]) for i in range(N)]
        ang = [math.radians(inner[i][2]) for i in range(N)]
        
        if all_disjoint(pos, ang):
            return inner, outer_center, outer_side, outer_angle_deg
        else:
            # If final check fails, fallback to initial guess (which is valid)
            pass

    # Fallback: Return the initial lattice packing (which is valid)
    inner = []
    for p, a in zip(init_positions, init_angles):
        inner.append([p[0], p[1], math.degrees(a) % 360.0])
    
    # Calculate side for fallback
    side = min_outer_side(init_positions, init_angles, init_alpha)
    
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
