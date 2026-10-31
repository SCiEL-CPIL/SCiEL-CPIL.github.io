"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Generate 6 vertices of a regular hexagon."""
    angles = [math.radians(angle_deg + 60 * k) for k in range(6)]
    return np.array([[cx + side * math.cos(a), cy + side * math.sin(a)] for a in angles])

def _polygon_overlap_penalty(v1, v2):
    """
    Calculate overlap penalty between two convex polygons (hexagons).
    Uses Separating Axis Theorem.
    If separated, returns 0.
    If overlapping, returns a penalty based on the depth of overlap along critical axes.
    """
    axes = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
            
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
            
    total_penalty = 0.0
    for axis in axes:
        proj1 = np.dot(v1, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        proj2 = np.dot(v2, axis)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        else:
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                total_penalty += overlap ** 2
                
    return total_penalty

def construct():
    sqrt3 = math.sqrt(3)
    
    # Heuristic initial arrangement for reference (not used directly in DE init if random)
    # But we can use it to set bounds or initial center
    
    # Objective function
    def objective(params):
        n = len(positions)
        pos = []
        for i in range(n):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*n + i] for i in range(n)]
        cx = params[3*n]
        cy = params[3*n + 1]
        s = params[3*n + 2]
        
        if s < 2.0:
            return 1e6
            
        verts = []
        for i in range(n):
            v = _hex_vertices(pos[i][0], pos[i][1], 1.0, angs[i])
            verts.append(v)
            
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                p_pen = _polygon_overlap_penalty(verts[i], verts[j])
                if p_pen > 0:
                    penalty += p_pen * 1000
                    if penalty > 100:
                        return 1e6 + penalty
        
        if penalty > 1e-6:
            return 1e6 + penalty
            
        # Containment check
        inradius_outer = s * sqrt3 / 2
        normal_angles = [math.radians(30 + 60*k) for k in range(6)]
        
        for i in range(n):
            v = verts[i]
            for k in range(6):
                px, py = v[k]
                dx = px - cx
                dy = py - cy
                proj = dx * math.cos(normal_angles[k]) + dy * math.sin(normal_angles[k])
                if proj > inradius_outer + 1e-6:
                    penalty += (proj - inradius_outer) * 100
                    if penalty > 100:
                        return 1e6 + penalty
                        
        if penalty > 1e-6:
            return 1e6 + penalty
            
        return s

    # Define bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
    for i in range(N):
        bounds.append((-30, 30))
    bounds.append((-5, 5))
    bounds.append((-5, 5))
    bounds.append((2.5, 10.0))
    
    # Multi-start DE
    best_result = None
    best_fun = float('inf')
    
    seeds = [42, 123, 999, 777]
    for seed in seeds:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=25,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=True
        )
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
            
        if best_fun < 3.5:
            break

    if best_result is None or best_result.fun > 100:
        # Fallback
        positions = []
        angles = []
        rows = [3, 4, 4]
        y_offset = 0.0
        for i, count in enumerate(rows):
            x_start = -(count - 1) * sqrt3 / 2
            for j in range(count):
                x = x_start + j * sqrt3
                y = y_offset
                positions.append([x, y])
                angles.append(0.0)
            y_offset += 1.5
            
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
            
        max_dist = 0
        for p in positions:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        s = max_dist + 1.5
        cx = 0.0
        cy = 0.0
    else:
        best_params = best_result.x
        pos = []
        for i in range(N):
            pos.append([best_params[2*i], best_params[2*i+1]])
        angs = [best_params[2*N + i] for i in range(N)]
        cx = best_params[3*N]
        cy = best_params[3*N + 1]
        s = best_params[3*N + 2]
        positions = pos
        angles = angs

    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    return inner, (cx, cy), s, 0.0
# EVOLVE-BLOCK-END
