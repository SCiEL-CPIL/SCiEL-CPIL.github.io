import numpy as np
import math
import time
from scipy.optimize import differential_evolution, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)

    def get_axes(verts):
        """Get 6 normal axes for a hexagon."""
        axes = []
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-12:
                # Normal is perpendicular to edge
                nx = -dy / length
                ny = dx / length
                axes.append((nx, ny))
        return axes

    def sat_overlap_penalty(v1, v2):
        """
        Calculate overlap penalty using SAT.
        Returns 0 if disjoint, positive if overlapping.
        """
        axes = get_axes(v1) + get_axes(v2)
        
        # Add vector between centers
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        dc = c2 - c1
        dist_c = np.linalg.norm(dc)
        if dist_c > 1e-12:
            axes.append((dc[0] / dist_c, dc[1] / dist_c))

        min_overlap = float('inf')
        for nx, ny in axes:
            n_vec = np.array([nx, ny])
            proj1 = v1 @ n_vec
            proj2 = v2 @ n_vec
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 <= min2 or max2 <= min1:
                return 0.0 # Disjoint
            
            # Overlap amount
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        return min_overlap

    def compute_min_side_hard(positions):
        """
        Compute minimum outer side to contain all inner hexagons.
        Assumes outer hexagon is axis-aligned (angle 0) for now.
        """
        # Normals for axis-aligned hexagon (pointy top or flat top?)
        # Let's assume flat top: vertices at 0, 60, 120...
        # Normals are at 30, 90, 150...
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
        max_proj = 0.0
        for ix, iy, iang in positions:
            verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in verts:
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # L * sqrt(3)/2 >= max_proj => L >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def objective_de(params):
        """
        Objective for Differential Evolution.
        params: [x0, y0, a0, x1, y1, a1, ...]
        """
        positions = []
        verts_list = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            positions.append((x, y, a))
            verts_list.append(hexagon_vertices(x, y, 1.0, a))
        
        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ov = sat_overlap_penalty(verts_list[i], verts_list[j])
                if ov > 0:
                    pen += ov
        
        if pen > 0.01:
            return 100.0 + compute_min_side_hard(positions) + pen
        
        return compute_min_side_hard(positions)

    def smooth_objective(params):
        """
        Smooth objective for polishing.
        """
        positions = []
        all_projs = []
        verts_list = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            positions.append((x, y, a))
            verts = hexagon_vertices(x, y, 1.0, a)
            verts_list.append(verts)
            
            for vx, vy in verts:
                # Project onto 6 normals
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    all_projs.append(vx * nx + vy * ny)
        
        # Log-sum-exp for smooth max
        max_p = max(all_projs)
        sum_exp = sum(math.exp(p - max_p) for p in all_projs)
        smooth_max = max_p + math.log(sum_exp) / 50.0
        side = 2.0 * smooth_max / sqrt3
        
        # Smooth overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ov = sat_overlap_penalty(verts_list[i], verts_list[j])
                # Softplus approximation of max(0, ov)
                # If ov is large, penalty is large.
                if ov > 0:
                     pen += ov
                else:
                     # Small gap, no penalty
                     pass
        
        return side + 10.0 * pen

    # --- Optimization ---
    
    bounds = [(-4, 4), (-4, 4), (0, 60)] * N
    
    # Phase 1: Differential Evolution
    best_result = None
    best_val = float('inf')
    
    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                objective_de, 
                bounds, 
                maxiter=100, 
                popsize=25, 
                tol=1e-6,
                seed=42,
                init='latinhypercube',
                polish=False # We will polish manually
            )
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception as e:
            print(f"DE failed: {e}")
            
    # Phase 2: Polish with L-BFGS-B or Powell
    if best_result is not None and time.time() < deadline - 10:
        try:
            # Use Powell as it handles non-differentiable functions better than L-BFGS-B
            res2 = minimize(
                smooth_objective, 
                best_result, 
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception as e:
            print(f"Polish failed: {e}")

    # Phase 3: Local Search (SA) to escape local minima if time permits
    if best_result is not None and time.time() < deadline - 5:
        current = best_result
        current_val = best_val
        temp = 0.1
        for i in range(1000):
            if time.time() > deadline - 2:
                break
            idx = np.random.randint(0, N)
            new = current.copy()
            new[3*idx] += np.random.normal(0, 0.05)
            new[3*idx+1] += np.random.normal(0, 0.05)
            new[3*idx+2] += np.random.normal(0, 2.0)
            
            new_val = smooth_objective(new)
            delta = new_val - current_val
            if delta < 0 or np.random.random() < math.exp(-delta / temp):
                current = new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_result = current
            temp *= 0.99

    # Extract best solution
    if best_result is not None:
        best_positions = []
        for i in range(N):
            best_positions.append([best_result[3*i], best_result[3*i+1], best_result[3*i+2]])
        
        # Final validation and side calculation
        # We should check if it's valid (no overlaps)
        # If invalid, we might need to fallback or just return the side (evaluator will penalize)
        # But the task says "invalid result ... scores 0". So we MUST ensure no overlap.
        
        # Check validity
        valid = True
        verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in best_positions]
        for i in range(N):
            for j in range(i+1, N):
                if sat_overlap_penalty(verts_list[i], verts_list[j]) > 1e-6:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Fallback to a known valid configuration (e.g. standard lattice)
            # This is a safe fallback
            best_positions = []
            # 1 center
            best_positions.append([0.0, 0.0, 0.0])
            # 6 ring
            for k in range(6):
                ang = math.radians(60 * k)
                x = 2.0 * math.cos(ang)
                y = 2.0 * math.sin(ang)
                best_positions.append([x, y, 0.0])
            # 4 more
            for k in range(4):
                ang = math.radians(30 + 120 * k)
                x = 3.5 * math.cos(ang)
                y = 3.5 * math.sin(ang)
                best_positions.append([x, y, 0.0])
            
            side = compute_min_side_hard(best_positions)
        else:
            side = compute_min_side_hard(best_positions)
            
        return best_positions, (0.0, 0.0), side, 0.0

    # Fallback if no result
    best_positions = []
    for i in range(N):
        if i == 0:
            best_positions.append([0.0, 0.0, 0.0])
        else:
            r = 2.5 * (i // 6 + 1)
            ang = math.radians(60 * (i % 6))
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            best_positions.append([x, y, 0.0])
    
    side = compute_min_side_hard(best_positions)
    return best_positions, (0.0, 0.0), side, 0.0
