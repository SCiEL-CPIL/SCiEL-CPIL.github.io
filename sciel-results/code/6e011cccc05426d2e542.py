import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        base = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
        verts = base @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side for given configuration."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def check_disjoint(centers, angles, tol=0.0001):
        """Check if all hexagons are disjoint using center distance (valid for same orientation)."""
        n = len(centers)
        min_dist = math.sqrt(3) - tol
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """High-precision polish of outer hexagon parameters via local search."""
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Multi-scale coordinate descent
        scales = [0.5, 0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*5, 0, scale*5]:
                            otheta_new = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta_new)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = otheta_new
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    # Generate initial configurations on hexagonal lattice
    sqrt3 = math.sqrt(3)
    d = sqrt3 + 0.0001  # slight separation to avoid touching
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*sqrt3/2.0])
    
    # Ring 1: 6 neighbors
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    
    # Generate ring 2 candidates
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    for p1 in ring1_vectors:
        for b in basis:
            p2 = p1 + b
            # Skip if it's ring 0 or ring 1
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=0.01):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc, atol=0.01):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple configurations
    num_configs = min(8, len(ring2_candidates))
    
    for config_idx in range(num_configs):
        if time.time() > deadline - 45:
            break
        
        # Select 5 ring 2 vectors
        if config_idx == 0:
            ring2_sel = ring2_candidates[:5]
        elif config_idx == 1:
            ring2_sel = ring2_candidates[5:10]
        elif config_idx == 2:
            ring2_sel = ring2_candidates[:3] + ring2_candidates[6:8]
        else:
            random.seed(config_idx * 7919)
            idxs = random.sample(range(len(ring2_candidates)), 5)
            ring2_sel = [ring2_candidates[i] for i in idxs]
        
        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        for vec in ring2_sel:
            centers.append([vec[0], vec[1]])
        
        # Center the configuration
        cx_mean = np.mean([c[0] for c in centers])
        cy_mean = np.mean([c[1] for c in centers])
        centers = [[c[0] - cx_mean, c[1] - cy_mean] for c in centers]
        
        angles = [0.0] * N
        
        # Check validity
        if not check_disjoint(centers, angles):
            continue
        
        # SLSQP optimization with outer hexagon as decision variables
        def objective(x):
            if time.time() > deadline - 30:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(N)]
            a = [x[3*i+2] for i in range(N)]
            ocx, ocy, otheta = x[3*N], x[3*N+1], x[3*N+2]
            return compute_outer_side(c, a, ocx, ocy, otheta)
        
        # Constraints: pairwise center distance >= sqrt(3)
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                def dist_constraint(x, i=i, j=j):
                    xi, yi = x[3*i], x[3*i+1]
                    xj, yj = x[3*j], x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - (sqrt3 - 0.001)**2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        # Initial params
        x0 = []
        for i in range(N):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])  # outer center and angle
        
        bounds = []
        for i in range(N):
            bounds.extend([(-5, 5), (-5, 5), (-30, 30)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.fun < best_side:
                c = [[res.x[3*i], res.x[3*i+1]] for i in range(N)]
                a = [res.x[3*i+2] for i in range(N)]
                ocx, ocy, otheta = res.x[3*N], res.x[3*N+1], res.x[3*N+2]
                
                # Verify disjointness
                if check_disjoint(c, a):
                    best_side = res.fun
                    best_result = (c, a, ocx, ocy, otheta)
                    
        except Exception:
            pass
    
    # Polish the best result
    if best_result is not None and time.time() < deadline - 5:
        c, a, ocx, ocy, otheta = best_result
        ocx, ocy, otheta, polished_side = polish_outer(c, a, ocx, ocy, otheta)
        if polished_side < best_side:
            best_side = polished_side
            best_result = (c, a, ocx, ocy, otheta)
    
    # Fallback: simple lattice arrangement
    if best_result is None:
        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        for vec in ring2_candidates[:5]:
            centers.append([vec[0], vec[1]])
        angles = [0.0] * N
        best_result = (centers, angles, 0.0, 0.0, 0.0)
        best_side = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)
    
    c, a, ocx, ocy, otheta = best_result
    
    # Build output: inner hexagons relative to outer center
    inner = []
    for i in range(N):
        inner.append([float(c[i][0] - ocx), float(c[i][1] - ocy), float(a[i])])
    
    return inner, (float(ocx), float(ocy)), float(best_side), float(otheta)
