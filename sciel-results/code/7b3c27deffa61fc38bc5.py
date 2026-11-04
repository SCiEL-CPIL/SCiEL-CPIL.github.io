import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        # Separating Axis Theorem for two convex polygons
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    def compute_outer_side(centers, angles, otheta_deg=0.0):
        # Outer hexagon centered at (0,0), side length L
        # Normals are at angles otheta + 30 + k*60
        # Apothem = L * sqrt(3) / 2
        # Condition: max projection <= Apothem
        # L = 2 * max_proj / sqrt(3)
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def objective(v):
        # v: [x1, y1, a1, x2, y2, a2, ...]
        if time.time() > deadline - 5:
            return 1e6
        centers = []
        angles = []
        for i in range(12):
            centers.append([v[3*i], v[3*i+1]])
            angles.append(v[3*i+2])
        
        # Fast collision check
        for i in range(12):
            for j in range(i+1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                # Min distance is sqrt(3) approx. 
                # If dist < sqrt(3), potential collision.
                if dist_sq < 3.0: 
                     # Do a more precise check? 
                     # For speed in objective, we might just penalize
                     # But we need to ensure we don't get stuck in invalid states
                     pass
        
        # Compute side length
        side = compute_outer_side(centers, angles, 0.0)
        
        # Penalty for collisions
        penalty = 0.0
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i+1, 12):
                # Fast reject
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx*dx + dy*dy > 4.0: # Max dist for collision is small
                    continue
                if not are_disjoint_sat(verts[i], verts[j]):
                    penalty += 100.0
        
        return side + penalty

    def generate_initial_config():
        # Honeycomb lattice
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        basis = [u, v, v - u, -u, -v, u - v]
        
        centers = [[0.0, 0.0]]
        angles = [30.0]
        
        # Ring 1
        for vec in basis:
            centers.append([float(vec[0]), float(vec[1])])
            angles.append(30.0)
            
        # Ring 2 candidates
        ring2_candidates = []
        for vec in basis:
            for b in basis:
                p = vec + b
                # Check if it's in ring 0 or 1
                is_r0 = np.allclose(p, np.array([0.0, 0.0]), atol=0.01)
                is_r1 = False
                for b1 in basis:
                    if np.allclose(p, b1, atol=0.01):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    ring2_candidates.append(p)
        
        # Remove duplicates
        unique_r2 = []
        for p in ring2_candidates:
            dup = False
            for u_p in unique_r2:
                if np.allclose(p, u_p, atol=0.01):
                    dup = True
                    break
            if not dup:
                unique_r2.append(p)
                
        # Pick 5 from ring 2
        # Try different subsets
        subsets = []
        # Subset 1: First 5
        subsets.append(unique_r2[:5])
        # Subset 2: Last 5
        subsets.append(unique_r2[-5:])
        # Subset 3: Random 5
        random.seed(123)
        random.shuffle(unique_r2)
        subsets.append(unique_r2[:5])
        
        return subsets

    best_side = float('inf')
    best_result = None
    
    subsets = generate_initial_config()
    
    for subset in subsets:
        if time.time() > deadline - 10:
            break
            
        centers_init = [[0.0, 0.0]]
        angles_init = [30.0]
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        basis = [u, v, v - u, -u, -v, u - v]
        
        for vec in basis:
            centers_init.append([float(vec[0]), float(vec[1])])
            angles_init.append(30.0)
            
        for vec in subset:
            centers_init.append([float(vec[0]), float(vec[1])])
            angles_init.append(30.0)
            
        x0 = []
        for i in range(12):
            x0.extend([centers_init[i][0], centers_init[i][1], angles_init[i]])
            
        # Add some noise
        x0_noisy = x0 + np.random.normal(0, 0.1, len(x0))
        
        try:
            res = minimize(objective, x0_noisy, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e5: # Validity check
                centers = []
                angles = []
                for i in range(12):
                    centers.append([res.x[3*i], res.x[3*i+1]])
                    angles.append(res.x[3*i+2])
                
                if check_validity(centers, angles):
                    side = compute_outer_side(centers, angles, 0.0)
                    if side < best_side:
                        best_side = side
                        best_result = (centers, angles)
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
        centers = [[0.0, 0.0]]
        angles = [30.0]
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        basis = [u, v, v - u, -u, -v, u - v]
        for vec in basis:
            centers.append([float(vec[0]), float(vec[1])])
            angles.append(30.0)
        # Add 5 from ring 2
        ring2 = []
        for vec in basis:
            for b in basis:
                p = vec + b
                if not np.allclose(p, np.array([0.0, 0.0]), atol=0.01):
                    is_r1 = False
                    for b1 in basis:
                        if np.allclose(p, b1, atol=0.01):
                            is_r1 = True
                            break
                    if not is_r1:
                        ring2.append(p)
        # Unique
        unique_r2 = []
        for p in ring2:
            dup = False
            for u_p in unique_r2:
                if np.allclose(p, u_p, atol=0.01):
                    dup = True
                    break
            if not dup:
                unique_r2.append(p)
        for vec in unique_r2[:5]:
            centers.append([float(vec[0]), float(vec[1])])
            angles.append(30.0)
        best_side = compute_outer_side(centers, angles, 0.0)
        best_result = (centers, angles)

    best_centers, best_angles = best_result
    
    # Final Polish
    x0 = []
    for i in range(12):
        x0.extend([best_centers[i][0], best_centers[i][1], best_angles[i]])
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < 1e5:
            centers = []
            angles = []
            for i in range(12):
                centers.append([res.x[3*i], res.x[3*i+1]])
                angles.append(res.x[3*i+2])
            if check_validity(centers, angles):
                side = compute_outer_side(centers, angles, 0.0)
                if side < best_side:
                    best_side = side
                    best_centers = centers
                    best_angles = angles
    except:
        pass

    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
