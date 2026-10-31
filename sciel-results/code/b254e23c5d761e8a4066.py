import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
MIN_DIST = 2.0 

def _hexagon_vertices(center, side, angle):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle: rotation in radians
    """
    cx, cy = center
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _polygons_overlap(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    v1, v2: arrays of shape (6, 2)
    Returns True if they overlap.
    """
    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                axes.append(normal / norm)
        return axes

    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    
    for axis in axes1 + axes2:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
            
    return True

def _is_valid_packing(positions, angles):
    """
    Check if the set of hexagons defined by positions and angles are disjoint.
    positions: (N, 2)
    angles: (N,)
    """
    verts_list = []
    for i in range(len(positions)):
        verts_list.append(_hexagon_vertices(positions[i], 1.0, angles[i]))
        
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _polygons_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def _optimize_outer_container(positions, angles):
    """
    Find the minimum side length of a regular hexagon that contains all inner hexagons.
    """
    all_verts = []
    for i in range(N):
        verts = _hexagon_vertices(positions[i], 1.0, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    def cost(params):
        cx, cy, angle_rad = params
        C = np.array([cx, cy])
        
        normals = []
        for k in range(6):
            theta = angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        normals = np.array(normals)
        
        diffs = all_verts - C
        projections = np.dot(diffs, normals)
        
        max_projs = np.max(projections, axis=0)
        apothem_req = np.max(max_projs)
        
        if apothem_req < 0:
            return 0.0
            
        S = apothem_req / (math.sqrt(3) / 2)
        return S
    
    com = np.mean(positions, axis=0)
    x0 = [com[0], com[1], 0.0]
    
    try:
        res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
        if res.success:
            best_cx, best_cy, best_angle = res.x
            best_side = res.fun
        else:
            best_cx, best_cy, best_angle = x0
            best_side = cost(x0)
    except Exception:
        best_cx, best_cy, best_angle = x0
        best_side = cost(x0)
        
    return (best_cx, best_cy), best_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_side = np.inf
    
    for seed in range(15):
        if time.time() > deadline - 20.0:
            break
            
        np.random.seed(seed)
        positions = np.random.uniform(-3, 3, (N, 2))
        angles = np.random.uniform(0, 2 * math.pi, N)
        velocities = np.zeros_like(positions)
        
        md_best_positions = positions.copy()
        md_best_angles = angles.copy()
        md_best_score = np.inf
        
        iterations = 800
        
        for step in range(iterations):
            if time.time() > deadline - 10.0:
                break
                
            forces = np.zeros_like(positions)
            
            for i in range(N):
                for j in range(i + 1, N):
                    r_vec = positions[i] - positions[j]
                    dist = np.linalg.norm(r_vec)
                    if dist < 1e-6:
                        r_vec = np.random.normal(0, 0.1, 2)
                        dist = np.linalg.norm(r_vec)
                    
                    if dist < 2.0:
                        f_mag = 100.0 * (2.0 - dist)
                        f_vec = (r_vec / dist) * f_mag
                        forces[i] += f_vec
                        forces[j] -= f_vec
                    else:
                        if dist < 6.0:
                            f_mag = 0.1 * (dist - 2.0)
                            f_vec = -(r_vec / dist) * f_mag
                            forces[i] += f_vec
                            forces[j] -= f_vec
                            
                forces[i] -= 0.05 * positions[i]
                
            velocities += forces * 0.01
            velocities *= 0.9
            positions += velocities * 0.01
            
            if step % 20 == 0:
                if _is_valid_packing(positions, angles):
                    center = np.mean(positions, axis=0)
                    radii = np.linalg.norm(positions - center, axis=1)
                    max_r = np.max(radii)
                    if max_r < md_best_score:
                        md_best_score = max_r
                        md_best_positions = positions.copy()
                        md_best_angles = angles.copy()
                        
        if md_best_score < np.inf:
            try:
                center, side, angle = _optimize_outer_container(md_best_positions, md_best_angles)
                if side < best_side:
                    best_side = side
                    best_result = (md_best_positions, md_best_angles, center, side, angle)
            except Exception:
                pass
                
    if best_result is None:
        inner = []
        for k in range(-(N // 2), N - N // 2):
            inner.append([4.0 * k, 0.0, 0.0])
        return inner, (0.0, 0.0), 40.0, 0.0
        
    positions, angles, center, side, angle = best_result
    
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], math.degrees(angles[i])])
        
    return inner, (center[0], center[1]), side, math.degrees(angle)
# EVOLVE-BLOCK-END
