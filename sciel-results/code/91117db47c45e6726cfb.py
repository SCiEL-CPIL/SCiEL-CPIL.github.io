"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses simulated annealing with a physics-based repulsion model and exact
minimal enclosing hexagon computation.
"""
import numpy as np
import time
import math
import random

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(cx, cy, angle_deg, side, dx, dy):
    """Support function of a regular hexagon in direction (dx, dy)."""
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(cx, cy, ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            inside = True
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j+1)%6]
                edge = v2 - v1
                rel = pt - v1
                cross = edge[0]*rel[1] - edge[1]*rel[0]
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check disjointness using SAT (separating axis theorem)
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            if _hexagons_overlap(vi, vj):
                return False
                
    return True

def _hexagons_overlap(v1, v2):
    """Check if two hexagons (given as vertex arrays) overlap using SAT."""
    for poly in [v1, v2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            
            min1, max1 = np.inf, -np.inf
            min2, max2 = np.inf, -np.inf
            for pt in v1:
                proj = np.dot(pt, normal)
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for pt in v2:
                proj = np.dot(pt, normal)
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            
            if max1 < min2 or max2 < min1:
                return False
    return True

def _overlap_penalty(centers, angles):
    """Compute penalty for overlapping inner hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            vi = _hex_verts(centers[i,0], centers[i,1], 1.0, angles[i])
            vj = _hex_verts(centers[j,0], centers[j,1], 1.0, angles[j])
            if _hexagons_overlap(vi, vj):
                # Estimate overlap depth by center distance
                dist = np.linalg.norm(centers[i] - centers[j])
                penalty += max(0, 2.0 - dist) ** 2
    return penalty

def _total_cost(centers, angles, outer_angle_deg=0.0):
    """Total cost: enclosing hex side + overlap penalty."""
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    penalty = _overlap_penalty(centers, angles)
    return S + penalty * 10.0

def _random_init():
    """Generate a random initial configuration."""
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i] = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
        angles[i] = random.uniform(0, 360)
    return centers, angles

def _hex_lattice_init(scale=1.0):
    """Generate a hexagonal lattice initial configuration."""
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    idx = 0
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            centers[idx] = pos
            angles[idx] = 0.0
            idx += 1
    
    # Center the configuration
    cx_mean = np.mean(centers[:, 0])
    cy_mean = np.mean(centers[:, 1])
    centers -= np.array([cx_mean, cy_mean])
    
    return centers, angles

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_centers = None
    best_angles = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init(1.0))
    inits.append(_hex_lattice_init(1.02))
    inits.append(_hex_lattice_init(1.05))
    inits.append(_random_init())
    inits.append(_random_init())
    
    for centers, angles in inits:
        if time.time() > deadline:
            break
        
        # Simulated annealing
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        centers_cur = centers.copy()
        angles_cur = angles.copy()
        
        cur_cost = _total_cost(centers_cur, angles_cur)
        best_centers_sa = centers_cur.copy()
        best_angles_sa = angles_cur.copy()
        best_cost_sa = cur_cost
        
        for step in range(20000):
            if time.time() > deadline:
                break
            
            if T < T_min:
                break
            
            # Generate a random perturbation
            idx = random.randint(0, N-1)
            if random.random() < 0.3:
                # Perturb angle
                angles_cur[idx] += random.gauss(0, 5)
            else:
                # Perturb position
                centers_cur[idx] += random.gauss(0, 0.1, size=2)
            
            new_cost = _total_cost(centers_cur, angles_cur)
            delta = new_cost - cur_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_cost = new_cost
                if new_cost < best_cost_sa:
                    best_cost_sa = new_cost
                    best_centers_sa = centers_cur.copy()
                    best_angles_sa = angles_cur.copy()
            else:
                # Revert
                if random.random() < 0.3:
                    angles_cur[idx] -= random.gauss(0, 5)
                else:
                    centers_cur[idx] -= random.gauss(0, 0.1, size=2)
            
            T *= cooling
        
        # Verify best from SA
        if best_cost_sa < best_score:
            # Check validity
            S_val = _min_enclosing_hex_side(best_centers_sa, best_angles_sa, 0.0)
            if _check_validity(best_centers_sa, best_angles_sa, S_val, 0.0):
                best_score = S_val
                best_centers = best_centers_sa.copy()
                best_angles = best_angles_sa.copy()
    
    # If no valid configuration found, use a guaranteed valid one
    if best_centers is None:
        centers, angles = _hex_lattice_init(1.05)
        S_val = _min_enclosing_hex_side(centers, angles, 0.0)
        inner = []
        for i in range(N):
            inner.append([float(centers[i,0]), float(centers[i,1]), float(angles[i])])
        return inner, (0.0, 0.0), float(S_val), 0.0
    
    # Final refinement with smaller steps
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    centers_cur = best_centers.copy()
    angles_cur = best_angles.copy()
    cur_cost = _total_cost(centers_cur, angles_cur)
    best_centers_ref = centers_cur.copy()
    best_angles_ref = angles_cur.copy()
    best_cost_ref = cur_cost
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        if T < T_min:
            break
        
        idx = random.randint(0, N-1)
        if random.random() < 0.3:
            angles_cur[idx] += random.gauss(0, 1)
        else:
            centers_cur[idx] += random.gauss(0, 0.02, size=2)
        
        new_cost = _total_cost(centers_cur, angles_cur)
        delta = new_cost - cur_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur_cost = new_cost
            if new_cost < best_cost_ref:
                best_cost_ref = new_cost
                best_centers_ref = centers_cur.copy()
                best_angles_ref = angles_cur.copy()
        else:
            if random.random() < 0.3:
                angles_cur[idx] -= random.gauss(0, 1)
            else:
                centers_cur[idx] -= random.gauss(0, 0.02, size=2)
        
        T *= cooling
    
    # Verify final result
    S_val = _min_enclosing_hex_side(best_centers_ref, best_angles_ref, 0.0)
    if _check_validity(best_centers_ref, best_angles_ref, S_val, 0.0):
        best_centers = best_centers_ref
        best_angles = best_angles_ref
        best_score = S_val
    
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i,0]), float(best_centers[i,1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_score), 0.0
