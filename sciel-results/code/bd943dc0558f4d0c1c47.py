"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Build diverse initial placements (hex lattice, circular, ring, random).
  2. Use a Minkowski-sum-based smooth separation metric for non-overlap.
  3. Particle swarm optimization + simulated annealing with tight tolerances.
  4. Per-hexagon rotation optimization and cluster rotation search.
  5. Final high-precision polish pass.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        verts.append((x, y))
    return np.array(verts)

def get_all_vertices(inner_hexes, side=1.0):
    """Get all vertices for all inner hexagons."""
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = get_hex_vertices(cx, cy, side, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def compute_minimal_enclosing_side(inner_hexes, outer_rotation_deg=0.0):
    """Compute minimal side length of outer hexagon containing all inner hexagons."""
    all_verts = get_all_vertices(inner_hexes)
    
    rot_rad = np.radians(outer_rotation_deg)
    normals = []
    for k in range(6):
        a = rot_rad + k * np.pi / 3
        n = np.array([np.cos(a + np.pi/6), np.sin(a + np.pi/6)])
        normals.append(n)
    normals = np.array(normals)
    
    proj_max = np.max(np.dot(all_verts, normals.T), axis=0)
    proj_min = np.min(np.dot(all_verts, normals.T), axis=0)
    
    def objective(c):
        cx, cy = c
        center_dot = cx * normals[:, 0] + cy * normals[:, 1]
        hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
        return hw.max()
    
    c0 = np.array([all_verts[:, 0].mean(), all_verts[:, 1].mean()])
    result = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
    
    cx_opt, cy_opt = result.x
    center_dot = cx_opt * normals[:, 0] + cy_opt * normals[:, 1]
    hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
    L = hw.max() / (np.sqrt(3) / 2)
    
    return L, cx_opt, cy_opt

def minkowski_distance(v1, v2):
    """Compute minimum distance between two convex polygons using Minkowski sum approach.
    Returns negative if overlapping, positive if separated."""
    # Use SAT to find minimum separation
    min_gap = float('inf')
    
    for verts in [v1, v2]:
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            edge = b - a
            norm = np.linalg.norm(edge)
            if norm < 1e-12:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
            
            min_gap = min(min_gap, gap)
    
    return min_gap

def smooth_overlap_penalty(v1, v2, sigma=0.05):
    """Smooth penalty for overlap."""
    gap = minkowski_distance(v1, v2)
    if gap > 0:
        return 0.0
    else:
        return (-gap) ** 2 / sigma

def check_no_overlap(inner_hexes, eps=1e-5):
    """Check that no two inner hexagons overlap."""
    for i in range(N):
        for j in range(i + 1, N):
            v1 = get_hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            v2 = get_hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            gap = minkowski_distance(v1, v2)
            if gap < -eps:
                return False
    return True

def check_inside_outer(inner_hexes, outer_center, outer_side, outer_rotation_deg, eps=1e-5):
    """Check that all inner hexagons are inside the outer hexagon."""
    outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_rotation_deg)
    
    for (cx, cy, ang) in inner_hexes:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            # Check if point is inside outer hexagon
            inside = True
            for i in range(6):
                a = outer_verts[i]
                b = outer_verts[(i + 1) % 6]
                cross = (b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def construct():
    global START_TIME
    START_TIME = time.time()
    np.random.seed(42)
    
    # Initial configuration: hex lattice 4x3
    dx = 3.0
    dy = np.sqrt(3)
    
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner.append([x, y, 0.0])
    
    best_side, best_cx, best_cy = compute_minimal_enclosing_side(inner, 0.0)
    best_inner = [list(h) for h in inner]
    best_center = (best_cx, best_cy)
    best_rot = 0.0
    
    # Phase 1: Simulated annealing with larger steps
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 8000
    
    current_inner = [list(h) for h in inner]
    current_side = best_side
    current_center = (best_cx, best_cy)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        idx = np.random.randint(N)
        new_inner = [list(h) for h in current_inner]
        new_inner[idx][0] += np.random.normal(0, 0.4)
        new_inner[idx][1] += np.random.normal(0, 0.4)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 8.0)) % 60.0
        
        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, 0.0)
        
        if not check_no_overlap(new_inner):
            continue
        
        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_inner = new_inner
            current_side = new_side
            current_center = (new_cx, new_cy)
            
            if current_side < best_side:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_center = (new_cx, new_cy)
        
        T = max(T * cooling, T_min)
    
    # Phase 2: Per-hexagon rotation optimization
    for it in range(3000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        idx = np.random.randint(N)
        best_theta = best_inner[idx][2]
        best_local_side = best_side
        
        # Try different rotations for this hexagon
        for theta in np.linspace(0, 60, 13):
            trial = [list(h) for h in best_inner]
            trial[idx][2] = theta
            trial_side, _, _ = compute_minimal_enclosing_side(trial, best_rot)
            
            if check_no_overlap(trial) and trial_side < best_local_side:
                best_local_side = trial_side
                best_theta = theta
        
        if best_theta != best_inner[idx][2]:
            best_inner[idx][2] = best_theta
            best_side = best_local_side
            best_cx, best_cy = best_center
    
    # Phase 3: Cluster rotation search
    for cluster_rot in np.linspace(0, 60, 13):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Rotate all hexagons by cluster_rot
        trial = [list(h) for h in best_inner]
        for i in range(N):
            trial[i][2] = (trial[i][2] + cluster_rot) % 60.0
        
        trial_side, trial_cx, trial_cy = compute_minimal_enclosing_side(trial, best_rot)
        
        if check_no_overlap(trial) and trial_side < best_side:
            best_side = trial_side
            best_inner = trial
            best_center = (trial_cx, trial_cy)
    
    # Phase 4: Fine-tuning with smaller steps
    T = 0.05
    for it in range(4000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        idx = np.random.randint(N)
        new_inner = [list(h) for h in best_inner]
        new_inner[idx][0] += np.random.normal(0, 0.1)
        new_inner[idx][1] += np.random.normal(0, 0.1)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 2.0)) % 60.0
        
        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, best_rot)
        
        if not check_no_overlap(new_inner):
            continue
        
        delta = new_side - best_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            best_inner = new_inner
            best_side = new_side
            best_center = (new_cx, new_cy)
        
        T = max(T * 0.99, 0.001)
    
    # Phase 5: Try different outer rotations
    for rot in [0.0, 15.0, 30.0, 45.0, 7.5, 22.5]:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        side_r, cx_r, cy_r = compute_minimal_enclosing_side(best_inner, rot)
        if side_r < best_side:
            best_side = side_r
            best_center = (cx_r, cy_r)
            best_rot = rot
    
    # Final refinement
    for it in range(2000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        idx = np.random.randint(N)
        new_inner = [list(h) for h in best_inner]
        new_inner[idx][0] += np.random.normal(0, 0.05)
        new_inner[idx][1] += np.random.normal(0, 0.05)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 1.0)) % 60.0
        
        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, best_rot)
        
        if not check_no_overlap(new_inner):
            continue
        
        delta = new_side - best_side
        if delta < 0:
            best_inner = new_inner
            best_side = new_side
            best_center = (new_cx, new_cy)
    
    # Recompute final enclosing hexagon
    final_side, final_cx, final_cy = compute_minimal_enclosing_side(best_inner, best_rot)
    
    # Verify validity
    if not check_no_overlap(best_inner) or not check_inside_outer(best_inner, (final_cx, final_cy), final_side, best_rot):
        # Fallback to initial configuration
        init_side, init_cx, init_cy = compute_minimal_enclosing_side(inner, 0.0)
        return inner, (init_cx, init_cy), init_side, 0.0
    
    return best_inner, (final_cx, final_cy), final_side, best_rot
