import math
import time
import numpy as np

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    # Hexagon geometry
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons
    
    # Generate ring-2 centers using axial coordinates
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
    
    # Ring 1 centers
    ring1_centers = []
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        ring1_centers.append((x, y))
    
    # Base centers: origin + ring 1
    base_centers = np.array([(0.0, 0.0)] + ring1_centers)  # shape (7, 2)
    ring2_arr = np.array(ring2_centers)  # shape (12, 2)
    
    # Hexagon vertices relative to center (pointy-top: vertices at 0, 60, 120, ...)
    hex_verts = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])  # shape (6, 2)
    
    # Generate all 495 combinations of 4 from 12
    from itertools import combinations
    all_combos = list(combinations(range(12), 4))
    n_combos = len(all_combos)  # 495
    
    # For each combination, we have 11 centers -> 66 vertices
    # We want to find min over theta of (2/sqrt(3)) * max_k max_i |P_i . n_k(theta)|
    
    # Strategy: vectorize over combinations
    # For each combo, centers = base_centers + ring2_arr[combo]
    # vertices = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
    # shape: (n_combos, 11, 6, 2)
    
    # Precompute all combo centers
    # combo_indices: shape (495, 4)
    combo_indices = np.array(all_combos)  # shape (495, 4)
    
    # All centers for all combos: shape (495, 11, 2)
    # base_centers: (7, 2), ring2 selected: (495, 4, 2)
    ring2_selected = ring2_arr[combo_indices]  # shape (495, 4, 2)
    all_centers = np.concatenate([
        base_centers[np.newaxis, :, :],  # (1, 7, 2) -> broadcast to (495, 7, 2)
        ring2_selected  # (495, 4, 2)
    ], axis=1)  # shape (495, 11, 2)
    
    # All vertices: shape (495, 66, 2)
    # centers[:, :, np.newaxis, :] + hex_verts[np.newaxis, np.newaxis, :, :]
    all_verts = all_centers[:, :, np.newaxis, :] + hex_verts[np.newaxis, np.newaxis, :, :]
    all_verts = all_verts.reshape(n_combos, 66, 2)  # shape (495, 66, 2)
    
    # Now for each theta, compute S for all combos
    # theta sweep: use a moderate grid first
    n_theta = 720  # 0.0833 degree steps
    theta_steps = np.linspace(0, math.pi / 3, n_theta, endpoint=False)
    
    # Normal vectors: shape (n_theta, 6, 2)
    k_angles = np.arange(6) * math.pi / 3
    all_angles = theta_steps[:, np.newaxis] + k_angles[np.newaxis, :]  # (n_theta, 6)
    normals = np.stack([np.cos(all_angles), np.sin(all_angles)], axis=-1)  # (n_theta, 6, 2)
    
    # For each combo (495), each theta (720), each normal (6): max over 66 vertices of |dot|
    # dots: (495, 720, 6, 66)
    # all_verts: (495, 66, 2) -> (495, 1, 1, 66, 2)
    # normals: (720, 6, 2) -> (1, 720, 6, 1, 2)
    
    # This is too large: 495 * 720 * 6 * 66 * 2 = ~282M floats = ~2.2GB. Too much!
    
    # Better approach: process in batches
    batch_size = 50
    best_S = float('inf')
    best_combo_idx = -1
    best_theta = 0.0
    
    for batch_start in range(0, n_combos, batch_size):
        if time.time() > deadline - safety_margin:
            break
        
        batch_end = min(batch_start + batch_size, n_combos)
        batch_verts = all_verts[batch_start:batch_end]  # (B, 66, 2)
        B = batch_end - batch_start
        
        # dots: (B, n_theta, 6, 66)
        # batch_verts: (B, 66, 2) -> (B, 1, 1, 66, 2)
        # normals: (n_theta, 6, 2) -> (1, n_theta, 6, 1, 2)
        dots = np.einsum('bpi,tpq->bpt', batch_verts, normals)  # (B, n_theta, 6)
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=0)  # (n_theta, 6) -> wait, need max over batch too? No, per batch
        
        # Actually: max over vertices (axis=-1) for each (b, t, k)
        max_over_verts = np.max(abs_dots, axis=-1)  # (B, n_theta, 6)
        max_over_normals = np.max(max_over_verts, axis=-1)  # (B, n_theta)
        S_per_theta = (2.0 / math.sqrt(3)) * max_over_normals  # (B, n_theta)
        
        # Find best (combo, theta) in this batch
        local_best_idx = np.unravel_index(np.argmin(S_per_theta), S_per_theta.shape)
        local_best_S = S_per_theta[local_best_idx]
        
        if local_best_S < best_S:
            best_S = local_best_S
            best_combo_idx = batch_start + local_best_idx[0]
            best_theta = theta_steps[local_best_idx[1]]
    
    # Get the best configuration
    best_combo = all_combos[best_combo_idx]
    best_centers = np.concatenate([base_centers, ring2_arr[list(best_combo)]], axis=0)
    
    # Refinement: fine-tune theta around best_theta
    if time.time() < deadline - safety_margin:
        step_size = (math.pi / 3) / n_theta
        window = 3 * step_size
        fine_theta_steps = np.linspace(best_theta - window, best_theta + window, 2000, endpoint=False)
        # Clip
        fine_theta_steps = fine_theta_steps[(fine_theta_steps >= 0) & (fine_theta_steps <= math.pi / 3)]
        
        if len(fine_theta_steps) > 0:
            # Recompute for best combo only
            best_verts = all_verts[best_combo_idx]  # (66, 2)
            
            k_angles = np.arange(6) * math.pi / 3
            all_angles_fine = fine_theta_steps[:, np.newaxis] + k_angles[np.newaxis, :]
            normals_fine = np.stack([np.cos(all_angles_fine), np.sin(all_angles_fine)], axis=-1)  # (n_fine, 6, 2)
            
            dots_fine = np.einsum('pi,tpq->pt', best_verts, normals_fine)  # (n_fine, 6)
            abs_dots_fine = np.abs(dots_fine)
            max_over_verts_fine = np.max(abs_dots_fine, axis=-1)  # (n_fine, 6)
            max_over_normals_fine = np.max(max_over_verts_fine, axis=-1)  # (n_fine,)
            S_fine = (2.0 / math.sqrt(3)) * max_over_normals_fine
            
            fine_best_idx = np.argmin(S_fine)
            if S_fine[fine_best_idx] < best_S:
                best_S = S_fine[fine_best_idx]
                best_theta = fine_theta_steps[fine_best_idx]
    
    # Even finer refinement
    if time.time() < deadline - safety_margin:
        step_size = (math.pi / 3) / 2000
        window = 2 * step_size
        finer_theta_steps = np.linspace(best_theta - window, best_theta + window, 5000, endpoint=False)
        finer_theta_steps = finer_theta_steps[(finer_theta_steps >= 0) & (finer_theta_steps <= math.pi / 3)]
        
        if len(finer_theta_steps) > 0:
            best_verts = all_verts[best_combo_idx]
            
            k_angles = np.arange(6) * math.pi / 3
            all_angles_finer = finer_theta_steps[:, np.newaxis] + k_angles[np.newaxis, :]
            normals_finer = np.stack([np.cos(all_angles_finer), np.sin(all_angles_finer)], axis=-1)
            
            dots_finer = np.einsum('pi,tpq->pt', best_verts, normals_finer)
            abs_dots_finer = np.abs(dots_finer)
            max_over_verts_finer = np.max(abs_dots_finer, axis=-1)
            max_over_normals_finer = np.max(max_over_verts_finer, axis=-1)
            S_finer = (2.0 / math.sqrt(3)) * max_over_normals_finer
            
            finer_best_idx = np.argmin(S_finer)
            if S_finer[finer_best_idx] < best_S:
                best_S = S_finer[finer_best_idx]
                best_theta = finer_theta_steps[finer_best_idx]
    
    # Format output
    inner_rows = []
    for cx, cy in best_centers:
        inner_rows.append([float(cx), float(cy), 0.0])
    
    center = (0.0, 0.0)
    side = float(best_S)
    angle_deg = float(math.degrees(best_theta))
    
    return inner_rows, center, side, angle_deg
