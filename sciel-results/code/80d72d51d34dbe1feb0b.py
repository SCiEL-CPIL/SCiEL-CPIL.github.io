import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Initial grid
    rows = 5
    cols = 6
    xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            positions.append([xs[j], ys[i]])
            
    positions = np.array(positions[:N])
    
    spacing = min(1.0/cols, 1.0/rows)
    r = spacing / 2.0 * 0.95
    radii = np.full(N, r)
    
    best_sum = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    epsilon = 1e-5
    step_size = 0.05
    stuck_count = 0
    
    for iteration in range(50000): # Reduced main iterations
        if time.time() - start_time > time_limit:
            break
            
        new_radii = radii * (1 + epsilon)
        pos = positions.copy()
        rad = new_radii.copy()
        
        # Vectorized Relaxation
        for _ in range(50): # Reduced relaxation steps
            # Calculate pairwise distances
            # diff[i, j] = pos[i] - pos[j]
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
            
            # Mask for overlaps (i < j to avoid double counting and self)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Check if any overlap
            overlap = dist < min_dist
            overlap = overlap & mask
            
            if not np.any(overlap):
                # Check boundaries
                boundary_violation = (pos < rad[:, np.newaxis]) | (pos > 1 - rad[:, np.newaxis])
                if not np.any(boundary_violation):
                    break
                # Push back
                pos = np.clip(pos, rad[:, np.newaxis], 1 - rad[:, np.newaxis])
                continue
            
            # Push apart
            # We need to update pos.
            # For each overlapping pair, push.
            # Vectorized push is tricky because updates are sequential.
            # We can do one pass of updates.
            
            # Calculate push vectors
            # push_vec[i, j] = (min_dist[i,j] - dist[i,j]) / 2 * (pos[i] - pos[j]) / dist[i,j]
            # Only for overlapping pairs.
            
            push = np.zeros_like(dist)
            push[overlap] = (min_dist[overlap] - dist[overlap]) / 2.0
            
            # Normalize diff
            # Avoid division by zero
            safe_dist = np.where(dist > 1e-10, dist, 1.0)
            norm_diff = diff / safe_dist[:, :, np.newaxis]
            
            # Apply push
            # pos[i] += sum_j push[i,j] * norm_diff[i,j]
            # pos[j] -= sum_i push[i,j] * norm_diff[i,j]
            
            # Sum of pushes for each i
            # push_sum[i] = sum_j push[i,j] * norm_diff[i,j]
            # Note: push is symmetric, norm_diff is antisymmetric.
            
            # We want to add to pos[i] the vector sum of pushes from j where i,j overlap.
            # push_vec[i, j] = push[i, j] * norm_diff[i, j]
            
            push_vec = push[:, :, np.newaxis] * norm_diff
            
            # Sum over j for each i
            # But we only want to apply the push once per pair.
            # If we sum push_vec over j, we get the total force on i.
            # However, push[i,j] is defined for i<j.
            # For i, it receives push from j>i and pushes to j<i.
            # This is getting complicated to vectorize perfectly with the mask.
            
            # Let's stick to the loop for relaxation but optimize it slightly.
            # Or just accept the speed and reduce iterations.
            
            # Let's try a simpler vectorized approach:
            # Just move centers apart by a small fraction of the overlap.
            
            # For simplicity and robustness, let's use the loop but limit N or steps.
            # N=26 is small.
            
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < min_dist[i, j]:
                        if dist[i, j] < 1e-10:
                            pos[i][0] += np.random.uniform(-0.01, 0.01)
                            pos[i][1] += np.random.uniform(-0.01, 0.01)
                        else:
                            push = (min_dist[i, j] - dist[i, j]) / 2.0
                            nx = diff[i, j, 0] / dist[i, j]
                            ny = diff[i, j, 1] / dist[i, j]
                            pos[i][0] += nx * push
                            pos[i][1] += ny * push
                            pos[j][0] -= nx * push
                            pos[j][1] -= ny * push
                        moved = True
            
            # Boundaries
            for i in range(N):
                if pos[i][0] < rad[i]:
                    pos[i][0] = rad[i]
                    moved = True
                if pos[i][0] > 1 - rad[i]:
                    pos[i][0] = 1 - rad[i]
                    moved = True
                if pos[i][1] < rad[i]:
                    pos[i][1] = rad[i]
                    moved = True
                if pos[i][1] > 1 - rad[i]:
                    pos[i][1] = 1 - rad[i]
                    moved = True
            
            if not moved:
                break
        
        # Check validity
        valid = True
        for i in range(N):
            if pos[i][0] < rad[i] - 1e-9 or pos[i][0] > 1 - rad[i] + 1e-9:
                valid = False
                break
            if pos[i][1] < rad[i] - 1e-9 or pos[i][1] > 1 - rad[i] + 1e-9:
                valid = False
                break
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (rad[i] + rad[j])**2
                if dist_sq < min_dist_sq - 1e-18:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            radii = rad
            positions = pos
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
                stuck_count = 0
            else:
                stuck_count += 1
        else:
            idx = np.random.randint(N)
            positions[idx][0] += np.random.uniform(-step_size, step_size)
            positions[idx][1] += np.random.uniform(-step_size, step_size)
            positions[idx][0] = np.clip(positions[idx][0], radii[idx], 1 - radii[idx])
            positions[idx][1] = np.clip(positions[idx][1], radii[idx], 1 - radii[idx])
            stuck_count += 1
            
            if stuck_count > 500:
                positions = best_positions.copy()
                radii = best_radii.copy()
                stuck_count = 0
                step_size *= 1.1
                if step_size > 0.5:
                    step_size = 0.5
                epsilon *= 0.9
                if epsilon < 1e-8:
                    break
                    
    result = np.column_stack((best_positions, best_radii))
    return result
