import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Create a hexagonal grid of points
    # Approximate number of rows
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    
    points = []
    count = 0
    for row in range(rows):
        y = row * 0.866 # sqrt(3)/2
        offset = 0.5 if row % 2 != 0 else 0.0
        for col in range(rows):
            x = col + offset
            points.append([x, y])
            count += 1
            if count >= N:
                break
        if count >= N:
            break
    
    pts = np.array(points[:N])
    
    # Normalize to fit in [-0.5, 0.5]^2
    # Find bounds
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)
    
    # Center and scale
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    pts = (pts - center) * scale
    # Add small random jitter to break symmetry
    pts += np.random.uniform(-0.01, 0.01, size=pts.shape)
    
    # Ensure inside bounds (with small margin)
    pts = np.clip(pts, -0.49, 0.49)
    
    best_r = 0.0
    best_pts = pts.copy()
    
    def calc_radius(pts):
        # Min distance between pairs
        # Use broadcasting for efficiency, but N=200 is small enough for O(N^2)
        # dist[i,j] = ||pts[i] - pts[j]||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Min distance to walls
        # Walls are at -0.5 and 0.5
        # dist to left wall: x - (-0.5) = x + 0.5
        # dist to right wall: 0.5 - x
        # dist to bottom wall: y + 0.5
        # dist to top wall: 0.5 - y
        d_wall = np.min([
            np.min(pts[:, 0] + 0.5),
            np.min(0.5 - pts[:, 0]),
            np.min(pts[:, 1] + 0.5),
            np.min(0.5 - pts[:, 1])
        ])
        
        return min(min_pair, d_wall)
    
    current_r = calc_radius(pts)
    if current_r > best_r:
        best_r = current_r
        best_pts = pts.copy()
        
    # Optimization Loop
    # We use a simple gradient ascent on the minimum distance.
    # To make it differentiable-ish, we sum gradients from all pairs closer than a threshold.
    
    steps = 0
    max_steps = 5000
    lr = 0.01
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Current radius
        d_wall = np.min([
            np.min(pts[:, 0] + 0.5),
            np.min(0.5 - pts[:, 0]),
            np.min(pts[:, 1] + 0.5),
            np.min(0.5 - pts[:, 1])
        ])
        r = min(np.min(dist), d_wall)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Calculate gradient (repulsion)
        # We want to increase the minimum distance.
        # For each point i, sum vectors (pts[i] - pts[j]) / dist[i,j] for j where dist[i,j] is small.
        # We use a soft weighting: weight = 1 if dist < r * 1.1 else 0
        # Or simply sum all repulsions with inverse distance weight.
        
        # Vectorized repulsion calculation
        # Unit vectors from j to i
        # dist is 200x200
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-8, dist, 1.0)
        unit_vecs = diff / safe_dist[:, :, np.newaxis]
        
        # Weight: we prioritize pairs that are too close.
        # If dist < r, they are overlapping or defining the min.
        # Let's use a threshold: if dist < r + 0.01, apply force.
        threshold = r + 0.005
        mask = dist < threshold
        mask = np.where(np.eye(N, dtype=bool), False, mask) # Exclude self
        
        # Sum forces
        # force[i] = sum_j mask[i,j] * unit_vecs[i,j]
        forces = np.sum(mask[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall forces
        # If close to wall, push inward.
        # Left wall: x < -0.5 + r -> push +x
        # Right wall: x > 0.5 - r -> push -x
        # Bottom wall: y < -0.5 + r -> push +y
        # Top wall: y > 0.5 - r -> push -y
        
        wall_forces = np.zeros_like(pts)
        # Left
        mask_l = pts[:, 0] < -0.5 + r + 0.005
        wall_forces[mask_l, 0] += 1.0
        # Right
        mask_r = pts[:, 0] > 0.5 - r - 0.005
        wall_forces[mask_r, 0] -= 1.0
        # Bottom
        mask_b = pts[:, 1] < -0.5 + r + 0.005
        wall_forces[mask_b, 1] += 1.0
        # Top
        mask_t = pts[:, 1] > 0.5 - r - 0.005
        wall_forces[mask_t, 1] -= 1.0
        
        total_forces = forces + wall_forces
        
        # Update
        pts += lr * total_forces
        
        # Clip to bounds (hard constraint)
        pts = np.clip(pts, -0.499, 0.499)
        
        steps += 1
        
        # Decay learning rate
        if steps % 100 == 0:
            lr *= 0.95
            
        # If no improvement in a while, we might be stuck, but for simplicity we just continue.
        
    # Final check
    final_r = calc_radius(best_pts)
    return best_pts
