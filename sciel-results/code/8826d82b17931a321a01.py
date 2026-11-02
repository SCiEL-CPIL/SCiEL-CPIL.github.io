"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Hexagonal grid
    # For N=200, a hexagonal grid is denser than square.
    # We can generate a hex grid and take the first N points.
    
    def hex_grid(n_points):
        # Approximate number of rows
        # Area of hex grid ~ n * (sqrt(3)/2) * r^2
        # We want to fit in [-0.5, 0.5]^2
        # Let's just generate a large grid and center it, then pick N closest to center?
        # Or just generate rows.
        
        points = []
        # Spacing s = 1.0 (will scale later)
        s = 1.0
        dy = s * np.sqrt(3) / 2
        rows = 0
        while len(points) < n_points + 10: # Generate a few extra
            y = rows * dy
            offset = 0 if rows % 2 == 0 else s / 2
            x_start = -offset
            # Estimate x range
            x = x_start
            while x < 10: # Large bound
                points.append([x, y])
                x += s
            rows += 1
            if rows > 100: break
            
        pts = np.array(points)
        # Center the cloud
        pts -= pts.mean(axis=0)
        # Sort by distance to origin to get the most compact cluster
        dists = np.sqrt(np.sum(pts**2, axis=1))
        indices = np.argsort(dists)[:n_points]
        return pts[indices]

    x = hex_grid(N)
    
    # Scale to fit inside [-0.5, 0.5] with some margin
    # Find max extent
    max_ext = np.max(np.abs(x))
    scale = 0.45 / max_ext # 0.45 leaves margin for radius
    x *= scale
    
    # 2. Local Search Optimization
    # We want to maximize min_dist = min(min_pairwise, min_boundary)
    
    def get_min_dist(pts):
        # Pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Boundary
        # dist to boundary = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
        # Actually, for a square [-0.5, 0.5], dist to boundary for point (px, py) is:
        # min(px + 0.5, 0.5 - px, py + 0.5, 0.5 - py)
        b_dist = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        min_bound = np.min(b_dist)
        
        return min(min_pair, min_bound), min_pair, min_bound

    current_min, _, _ = get_min_dist(x)
    
    max_iters = 2000
    kick_threshold = 50
    no_improve = 0
    
    for i in range(max_iters):
        # Calculate forces/moves
        # We will move points to increase their local min distance.
        
        # Recompute distances to find bottlenecks
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        b_dist = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        
        # For each point, find its nearest neighbor and distance to boundary
        # Nearest neighbor index
        min_pair_dist = np.min(dist, axis=1)
        min_pair_idx = np.argmin(dist, axis=1)
        
        # Effective local distance for point i
        # It is limited by min(min_pair_dist[i], b_dist[i])
        
        # Move strategy:
        # For each point i, compute a displacement vector.
        # If boundary is limiting, move towards center.
        # If pair is limiting, move away from neighbor.
        
        moves = np.zeros_like(x)
        
        # Boundary push
        # If b_dist[i] < min_pair_dist[i], we are close to boundary.
        # Direction: towards center (0,0)
        # Vector from boundary to center is just -x[i] roughly? 
        # Better: gradient of distance to boundary.
        # d(0.5-x)/dx = -1, d(x+0.5)/dx = 1.
        # We want to increase distance.
        # If x[i][0] > 0, we are closer to right boundary (0.5). Move left (-1).
        # If x[i][0] < 0, we are closer to left boundary (-0.5). Move right (+1).
        
        # Let's calculate the gradient of the boundary distance.
        # b_dist_i = min(0.5-x, x+0.5, 0.5-y, y+0.5)
        # If 0.5-x is min, grad is (-1, 0).
        # If x+0.5 is min, grad is (1, 0).
        # etc.
        
        # Simpler: just move towards center if close to boundary.
        close_to_boundary = b_dist < min_pair_dist
        # Move towards origin
        moves[close_to_boundary] -= x[close_to_boundary] * 0.1
        
        # Pair repulsion
        # If min_pair_dist[i] <= b_dist[i], we are close to a neighbor.
        close_to_pair = ~close_to_boundary
        
        # For those, move away from nearest neighbor
        # Neighbor position: x[min_pair_idx[i]]
        # Vector: x[i] - x[neighbor]
        # Normalize and add to move
        
        neighbor_indices = min_pair_idx[close_to_pair]
        current_indices = np.arange(N)[close_to_pair]
        
        vecs = x[current_indices] - x[neighbor_indices]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-6
        vecs = vecs / norms
        
        # Add repulsion
        moves[current_indices] += vecs * 0.1
        
        # Add small random jitter to break symmetry and escape local minima
        moves += np.random.normal(0, 0.001, x.shape)
        
        # Apply moves
        x_new = x + moves
        
        # Check validity (stay in bounds)
        # Clip to [-0.5, 0.5]
        x_new = np.clip(x_new, -0.5, 0.5)
        
        # Evaluate new min dist
        new_min, _, _ = get_min_dist(x_new)
        
        if new_min > current_min:
            x = x_new
            current_min = new_min
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > kick_threshold:
                # Kick: randomize positions slightly
                x += np.random.normal(0, 0.05, x.shape)
                x = np.clip(x, -0.5, 0.5)
                no_improve = 0
                # Re-evaluate after kick (might be worse, but we try to recover)
                current_min, _, _ = get_min_dist(x)

    # Final refinement: try to push boundary points in if they are limiting
    # And ensure strictly inside
    x = np.clip(x, -0.4999, 0.4999)
    
    return x
# EVOLVE-BLOCK-END
