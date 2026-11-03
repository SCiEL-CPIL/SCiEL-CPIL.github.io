import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
import math

N = 200

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        """Compute the maximum common radius for given centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < -1e-9:
            return 0.0
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def local_search(centers, max_iters=200, time_budget=30.0):
        """
        Local search using bottleneck pair moves with simulated annealing.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        tree = cKDTree(centers)
        
        def get_bottleneck(tree):
            """Get the bottleneck pair and wall distances."""
            # Nearest neighbors
            nn_dist, nn_idx = tree.query(centers, k=2)
            nn_dist = nn_dist[:, 1:]  # Exclude self
            nn_idx = nn_idx[:, 1:]
            min_pair_dist = np.min(nn_dist)
            min_pair_idx = np.argmin(nn_dist)
            min_pair_i = min_pair_idx // (N - 1)
            min_pair_j = nn_idx[min_pair_i][min_pair_idx % (N - 1)]
            
            # Wall distances
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            min_wall_idx = np.argmin(wall_dists)
            
            if min_wall < min_pair_dist / 2.0:
                return 'wall', min_wall_idx, -1, min_wall
            else:
                return 'pair', min_pair_i, min_pair_j, min_pair_dist / 2.0
        
        # Simulated annealing parameters
        T = 0.01
        cooling_rate = 0.999
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            if T < 1e-10:
                break
                
            # Get bottleneck
            btype, i, j, bottleneck_val = get_bottleneck(tree)
            
            # Generate candidate move
            if btype == 'wall':
                # Move point i away from wall
                orig = centers[i].copy()
                # Determine direction to move (away from nearest wall)
                dx = centers[i, 0]
                dy = centers[i, 1]
                
                # Distance to each wall
                d_left = centers[i, 0] + 0.5
                d_right = 0.5 - centers[i, 0]
                d_bottom = centers[i, 1] + 0.5
                d_top = 0.5 - centers[i, 1]
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                if min_d == d_left:
                    direction = np.array([1.0, 0.0])
                elif min_d == d_right:
                    direction = np.array([-1.0, 0.0])
                elif min_d == d_bottom:
                    direction = np.array([0.0, 1.0])
                else:
                    direction = np.array([0.0, -1.0])
                
                # Try a few step sizes
                best_move = None
                best_new_r = -np.inf
                for step in [0.001, 0.002, 0.005, 0.01]:
                    new_pos = orig + direction * step
                    if -0.5 <= new_pos[0] <= 0.5 and -0.5 <= new_pos[1] <= 0.5:
                        centers[i] = new_pos
                        tree = cKDTree(centers)
                        new_r = compute_radius(centers)
                        if new_r > best_new_r:
                            best_new_r = new_r
                            best_move = new_pos.copy()
                if best_move is None:
                    continue
                centers[i] = best_move
                new_r = best_new_r
                moved_idx = i
            else:
                # Move pair (i, j) apart
                orig_i = centers[i].copy()
                orig_j = centers[j].copy()
                direction = orig_i - orig_j
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                
                best_move_i = None
                best_move_j = None
                best_new_r = -np.inf
                for step in [0.001, 0.002, 0.005, 0.01]:
                    new_i = orig_i + direction * step
                    new_j = orig_j - direction * step
                    if (-0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and 
                        -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5):
                        centers[i] = new_i
                        centers[j] = new_j
                        tree = cKDTree(centers)
                        new_r = compute_radius(centers)
                        if new_r > best_new_r:
                            best_new_r = new_r
                            best_move_i = new_i.copy()
                            best_move_j = new_j.copy()
                if best_move_i is None:
                    continue
                centers[i] = best_move_i
                centers[j] = best_move_j
                new_r = best_new_r
                moved_idx = i
            
            # Accept/reject
            if new_r > current_r or np.random.rand() < math.exp((new_r - current_r) / T):
                current_r = new_r
                tree = cKDTree(centers)
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                if btype == 'wall':
                    centers[i] = orig
                else:
                    centers[i] = orig_i
                    centers[j] = orig_j
                tree = cKDTree(centers)
                no_improve += 1
            
            T *= cooling_rate
            
            # If stuck for too long, perturb
            if no_improve > 50:
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.005
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                tree = cKDTree(centers)
                current_r = compute_radius(centers)
                no_improve = 0
                T = max(T, 0.005)
        
        return best_centers, best_r

    def hexagonal_init():
        """Generate a hexagonal lattice initial guess."""
        # For r ≈ 0.0366, d = 2r ≈ 0.073
        # Hexagonal spacing: horizontal d, vertical d*sqrt(3)/2
        d = 0.073
        y_spacing = d * math.sqrt(3) / 2
        
        # Estimate rows and cols
        # Width: 1.0, so cols ≈ 1.0/d + 1 ≈ 15
        # Height: 1.0, so rows ≈ 1.0/y_spacing + 1 ≈ 16
        cols = 15
        rows = 16
        
        pts = []
        total_height = (rows - 1) * y_spacing
        y_start = -total_height / 2
        
        for i in range(rows):
            y = y_start + i * y_spacing
            if i % 2 == 0:
                x_start = -0.5 + d/2
                for j in range(cols):
                    x = x_start + j * d
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for j in range(cols - 1):
                    x = x_start + j * d
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            diff = N - len(pts)
            random_pts = np.random.uniform(-0.45, 0.45, (diff, 2))
            pts = np.vstack([pts, random_pts])
        
        return pts

    def square_init():
        """Generate a square grid initial guess."""
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) > N:
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        return grid

    def random_init():
        """Generate random initial guess."""
        return np.random.uniform(-0.45, 0.45, (N, 2))

    best_centers = None
    best_radius = 0.0

    # Multi-restart strategy
    inits = [
        ("hex", hexagonal_init()),
        ("sq", square_init()),
    ]
    
    for name, init_centers in inits:
        if time.time() > deadline - 40:
            break
            
        # Local search
        polished, r = local_search(init_centers, max_iters=300, time_budget=min(30.0, deadline - time.time() - 10))
        
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # If time remains, try more random inits
    num_random_tries = 0
    while time.time() < deadline - 25:
        if time.time() > deadline - 10:
            break
        init_centers = random_init()
        polished, r = local_search(init_centers, max_iters=200, time_budget=min(15.0, deadline - time.time() - 10))
        
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
        num_random_tries += 1
        if num_random_tries >= 5:
            break

    # Final polish on best
    if best_centers is not None and time.time() < deadline - 15:
        final_centers, final_r = local_search(best_centers, max_iters=100, time_budget=min(10.0, deadline - time.time() - 5))
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final_centers

    # Ensure valid output
    if best_centers is None:
        best_centers = square_init()
        
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation
    final_r = compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to a known valid configuration
        best_centers = square_init()
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
