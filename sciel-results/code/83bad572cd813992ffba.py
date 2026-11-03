"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def compute_radius_and_bottleneck(centers):
        """Compute radius and identify the bottleneck (closest pair or wall)."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        wall_idx = np.argmin(wall_dists)
        wall_coord = np.argmin(wall_dists[wall_idx])
        
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        pair_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        
        r = min(min_wall, min_pair / 2.0)
        
        return r, (min_pair, pair_idx, min_wall, (wall_idx, wall_coord))

    def hexagonal_packing():
        """Generate a hexagonal packing of 200 circles."""
        # Use a hexagonal lattice with appropriate spacing
        # For N=200, we need a grid that fits 200 points
        # Hexagonal packing: rows with alternating offset
        
        # Try different spacings to find one that fits exactly or close to 200
        best_centers = None
        best_r = 0.0
        
        # Spacing d between centers
        # For hexagonal packing, row spacing = d * sqrt(3)/2
        # Column spacing = d
        # Alternate rows offset by d/2
        
        for d in np.linspace(0.06, 0.10, 20):
            centers = []
            row = 0
            y = -0.5 + d  # Start with margin from edge
            
            while y <= 0.5 - d:
                x_offset = 0.0 if row % 2 == 0 else d / 2.0
                x = -0.5 + d + x_offset
                
                while x <= 0.5 - d:
                    centers.append([x, y])
                    x += d
                
                y += d * math.sqrt(3) / 2.0
                row += 1
            
            centers = np.array(centers)
            
            if len(centers) < N:
                continue
            
            # If we have more than N, select N that maximize minimum distance
            # For simplicity, take first N and let local search improve
            if len(centers) > N:
                # Select N points that are well-spread
                # Use a greedy approach: start with corners, add points that maximize min distance
                selected = []
                remaining = list(range(len(centers)))
                
                # Start with a point near center
                center_dist = np.linalg.norm(centers, axis=1)
                idx = np.argmin(center_dist)
                selected.append(idx)
                remaining.remove(idx)
                
                while len(selected) < N and remaining:
                    # Find point that maximizes minimum distance to already selected
                    min_dists = np.full(len(remaining), np.inf)
                    for i, rem_idx in enumerate(remaining):
                        d = np.min(cdist(centers[rem_idx:rem_idx+1], centers[selected]))
                        min_dists[i] = d
                    
                    best_idx = np.argmax(min_dists)
                    selected.append(remaining[best_idx])
                    remaining.pop(best_idx)
                
                centers = centers[selected]
            
            if len(centers) == N:
                r = compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
        
        if best_centers is None:
            # Fallback: square grid
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_centers = grid[:N]
        
        return best_centers

    def targeted_local_search(centers, max_time=120.0):
        """
        Targeted local search: only move the bottleneck point (closest pair or wall-violating).
        This focuses computation on the most critical constraint.
        """
        centers = centers.copy()
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        t_start = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - t_start < max_time:
            r, bottleneck = compute_radius_and_bottleneck(centers)
            
            if r > best_r + 1e-12:
                best_r = r
                best_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
            
            # Identify which constraint is binding
            min_pair, pair_idx, min_wall, wall_info = bottleneck
            r_pair = min_pair / 2.0
            r_wall = min_wall
            
            if r_pair <= r_wall:
                # Move one of the closest pair
                i, j = pair_idx
                # Try moving point i in several directions
                best_move = None
                best_new_r = r
                step_size = max(0.0001, min_pair * 0.1)
                
                # Try moving i in 8 directions + some random
                for angle in np.linspace(0, 2*np.pi, 17, endpoint=False):
                    for scale in [0.5, 1.0, 2.0]:
                        dx = step_size * scale * math.cos(angle)
                        dy = step_size * scale * math.sin(angle)
                        new_c = centers.copy()
                        new_c[i] += [dx, dy]
                        # Ensure within bounds
                        new_c[i, 0] = np.clip(new_c[i, 0], -0.5, 0.5)
                        new_c[i, 1] = np.clip(new_c[i, 1], -0.5, 0.5)
                        
                        new_r = compute_radius(new_c)
                        if new_r > best_new_r:
                            best_new_r = new_r
                            best_move = (i, dx, dy)
                
                # Also try moving j
                for angle in np.linspace(0, 2*np.pi, 17, endpoint=False):
                    for scale in [0.5, 1.0, 2.0]:
                        dx = step_size * scale * math.cos(angle)
                        dy = step_size * scale * math.sin(angle)
                        new_c = centers.copy()
                        new_c[j] += [dx, dy]
                        new_c[j, 0] = np.clip(new_c[j, 0], -0.5, 0.5)
                        new_c[j, 1] = np.clip(new_c[j, 1], -0.5, 0.5)
                        
                        new_r = compute_radius(new_c)
                        if new_r > best_new_r:
                            best_new_r = new_r
                            best_move = (j, dx, dy)
                
                if best_move is not None:
                    idx, dx, dy = best_move
                    centers[idx] += [dx, dy]
                    centers[idx, 0] = np.clip(centers[idx, 0], -0.5, 0.5)
                    centers[idx, 1] = np.clip(centers[idx, 1], -0.5, 0.5)
                else:
                    # No improvement found, add small random perturbation to escape local minimum
                    idx = i if np.random.random() < 0.5 else j
                    centers[idx] += np.random.randn(2) * 0.001
                    centers[idx, 0] = np.clip(centers[idx, 0], -0.5, 0.5)
                    centers[idx, 1] = np.clip(centers[idx, 1], -0.5, 0.5)
            else:
                # Wall constraint is binding
                wall_idx, wall_coord = wall_info
                # Push point away from wall
                step_size = max(0.0001, min_wall * 0.5)
                new_c = centers.copy()
                if wall_coord == 0:
                    if centers[wall_idx, 0] > 0:
                        new_c[wall_idx, 0] -= step_size
                    else:
                        new_c[wall_idx, 0] += step_size
                else:
                    if centers[wall_idx, 1] > 0:
                        new_c[wall_idx, 1] -= step_size
                    else:
                        new_c[wall_idx, 1] += step_size
                
                new_c[wall_idx, 0] = np.clip(new_c[wall_idx, 0], -0.5, 0.5)
                new_c[wall_idx, 1] = np.clip(new_c[wall_idx, 1], -0.5, 0.5)
                
                new_r = compute_radius(new_c)
                if new_r >= r:
                    centers = new_c
                else:
                    # Add small random perturbation
                    centers[wall_idx] += np.random.randn(2) * 0.001
                    centers[wall_idx, 0] = np.clip(centers[wall_idx, 0], -0.5, 0.5)
                    centers[wall_idx, 1] = np.clip(centers[wall_idx, 1], -0.5, 0.5)
            
            # Check deadline
            if time.time() > deadline - 5:
                break
        
        return best_centers, best_r

    # 1. Initialize with hexagonal packing
    centers = hexagonal_packing()
    
    # 2. Targeted local search
    refined_centers, refined_r = targeted_local_search(centers, max_time=130.0)
    
    # 3. Final check and return
    if not np.all(np.isfinite(refined_centers)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        refined_centers = grid[:N]
    
    return refined_centers

# EVOLVE-BLOCK-END
