"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin
    
    # Start with a structured grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    def compute_radius(c):
        """Compute the maximum common radius for given centres."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if c.shape != (N, 3):
            return 0.0
        # Check all centres are within the cube
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 0.0
        # Minimum distance to walls
        wall_min = 0.5 - np.max(np.abs(c), axis=1)
        if np.any(wall_min < -1e-10):
            return 0.0
        r_wall = np.min(wall_min)
        # Minimum pairwise distance
        from scipy.spatial.distance import pdist
        d = pdist(c)
        if len(d) == 0:
            r_pair = np.inf
        else:
            r_pair = np.min(d) / 2.0
        return min(r_wall, r_pair)
    
    def energy(c):
        """Negative radius to minimize (maximize radius)."""
        r = compute_radius(c)
        if r <= 0:
            return 1.0
        return -r
    
    # Local optimization using Nelder-Mead on a subset of points
    # We'll do iterative local optimization
    best_centres = centres.copy()
    best_r = compute_radius(best_centres)
    
    # Try multiple random restarts of local optimization
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the current best slightly
        if restart == 0:
            c = best_centres.copy()
        else:
            c = best_centres + rng.normal(0, 0.01, (N, 3))
            # Clamp to cube
            c = np.clip(c, -0.49, 0.49)
        
        # Optimize in batches to stay within time budget
        # Use a simple gradient-free approach: coordinate descent with small steps
        
        improved = True
        max_iters = 500
        step_size = 0.005
        lr = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute current radius
            r = compute_radius(c)
            
            # Find the most constraining pairs/walls
            # Compute all pairwise distances
            from scipy.spatial.distance import pdist, squareform
            d = pdist(c)
            if len(d) == 0:
                break
            min_d = np.min(d)
            
            # Find indices of closest pairs
            d_matrix = squareform(d)
            np.fill_diagonal(d_matrix, np.inf)
            min_idx = np.argmin(d_matrix)
            i, j = np.unravel_index(min_idx, d_matrix.shape)
            
            # Also check wall constraints
            wall_dists = 0.5 - np.max(np.abs(c), axis=1)
            min_wall_idx = np.argmin(wall_dists)
            
            # Determine if the limiting constraint is pair or wall
            if min_d / 2.0 < wall_dists[min_wall_idx]:
                # Limiting constraint is pair (i, j)
                # Push them apart
                diff = c[i] - c[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    continue
                direction = diff / norm
                # Move i away from j and j away from i
                c[i] += direction * step_size * 0.5
                c[j] -= direction * step_size * 0.5
            else:
                # Limiting constraint is wall for point min_wall_idx
                # Push it inward
                point = c[min_wall_idx]
                # Find which wall is closest
                for k in range(3):
                    if point[k] > 0:
                        # Close to +0.5 wall
                        if 0.5 - point[k] == wall_dists[min_wall_idx]:
                            c[min_wall_idx, k] -= step_size
                            break
                    else:
                        # Close to -0.5 wall
                        if 0.5 + point[k] == wall_dists[min_wall_idx]:
                            c[min_wall_idx, k] += step_size
                            break
            
            # Clamp to cube
            c = np.clip(c, -0.49, 0.49)
            
            # Check if we improved
            new_r = compute_radius(c)
            if new_r > r + 1e-8:
                pass  # Good
            elif new_r < r - 1e-6:
                # Revert
                c[i] -= direction * step_size * 0.5 if min_d / 2.0 < wall_dists[min_wall_idx] else 0
                # Simplified: just break and try smaller step
                step_size *= 0.95
                if step_size < 1e-6:
                    break
            else:
                # No significant change, reduce step size
                step_size *= 0.98
                if step_size < 1e-6:
                    break
        
        # Update best
        final_r = compute_radius(c)
        if final_r > best_r:
            best_r = final_r
            best_centres = c.copy()
    
    # Final validation
    if not np.all(np.isfinite(best_centres)):
        return centres
    if best_centres.shape != (N, 3):
        return centres
    if np.any(np.abs(best_centres) > 0.5 + 1e-10):
        return centres
    
    return best_centres
# EVOLVE-BLOCK-END
