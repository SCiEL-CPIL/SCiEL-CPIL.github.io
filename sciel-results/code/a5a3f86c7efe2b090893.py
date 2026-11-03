"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START
def _generate_bcc_points(n_points):
    """
    Generate initial points based on a Body-Centered Cubic (BCC) lattice.
    BCC has a packing density of ~0.68, higher than simple cubic (~0.52).
    We generate a grid of BCC unit cells and take the first n_points.
    """
    points = []
    # BCC unit cell has 2 atoms. To get ~100 points, we need ~50 cells.
    # A 4x4x4 grid of cells gives 64 cells * 2 = 128 points.
    # We will scale the lattice to fit in the cube [-0.5, 0.5]^3.
    
    # Let's try a 5x5x5 grid of simple cubic cells, adding center points.
    # Actually, let's just generate a dense set of candidate points from BCC
    # and pick the best N or just take the first N after sorting/shuffling.
    
    # Better approach: Generate a 4x4x4 BCC lattice, then scale.
    # 4x4x4 = 64 cells. 2 points/cell = 128 points.
    # We take the first 100.
    
    size = 4
    raw_points = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # Corner
                raw_points.append([i, j, k])
                # Center
                raw_points.append([i + 0.5, j + 0.5, k + 0.5])
    
    raw_points = np.array(raw_points)
    
    # Normalize raw points to be roughly in [0, 1]
    # The extent is [0, size-1+0.5] = [0, 3.5]
    # We want to map this to [-0.45, 0.45] to leave some room for optimization
    # or just [-0.5, 0.5] and let SA handle boundaries.
    
    # Center the lattice
    center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
    raw_points = raw_points - center
    
    # Scale to fit in cube.
    # The max distance from center in raw coords is approx sqrt(3 * (size/2)^2)
    # Let's just scale so that the bounding box fits in [-0.4, 0.4] initially
    # to allow room for expansion.
    max_coord = np.max(np.abs(raw_points))
    scale = 0.4 / max_coord
    raw_points *= scale
    
    # Shuffle to make the selection of first N points less biased towards corners
    idx = np.random.default_rng(42).permutation(len(raw_points))
    raw_points = raw_points[idx]
    
    # Take first N
    return raw_points[:n_points].copy()

def _compute_radius_kdtree(centers):
    """
    Compute the maximum radius for given centers using KDTree for speed.
    r = min( min_dist_to_wall, min_pair_dist / 2 )
    """
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    # Query the nearest neighbor for each point
    # dists[:, 0] is distance to self (0), dists[:, 1] is distance to nearest neighbor
    # We want the global minimum distance between any pair.
    # min_pair_dist = min( dists[:, 1] )
    # Note: cKDTree.query returns (distances, indices)
    # For N=100, this is very fast.
    
    # k=2 to get self and nearest neighbor
    distances, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(distances[:, 1])
    
    return min(min_wall, min_pair_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization
    centers = _generate_bcc_points(N)
    
    # Ensure all points are strictly inside [-0.5, 0.5]
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius_kdtree(best_centers)
    
    current_centers = centers.copy()
    current_radius = best_radius
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.9995  # Cooling rate
    
    T = T_start
    
    # Pre-allocate tree for speed? 
    # cKDTree construction is O(N log N). 
    # We can rebuild it every iteration. For N=100, this is ~microseconds.
    
    while T > T_end and time.time() < deadline:
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Propose a move
        # Step size should be proportional to T or fixed?
        # Let's use a step size that shrinks slightly or is fixed relative to T.
        step_size = T * 2.0 
        move = np.random.normal(0, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += move
        
        # Check boundaries
        if np.any(np.abs(new_centers[i]) > 0.49):
            T *= alpha
            continue
            
        # Calculate new radius
        new_radius = _compute_radius_kdtree(new_centers)
        
        # Metropolis Criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Better, accept
            current_centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Worse, accept with probability exp(delta / T)
            # delta is negative, so exp(delta/T) < 1
            if np.random.random() < np.exp(delta / T):
                current_centers = new_centers
                current_radius = new_radius
                
        T *= alpha
        
    # 3. Final Local Polish (Optional, if time permits)
    # A few steps of hill climbing to clean up
    if time.time() < deadline - 5.0:
        step = 0.005
        for _ in range(100):
            improved = False
            for i in range(N):
                for d in range(3):
                    for sign in [-1, 1]:
                        new_c = current_centers.copy()
                        new_c[i, d] += sign * step
                        if np.any(np.abs(new_c[i]) > 0.49):
                            continue
                        new_r = _compute_radius_kdtree(new_c)
                        if new_r > current_radius + 1e-10:
                            current_centers = new_c
                            current_radius = new_r
                            improved = True
                            if current_radius > best_radius:
                                best_radius = current_radius
                                best_centers = current_centers.copy()
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
                
    return best_centers
# EVOLVE-BLOCK-END
