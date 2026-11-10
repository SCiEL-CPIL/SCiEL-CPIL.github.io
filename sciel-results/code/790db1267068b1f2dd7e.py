# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    # Generate a triangular lattice
    # Size 15x15 should be enough to contain the optimal 16-point cluster
    # The 19-point hexagon has radius 2. A 15x15 lattice covers radius ~7.
    L = 15
    i_range = np.arange(-L//2, L//2 + 1)
    j_range = np.arange(-L//2, L//2 + 1)
    
    # Create meshgrid
    I, J = np.meshgrid(i_range, j_range, indexing='ij')
    X = I + 0.5 * J
    Y = (np.sqrt(3) / 2) * J
    
    lattice_points = np.column_stack([X.ravel(), Y.ravel()])
    N_lat = lattice_points.shape[0]
    
    # Precompute distance matrix for the lattice
    # This is O(N_lat^2) which is 225^2 ~ 50k, very fast
    D_lat = pdist(lattice_points, metric='euclidean', out=None)
    # Convert condensed form to full matrix for easier indexing? 
    # Or just keep it condensed? 
    # For subset selection, we need distances between selected points.
    # If we have indices `sel` of length 16, we can get all pairs.
    # It's easier to have a full matrix D_full of shape (N_lat, N_lat)
    D_full = squareform(D_lat) # from scipy.spatial.distance import squareform
    
    # Initialize with the 19-point hexagon (radius 2)
    # Points with hexagonal norm <= 2
    # Hex norm for (i,j) in axial coords? 
    # In our lattice, distance from origin (0,0) is sqrt(X^2+Y^2).
    # We want points with distance <= 2 * lattice_spacing.
    # Lattice spacing is 1.
    # So we want points with Euclidean distance from origin <= 2.
    # Actually, the 19-point hexagon consists of points with hex distance <= 2.
    # Let's just pick the 16 points closest to the origin from the lattice?
    # That would be a circle, not a hexagon.
    # Better: pick the 19-point hexagon, then remove 3 points to optimize.
    
    # Identify the 19-point hexagon indices
    # Points in the hexagon of radius 2 are those with max(|i|, |j|, |i+j|) <= 2? 
    # In axial coordinates (q, r), hex distance is (|q| + |r| + |q+r|)/2.
    # Our lattice points are (i, j) where x = i + j/2, y = j*sqrt(3)/2.
    # This is not standard axial.
    # Let's just use Euclidean distance from origin for initialization.
    dists_to_origin = np.linalg.norm(lattice_points, axis=1)
    # The 19 points of the hexagon are the ones with dist <= 2.0 + epsilon?
    # Let's check the distances.
    # Center (0,0) dist 0.
    # Ring 1: 6 points at dist 1.
    # Ring 2: 12 points at dist 2? 
    # In triangular lattice, points at distance 2 from origin:
    # (2,0), (-2,0), (1,1), (-1,-1), (0,2), (0,-2) ... 
    # Actually, let's just sort by distance and take the first 19?
    # The 19th point might be slightly further than 2?
    # No, the hexagon is compact.
    idx_sorted = np.argsort(dists_to_origin)
    init_indices = idx_sorted[:19]
    
    # We need 16 points. Let's start with the 16 closest?
    # Or start with 19 and remove 3?
    # Let's start with the 16 closest to the origin. This forms a roughly circular cluster.
    init_indices_16 = idx_sorted[:16]
    
    best_indices = init_indices_16.copy()
    best_score = compute_score(lattice_points[best_indices])
    
    # Local Search / Simulated Annealing
    # We will try to swap points to improve the score.
    # State: set of 16 indices.
    
    # Precompute list of all indices
    all_indices = np.arange(N_lat)
    
    # To speed up, we can restrict the "add" candidates to points near the current cluster.
    # But N_lat is small, so we can check all.
    
    T = 0.1 # Temperature
    T_min = 0.001
    cooling = 0.999
    
    current_indices = init_indices_16.copy()
    current_score = best_score
    
    # Set of current indices for O(1) lookup
    current_set = set(current_indices.tolist())
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        # Pick a random point in the set to remove
        i_remove = np.random.choice(current_indices)
        
        # Pick a random point NOT in the set to add
        # Filter all_indices to those not in current_set
        # This is slow if done every time. 
        # Since N_lat is small, we can just pick randomly and check.
        while True:
            i_add = np.random.choice(N_lat)
            if i_add not in current_set:
                break
        
        # Perform swap
        new_indices = current_indices.copy()
        new_indices[new_indices == i_remove] = i_add
        # Ensure uniqueness (should be fine)
        
        new_score = compute_score(lattice_points[new_indices])
        
        if new_score > current_score:
            # Accept
            current_indices = new_indices
            current_score = new_score
            current_set.remove(i_remove)
            current_set.add(i_add)
            
            if current_score > best_score:
                best_score = current_score
                best_indices = new_indices.copy()
        else:
            # Metropolis acceptance
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_indices = new_indices
                    current_score = new_score
                    current_set.remove(i_remove)
                    current_set.add(i_add)
        
        T *= cooling
        if T < T_min:
            T = T_min # Or restart with higher T?
            # Let's just keep going with low T to polish
    
    # Final Polish: Hill Climbing
    # Try all single swaps to see if we can improve
    improved = True
    while improved:
        improved = False
        current_set = set(current_indices.tolist())
        for i_remove in current_indices:
            for i_add in all_indices:
                if i_add in current_set:
                    continue
                new_indices = current_indices.copy()
                new_indices[new_indices == i_remove] = i_add
                new_score = compute_score(lattice_points[new_indices])
                if new_score > current_score:
                    current_indices = new_indices
                    current_score = new_score
                    current_set.remove(i_remove)
                    current_set.add(i_add)
                    if current_score > best_score:
                        best_score = current_score
                        best_indices = new_indices.copy()
                    improved = True
                    break
            if improved:
                break
        
        if time.time() > deadline:
            break

    # Return the best points found
    return lattice_points[best_indices]

# EVOLVE-BLOCK-END
