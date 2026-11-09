"""Improved program using FCC lattice packing for 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Generate a face-centered cubic (FCC) lattice of points, filter those inside the cube,
    and select the 100 points using a greedy max-min algorithm to maximize the minimum distance.
    """
    def generate_fcc_points(a, n_range):
        """Generate FCC lattice points with lattice constant a."""
        points = []
        # The lattice points are at (i*a/2, j*a/2, k*a/2) where i+j+k is even.
        # This creates a lattice with nearest neighbor distance a/sqrt(2).
        for i in range(-n_range, n_range + 1):
            for j in range(-n_range, n_range + 1):
                for k in range(-n_range, n_range + 1):
                    if (i + j + k) % 2 == 0:
                        points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
        return np.array(points)

    def greedy_select(points, N):
        """
        Select N points from 'points' to maximize the minimum distance between any two points
        and between any point and the boundary of the cube [-0.5, 0.5]^3.
        """
        if len(points) < N:
            return None
        
        # Calculate distance to boundary for all points
        # dist_to_boundary = min(0.5 - x, 0.5 - y, 0.5 - z) for point (x,y,z) inside [-0.5, 0.5]
        # Since points are generated inside, abs(p) <= 0.5.
        dists_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        
        selected_indices = []
        selected_points = []
        
        # We will maintain a list of distances from each candidate to the closest selected point.
        # And the distance to boundary.
        # current_min_dist[i] = min(dist_to_boundary[i], min_{j in selected} dist(points[i], points[j]))
        
        current_min_dists = dists_to_boundary.copy()
        
        for _ in range(N):
            # Find the point with the maximum current_min_dist
            idx = np.argmax(current_min_dists)
            selected_indices.append(idx)
            selected_points.append(points[idx])
            
            if len(selected_indices) == N:
                break
                
            # Update current_min_dists for all unselected points
            # We need to calculate distance from points[idx] to all other points
            # and update current_min_dists[i] = min(current_min_dists[i], dist(points[i], points[idx]))
            
            # Efficient update:
            # dists = cdist(points, points[idx].reshape(1,3))[:,0]
            # But cdist is expensive if done N times. 
            # We can use KDTree.
            
            # Let's build a KDTree once outside the loop? 
            # No, the set of 'selected' grows. 
            # However, we can just compute distances from the new point to ALL points 
            # and take the min with the existing current_min_dists.
            
            # To avoid rebuilding tree every time, we can just do vectorized distance calculation.
            # N is 100, number of points is ~150-200. 
            # 100 iterations * 200 points * 3 dims = 60,000 ops. Very fast.
            
            diffs = points - points[idx]
            dists_to_new = np.sqrt(np.sum(diffs**2, axis=1))
            
            # Update
            current_min_dists = np.minimum(current_min_dists, dists_to_new)
            
            # Mark selected point as -1 or something so it's not picked again?
            # Actually, dists_to_new[idx] is 0. So current_min_dists[idx] becomes 0.
            # It won't be picked again unless all others are negative (impossible) or 0.
            # But we should mask it to be safe.
            current_min_dists[idx] = -1.0

        return np.array(selected_points)

    best_points = None
    best_score = 0.0

    # Search for optimal lattice constant 'a'
    # a is the parameter in the generator. Nearest neighbor distance is a/sqrt(2).
    # Radius r = a/(2*sqrt(2)).
    # We want to maximize r.
    
    # Coarse search
    for a in np.linspace(0.28, 0.34, 13):
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        
        # Filter points strictly inside or on boundary
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
        
        selected = greedy_select(points, N)
        
        if selected is None:
            continue
            
        # Calculate score
        # Min pair distance
        if len(selected) < 2:
            min_pair_dist = 1.0
        else:
            tree = cKDTree(selected)
            dists, _ = tree.query(selected, k=2)
            min_pair_dist = np.min(dists[:, 1])
            
        # Min boundary distance
        min_boundary_dist = np.min(0.5 - np.max(np.abs(selected), axis=1))
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if radius > best_score:
            best_score = radius
            best_points = selected

    # Fine-tune around the best 'a'
    if best_points is not None:
        # We don't track which 'a' was best, so we re-scan a finer range around the likely optimum.
        # Based on previous logic, optimum is around a=0.315.
        for a in np.linspace(0.30, 0.33, 31):
            n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
            points = generate_fcc_points(a, n_range)
            
            mask = np.all(np.abs(points) <= 0.5, axis=1)
            points = points[mask]
            
            if len(points) < N:
                continue
            
            selected = greedy_select(points, N)
            
            if selected is None:
                continue
                
            if len(selected) < 2:
                min_pair_dist = 1.0
            else:
                tree = cKDTree(selected)
                dists, _ = tree.query(selected, k=2)
                min_pair_dist = np.min(dists[:, 1])
            
            min_boundary_dist = np.min(0.5 - np.max(np.abs(selected), axis=1))
            
            radius = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if radius > best_score:
                best_score = radius
                best_points = selected

    if best_points is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
# EVOLVE-BLOCK-END
