"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import itertools

DEADLINE_MARGIN = 10.0

def construct():
    start_time = time.time()
    deadline = start_time + 300.0 - DEADLINE_MARGIN
    
    # Precompute all points
    all_points = np.array(list(itertools.product(range(N), repeat=2)))
    
    best_set = None
    best_size = N * N + 1
    
    def check_isosceles_fast(point, chosen_coords):
        """
        Vectorized check: does 'point' form an isosceles triangle with any pair in 'chosen_coords'?
        chosen_coords is a numpy array of shape (K, 2).
        """
        if len(chosen_coords) < 2:
            return False
            
        # Distances from point to all chosen points
        # d_i = ||point - chosen_i||^2
        diffs = chosen_coords - point
        d_point_to_chosen = np.sum(diffs**2, axis=1)
        
        # Distances between all pairs of chosen points
        # This is O(K^2) but K is small (< 30 usually)
        # We can optimize by precomputing pair distances for the current set, 
        # but since we add one by one, we just need to check if 
        # d_point_to_chosen[i] == d_point_to_chosen[j] (Isosceles with vertex at point)
        # OR d_point_to_chosen[i] == dist(chosen[i], chosen[j]) (Isosceles with vertex at chosen[i])
        # OR d_point_to_chosen[j] == dist(chosen[i], chosen[j]) (Isosceles with vertex at chosen[j])
        
        # Case 1: Vertex at point.
        # Check if any d_point_to_chosen[i] == d_point_to_chosen[j]
        # Since we just need to know if ANY pair is equal, we can check uniqueness.
        # If all d_point_to_chosen are unique, no isosceles with vertex at point.
        if len(np.unique(d_point_to_chosen)) < len(d_point_to_chosen):
            return True
            
        # Case 2 & 3: Vertex at chosen[i] or chosen[j].
        # We need to check if d_point_to_chosen[i] == dist(chosen[i], chosen[j])
        # or d_point_to_chosen[j] == dist(chosen[i], chosen[j])
        
        # For each pair (i, j), compute dist(i, j)
        # This part is still O(K^2). Let's optimize by only checking pairs.
        K = len(chosen_coords)
        for i in range(K):
            for j in range(i+1, K):
                # dist between chosen[i] and chosen[j]
                diff_ij = chosen_coords[i] - chosen_coords[j]
                d_ij = np.sum(diff_ij**2)
                
                if d_point_to_chosen[i] == d_ij:
                    return True
                if d_point_to_chosen[j] == d_ij:
                    return True
                    
        return False

    def greedy_construction(order):
        chosen = []
        chosen_coords = np.empty((0, 2))
        
        for p in order:
            # Check if p can be added
            if check_isosceles_fast(p, chosen_coords):
                continue
            chosen.append(p)
            # Update coords array
            if len(chosen) == 1:
                chosen_coords = np.array([p])
            else:
                chosen_coords = np.vstack([chosen_coords, [p]])
                
        return chosen

    # Strategy 1: Random Greedy Restarts
    # Use NumPy for shuffling for speed
    while time.time() < deadline:
        # Shuffle points
        order = all_points.copy()
        np.random.shuffle(order)
        
        chosen = greedy_construction(order)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen
            
        # If we find a very small set, we might break early or continue to see if we can do better
        # For N=16, optimal is likely small. Let's keep searching until deadline.
        
    if best_set is None:
        # Fallback
        best_set = greedy_construction(all_points)
        
    return best_set
# EVOLVE-BLOCK-END
