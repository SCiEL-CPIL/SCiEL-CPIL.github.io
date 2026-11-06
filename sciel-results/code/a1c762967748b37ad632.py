"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import itertools

N = 16

# EVOLVE-BLOCK-START
DEADLINE_MARGIN = 10.0

def construct():
    start_time = time.time()
    deadline = start_time + 300.0 - DEADLINE_MARGIN
    
    all_points = np.array(list(itertools.product(range(N), repeat=2)))
    grid_indices = np.arange(N * N)
    
    # Precompute squared distances between all pairs? 
    # Or just use vectorized checks on the fly.
    
    def check_isosceles_fast(point, chosen_coords):
        """
        Returns True if 'point' forms an isosceles triangle with any pair in 'chosen_coords'.
        """
        if len(chosen_coords) < 2:
            return False
            
        diffs = chosen_coords - point
        d_point_to_chosen = np.sum(diffs**2, axis=1)
        
        # Case 1: Vertex at point.
        # If any distance is repeated, it's isosceles.
        if len(np.unique(d_point_to_chosen)) < len(d_point_to_chosen):
            return True
            
        # Case 2 & 3: Vertex at chosen[i] or chosen[j].
        K = len(chosen_coords)
        for i in range(K):
            for j in range(i+1, K):
                diff_ij = chosen_coords[i] - chosen_coords[j]
                d_ij = np.sum(diff_ij**2)
                
                if d_point_to_chosen[i] == d_ij:
                    return True
                if d_point_to_chosen[j] == d_ij:
                    return True
                    
        return False

    def is_maximal(chosen_coords):
        """
        Check if the set represented by chosen_coords is maximal.
        A set is maximal if no other point can be added without forming an isosceles triangle.
        """
        chosen_set = set(map(tuple, chosen_coords))
        
        for p in all_points:
            if p in chosen_set:
                continue
            
            # Check if p can be added
            if not check_isosceles_fast(p, chosen_coords):
                # If it can be added, the set is NOT maximal
                return False
                
        return True

    def greedy_maximize(chosen_coords):
        """
        Greedily add points to chosen_coords to make it maximal.
        Returns the new maximal set.
        """
        chosen_list = [tuple(p) for p in chosen_coords]
        chosen_set = set(chosen_list)
        coords_arr = np.array(chosen_list) if chosen_list else np.empty((0, 2))
        
        # Shuffle points for randomization? 
        # For LNS kick, we want a specific maximal completion.
        # Standard greedy: iterate all points, add if possible.
        # To allow different outcomes, we can shuffle the order of candidates.
        
        candidates = [p for p in all_points if tuple(p) not in chosen_set]
        np.random.shuffle(candidates)
        
        for p in candidates:
            if not check_isosceles_fast(p, coords_arr):
                chosen_list.append(tuple(p))
                chosen_set.add(tuple(p))
                if len(chosen_list) == 1:
                    coords_arr = np.array([p])
                else:
                    coords_arr = np.vstack([coords_arr, [p]])
                    
        return np.array(chosen_list)

    def reduce_maximal_set(chosen_coords):
        """
        Try to remove points from the maximal set to make it smaller, while maintaining maximality.
        Returns the reduced set.
        """
        current_set = chosen_coords.copy()
        improved = True
        
        while improved:
            improved = False
            # Try removing each point
            # Shuffle order to avoid bias
            indices = list(range(len(current_set)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if len(current_set) <= 1:
                    break
                
                # Create set without this point
                mask = np.ones(len(current_set), dtype=bool)
                mask[idx] = False
                test_set = current_set[mask]
                
                # Check if test_set is still maximal
                if is_maximal(test_set):
                    current_set = test_set
                    improved = True
                    # Break to restart the loop with the new set
                    break
                    
        return current_set

    # Main Search Loop
    best_set = None
    best_size = N * N + 1
    
    # Initial Greedy to get a valid maximal set
    # We do this once to have a starting point, then use LNS
    
    # Generate a few initial random maximal sets to pick a good starting point
    initial_candidates = []
    while time.time() < deadline and len(initial_candidates) < 5:
        order = all_points.copy()
        np.random.shuffle(order)
        
        chosen = []
        chosen_coords = np.empty((0, 2))
        for p in order:
            if not check_isosceles_fast(p, chosen_coords):
                chosen.append(p)
                if len(chosen) == 1:
                    chosen_coords = np.array([p])
                else:
                    chosen_coords = np.vstack([chosen_coords, [p]])
        
        # Ensure it's maximal (greedy usually produces maximal if we iterate all, 
        # but since we stopped at end of list, it is maximal wrt the list order? 
        # Actually, greedy on all points produces a maximal set.)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
            
        initial_candidates.append(np.array(chosen))

    if not initial_candidates:
        return best_set if best_set else []

    # Start LNS from the best initial candidate
    current_set = initial_candidates[np.argmin([len(s) for s in initial_candidates])]
    
    # LNS Loop
    while time.time() < deadline:
        # 1. Reduction Phase
        reduced_set = reduce_maximal_set(current_set)
        
        if len(reduced_set) < len(current_set):
            current_set = reduced_set
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                
        # 2. Kick Phase (if stuck or periodically)
        # Try to kick out 2-4 points
        kick_size = random.randint(2, 4)
        if len(current_set) > kick_size:
            kick_indices = np.random.choice(len(current_set), kick_size, replace=False)
            mask = np.ones(len(current_set), dtype=bool)
            mask[kick_indices] = False
            kicked_out_set = current_set[mask]
            
            # Greedily maximize the kicked-out set
            new_set = greedy_maximize(kicked_out_set)
            
            # Check if new_set is valid (it should be, as greedy ensures no isosceles)
            # And check size
            if len(new_set) < len(current_set):
                current_set = new_set
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
            elif len(new_set) == len(current_set) and random.random() < 0.1:
                # Accept equal size with small prob to escape local minima?
                # Actually, if size is same, we might want to explore different configurations.
                # But we need to maintain maximality. new_set is maximal.
                current_set = new_set

    return best_set
# EVOLVE-BLOCK-END
