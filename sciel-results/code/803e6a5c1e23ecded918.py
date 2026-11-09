import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # Precompute all points as a numpy array for fast distance calculations
    # Shape: (M, 2)
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances between all pairs of points
    # dist_matrix[i, j] = squared distance between points[i] and points[j]
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sum(diffs * diffs, axis=2)
    del diffs
    
    # Maximum possible squared distance
    max_dist_sq = 2 * (N - 1) ** 2
    # We need to represent distances up to max_dist_sq
    # Using Python integers as bitsets (unbounded)
    
    best_size = 0
    best_S_list = []
    best_D = {}  # D[p_idx] = Python int bitset where bit d is set if distance d is present
    
    def try_add(S, D, p_idx):
        """
        Attempts to add point p_idx to set S.
        Returns True if successful, updating S and D.
        """
        if p_idx in S:
            return False
            
        if len(S) == 0:
            S.add(p_idx)
            D[p_idx] = 0
            return True
            
        # Get distances from p_idx to all points in S
        S_list = list(S)
        dists = dist_matrix[p_idx, S_list]  # numpy array of squared distances
        
        # Check for duplicate distances from p to any two points in S
        # This handles the case where p is the apex of an isosceles triangle
        dists_int = dists.astype(np.int64)
        unique_dists = np.unique(dists_int)
        if len(unique_dists) != len(dists_int):
            return False
            
        # Build bitset for p_idx's distances
        new_bitset = 0
        for d in dists_int:
            new_bitset |= (1 << int(d))
            
        # Check if any distance is already present in the distance set of existing points
        # This handles the case where an existing point is the apex of an isosceles triangle
        # involving p and another point already in S
        for s_idx in S_list:
            # If any bit in new_bitset is also set in D[s_idx], there's a conflict
            if (new_bitset & D[s_idx]) != 0:
                return False
                
        # Validity confirmed. Update data structures.
        S.add(p_idx)
        D[p_idx] = new_bitset
        
        # Update distance bitsets for all existing points
        for i, s_idx in enumerate(S_list):
            d_val = int(dists_int[i])
            D[s_idx] |= (1 << d_val)
            
        return True

    def remove_point(S, D, p_idx):
        """
        Removes point p_idx from set S and updates D.
        """
        if p_idx not in S:
            return
        S.discard(p_idx)
        D.pop(p_idx, None)
        
        # Update D for remaining points
        for s_idx in S:
            d_sq = int(dist_matrix[p_idx, s_idx])
            D[s_idx] &= ~(1 << d_sq)

    def greedy_pass(order, time_limit):
        """
        Runs a greedy pass with the given order of points.
        """
        S = set()
        D = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
            try_add(S, D, p_idx)
                
        return S, D

    # Phase 1: Random Greedy Restarts
    # The goal is to find a good initial solution quickly.
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_pass(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)
            best_D = dict(D)

    # Phase 2: Local Search with swap-based moves
    # Refine the best solution found so far.
    S = set(best_S_list)
    D = dict(best_D)
    
    no_improve_count = 0
    max_no_improve = 50
    last_improve_time = time.time()
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try adding a point
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if try_add(S, D, p_idx):
                improved = True
                break
        
        # If no add worked, try swap-based local search
        if not improved:
            # Try to swap: remove one point, add another
            if len(S) > 3:
                S_list = list(S)
                random.shuffle(S_list)
                for remove_idx in S_list[:20]:  # Try up to 20 removals
                    # Temporarily remove
                    S.discard(remove_idx)
                    D.pop(remove_idx, None)
                    for s_idx in S:
                        d_sq = int(dist_matrix[remove_idx, s_idx])
                        D[s_idx] &= ~(1 << d_sq)
                    
                    # Try to add a new point
                    new_candidates = [i for i in range(M) if i not in S]
                    random.shuffle(new_candidates)
                    
                    added = False
                    for add_idx in new_candidates:
                        if try_add(S, D, add_idx):
                            added = True
                            break
                    
                    if added:
                        improved = True
                        break
                    else:
                        # Restore the removed point
                        # Re-add distances to all current points
                        for s_idx in S:
                            d_sq = int(dist_matrix[remove_idx, s_idx])
                            D[s_idx] |= (1 << d_sq)
                        S.add(remove_idx)
                        D[remove_idx] = 0
                        for s_idx in S:
                            if s_idx != remove_idx:
                                d_sq = int(dist_matrix[remove_idx, s_idx])
                                D[remove_idx] |= (1 << d_sq)
        
        if improved:
            no_improve_count = 0
            last_improve_time = time.time()
            if len(S) > best_size:
                best_size = len(S)
                best_S_list = list(S)
                best_D = dict(D)
        else:
            no_improve_count += 1
            
            # If we haven't improved in a while, perturb
            if no_improve_count > max_no_improve:
                # Aggressive Perturbation: Remove 2-5 points
                if len(S) > 5:
                    k = random.randint(2, 5)
                    to_remove = random.sample(list(S), min(k, len(S)))
                    for r in to_remove:
                        remove_point(S, D, r)
                else:
                    # If set is very small, full restart
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_pass(order, deadline)
                
                no_improve_count = 0
                last_improve_time = time.time()

    # Return the best solution found
    # Ensure we return tuples of (x, y)
    return [tuple(points[i]) for i in best_S_list]
