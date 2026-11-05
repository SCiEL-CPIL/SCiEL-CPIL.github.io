"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_valid_addition(p, S_list):
        """
        Check if point p can be added to the set of points S_list (list of tuples)
        without forming an isosceles triangle.
        """
        # 1. Check distances from p to all points in S_list are distinct
        dists = []
        for q in S_list:
            d = sq_dist(p, q)
            dists.append(d)
        
        if len(set(dists)) != len(dists):
            return False
            
        # 2. Check for isosceles triangles with vertex at any point in S_list
        # For each q in S_list, dist(p, q) must not equal dist(q, r) for any r in S_list, r != q
        n = len(S_list)
        for i in range(n):
            d_pq = dists[i]
            q = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                r = S_list[j]
                if sq_dist(q, r) == d_pq:
                    return False
        return True

    def greedy_construct(order):
        """
        Greedily construct a valid set using the given order of points.
        Returns a set of points.
        """
        S_list = []
        S_set = set()
        
        for p in order:
            if time.time() > deadline:
                break
            if is_valid_addition(p, S_list):
                S_list.append(p)
                S_set.add(p)
                
        return S_set, S_list

    def greedy_expand(S_list, all_points, rng):
        """
        Try to add more points to S_list from all_points.
        Uses a random order for candidates to escape local minima.
        Returns the expanded list.
        """
        # Create a list of candidate points not in S_list
        S_set = set(S_list)
        candidates = [p for p in all_points if p not in S_set]
        
        # Shuffle candidates
        rng.shuffle(candidates)
        
        current_list = S_list.copy()
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(p, current_list):
                current_list.append(p)
                
        return current_list

    # Generate all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    best_set = set()
    best_list = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Randomized Greedy to find a good starting set
    # Run a few initial greedy passes
    num_initial_runs = 20
    for i in range(num_initial_runs):
        if time.time() > deadline - 10:
            break
        shuffled_points = all_points.copy()
        rng.shuffle(shuffled_points)
        
        S_set, S_list = greedy_construct(shuffled_points)
        
        if len(S_set) > best_size:
            best_set = S_set
            best_list = S_list
            best_size = len(S_set)
            
    # Phase 2: Local Search (Remove-Refill)
    # We try to remove one point and see if we can add more than one back.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_list = best_list.copy()
        
        # Shuffle the order in which we try to remove points
        removal_order = current_list.copy()
        rng.shuffle(removal_order)
        
        for p in removal_order:
            if time.time() > deadline - 5:
                break
            
            # Temporarily remove p
            temp_list = [q for q in current_list if q != p]
            
            # Try to expand this reduced set
            expanded_list = greedy_expand(temp_list, all_points, rng)
            
            # If we managed to add more points than we removed (size increased)
            if len(expanded_list) > len(current_list):
                best_set = set(expanded_list)
                best_list = expanded_list
                best_size = len(expanded_list)
                improved = True
                break # Restart the loop with the new best_list
    
    # Return the best set found
    return list(best_set)
# EVOLVE-BLOCK-END
