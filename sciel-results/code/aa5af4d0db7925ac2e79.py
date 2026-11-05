import time
import random

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # Generate candidate pool: Hexagon of radius 10
    R = 10
    pool = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                pool.append((a, b))
    
    # Map point to index in pool for fast lookup?
    # Or just use sets of tuples.
    
    best_score = 0
    best_set = None
    
    n_points = 32
    n_restarts = 50 # Should be plenty fast
    
    for _ in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize random set
        current_set = set(random.sample(pool, n_points))
        
        # Precompute neighbors in pool for each point?
        # Or just calculate on the fly.
        
        current_score = 0
        # Calculate initial score
        points_list = list(current_set)
        for i in range(n_points):
            for j in range(i+1, n_points):
                if is_adjacent(points_list[i], points_list[j]):
                    current_score += 1
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try all possible swaps
            # To speed up, we can iterate through points in set and points in pool
            # But 32 * 300 = 9600 swaps. Each swap O(32) = 300k ops.
            # 9600 * 300k = 2.8 billion ops. Too slow in Python if done naively.
            
            # We need to be smarter or limit the search space.
            # Actually, 32*300 = 9600.
            # For each swap, we check neighbors.
            # Let's optimize the swap evaluation.
            
            # Alternative: Hill climbing with random restarts of the "try all swaps" loop.
            # Or, just try a limited number of random swaps per iteration.
            
            # Let's try random swaps first to escape local minima, then local search.
            # Or just simple hill climbing: find the best swap in one pass.
            
            best_swap_score = current_score
            best_swap_in = None
            best_swap_out = None
            
            # To make this faster, we can restrict the candidate 'out' points
            # to those close to the current set.
            # Or just iterate all. 9600 iterations is small.
            # The cost is in the neighbor counting.
            
            for p_in in current_set:
                # Calculate degree of p_in in current_set
                deg_in = 0
                for p_other in current_set:
                    if p_in != p_other and is_adjacent(p_in, p_other):
                        deg_in += 1
                # deg_in is number of edges removed if we remove p_in
                
                for p_out in pool:
                    if p_out in current_set:
                        continue
                    # Calculate degree of p_out in (current_set - {p_in})
                    # This is deg_out = (neighbors of p_out in current_set) - (1 if p_in is neighbor of p_out else 0)
                    deg_out = 0
                    for p_other in current_set:
                        if p_out != p_other and is_adjacent(p_out, p_other):
                            deg_out += 1
                    # Note: if p_in is adjacent to p_out, it was counted in deg_out.
                    # But p_in is removed, so we should subtract 1 if they are adjacent.
                    if is_adjacent(p_in, p_out):
                        deg_out -= 1
                        
                    new_score = current_score - deg_in + deg_out
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_in = p_in
                        best_swap_out = p_out
            
            if best_swap_in is not None:
                current_set.remove(best_swap_in)
                current_set.add(best_swap_out)
                current_score = best_swap_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    
    # Convert best_set to list of lists
    return [[a, b] for a, b in best_set]
