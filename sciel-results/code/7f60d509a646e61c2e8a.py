import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Target number of points
    n_target = 32
    
    # Helper to calculate distance squared
    def dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db
    
    # Helper to get neighbors
    def get_neighbors(p):
        a, b = p
        return [
            (a+1, b), (a-1, b),
            (a, b+1), (a, b-1),
            (a+1, b-1), (a-1, b+1)
        ]
    
    # Helper to count edges
    def count_edges(points):
        p_set = set(points)
        edges = 0
        for p in points:
            for n in get_neighbors(p):
                if n in p_set:
                    edges += 1
        return edges // 2

    # Greedy Construction
    # Start with (0,0)
    points = [(0,0)]
    point_set = set(points)
    
    # Candidate generation
    # We maintain a set of candidates (neighbors of points)
    candidates = set()
    for p in points:
        for n in get_neighbors(p):
            candidates.add(n)
            
    while len(points) < n_target:
        best_candidate = None
        max_neighbors = -1
        # To ensure determinism or specific tie-breaking, we can sort or pick first max
        # But random tie-breaking might help escape local minima in greedy phase?
        # For now, deterministic: pick the one with max neighbors, then min (a,b)
        for c in candidates:
            if c in point_set:
                continue
            nbr_count = 0
            for n in get_neighbors(c):
                if n in point_set:
                    nbr_count += 1
            if nbr_count > max_neighbors:
                max_neighbors = nbr_count
                best_candidate = c
            elif nbr_count == max_neighbors:
                # Tie breaker: closer to origin?
                if abs(c[0]) + abs(c[1]) < abs(best_candidate[0]) + abs(best_candidate[1]):
                    best_candidate = c
        
        if best_candidate is None:
            break
            
        points.append(best_candidate)
        point_set.add(best_candidate)
        
        # Update candidates
        for n in get_neighbors(best_candidate):
            candidates.add(n)
            
    # Local Search to improve edge count
    current_edges = count_edges(points)
    best_points = points[:]
    best_edges = current_edges
    
    # Simple hill climbing
    # Random swaps or moves
    # Move: pick a point, move it to a random empty neighbor or random location near centroid
    # Swap: pick two points, swap them? No, must remain distinct.
    # Better: Pick a point, remove it, add a new point from candidates.
    
    # We will do a limited number of iterations
    max_iters = 5000
    iter_count = 0
    
    # Prepare a list of points for efficient indexing
    # We'll use a set for fast lookup
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, len(points) - 1)
        p_old = points[idx]
        
        # Remove it
        points.pop(idx)
        point_set.remove(p_old)
        
        # Find best replacement
        # Candidates are neighbors of the remaining points, plus some random points near the shape
        # To keep it connected-ish, we look at neighbors of the current set
        # But we can also allow jumps if we want to explore more
        # For now, let's stick to neighbors of the set to maintain connectivity roughly
        
        replacement_candidates = set()
        for p in points:
            for n in get_neighbors(p):
                if n not in point_set:
                    replacement_candidates.add(n)
        
        if not replacement_candidates:
            # Restore
            points.append(p_old)
            point_set.add(p_old)
            iter_count += 1
            continue
            
        # Pick a few random candidates to evaluate
        # Or evaluate all if small
        # n=32, candidates ~ 100-200. Evaluating all is fast.
        best_new_p = None
        best_new_edges = current_edges - 1 # Pessimistic
        
        # We can optimize by only checking edges affected by p_old and new_p
        # But count_edges is O(n^2) = 1000 ops. 5000 * 1000 = 5e6. Fast.
        
        # Let's sample 10 candidates to be safe on time
        sample_size = min(10, len(replacement_candidates))
        sampled = random.sample(list(replacement_candidates), sample_size)
        
        for new_p in sampled:
            points.append(new_p)
            point_set.add(new_p)
            
            e = count_edges(points)
            
            if e > best_new_edges:
                best_new_edges = e
                best_new_p = new_p
            else:
                # If not best, remove and try next?
                # We need to restore state for next candidate
                points.pop()
                point_set.remove(new_p)
                
        if best_new_p:
            # Ensure best_new_p is in the set (it might have been popped in the loop above if not the last)
            # Actually, the logic above pops if not best. So if best_new_p is set, it should be the one that remained?
            # No, the loop pops for every non-best candidate.
            # If the best candidate was the last one, it remains.
            # If it was an earlier one, it was popped.
            # So we must add it back.
            if best_new_p not in point_set:
                points.append(best_new_p)
                point_set.add(best_new_p)
            else:
                # It is already there (if it was the last sampled)
                pass
                
            if best_new_edges > best_edges:
                best_edges = best_new_edges
                best_points = points[:]
                current_edges = best_new_edges
            else:
                # Revert to previous state if not improved
                # Remove best_new_p
                if best_new_p in point_set:
                    points.remove(best_new_p)
                    point_set.remove(best_new_p)
                # Add back p_old
                points.append(p_old)
                point_set.add(p_old)
                # current_edges remains the same as before the move attempt?
                # Yes, because we reverted.
        
        iter_count += 1
        
    # Return the best found
    # Ensure exactly 32 points
    # If local search failed to improve, return greedy result
    result = best_points if best_edges > current_edges else points
    
    # Final check and formatting
    # Convert to list of lists
    return [[a, b] for a, b in result]

