import time
import random
import itertools
import math

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs?
# Or just compute on the fly?
# N=16, 256 points. Pairs = 256*255/2 ~ 32k.
# Checking isosceles for a triple is O(1).

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_isosceles(p1, p2, p3):
    d12 = dist_sq(p1, p2)
    d13 = dist_sq(p1, p3)
    d23 = dist_sq(p2, p3)
    return d12 == d13 or d12 == d23 or d13 == d23

def can_add(point_set, p):
    """Check if p can be added to point_set without creating an isosceles triangle."""
    # If point_set has < 2 points, it's always possible
    if len(point_set) < 2:
        return True
    
    # Check against all pairs in point_set
    for a, b in itertools.combinations(point_set, 2):
        if is_isosceles(p, a, b):
            return False
    return True

def is_maximal(point_set):
    """Check if no other point can be added to point_set."""
    s = set(point_set)
    for p in POINTS:
        if p in s:
            continue
        if can_add(point_set, p):
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 280.0  # Safety margin for 300s limit
    
    # 1. Greedy Initialization
    # Heuristic: Pick points that "block" the most other points.
    # A point p blocks q if p and q are part of an isosceles triangle with some r?
    # Actually, let's just use a random greedy start and optimize.
    
    # Better Greedy:
    # Sort points by some heuristic?
    # Let's try a few random restarts of a greedy construction.
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    for restart in range(50):
        if time.time() - start_time > time_limit * 0.2:
            break
            
        random.seed(restart)
        current_set = []
        # Random order greedy
        points = POINTS[:]
        random.shuffle(points)
        
        for p in points:
            if can_add(current_set, p):
                current_set.append(p)
        
        # Local Search Optimization
        # Try to minimize size while maintaining maximality
        # 1. Try removing points
        # 2. Try swapping points
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Step 1: Try to remove a point
            # We want to remove a point if the set remains maximal.
            # To be efficient, we can check which points are "critical" for maximality.
            # A point u is critical if removing it allows some v to be added.
            
            # Simple approach: try removing each point
            removable_candidates = []
            for u in current_set:
                new_set = [x for x in current_set if x != u]
                if is_maximal(new_set):
                    removable_candidates.append(u)
            
            if removable_candidates:
                # Remove one (randomly or the one that seems most redundant)
                u_to_remove = random.choice(removable_candidates)
                current_set = [x for x in current_set if x != u_to_remove]
                improved = True
                continue
            
            # Step 2: Try swaps if no removal is possible
            # If we can't remove, we might need to swap to enable a removal.
            # Swap u in current_set with v not in current_set.
            # Condition: new_set is valid and maximal.
            
            # To speed up, we can restrict swaps to points that are "close" in some sense or random.
            # With N=16, 256 points, this might be slow if done exhaustively.
            # Let's try a limited number of random swaps.
            
            s_set = set(current_set)
            candidates_out = current_set[:]
            random.shuffle(candidates_out)
            
            swapped = False
            for u in candidates_out:
                if time.time() - start_time > time_limit:
                    break
                
                # Find a valid v to swap with u
                # v must be addable to current_set \ {u}
                # And the new set must be maximal.
                
                base_set = [x for x in current_set if x != u]
                
                # Find all v that can be added to base_set
                valid_vs = []
                for v in POINTS:
                    if v in s_set:
                        continue
                    if can_add(base_set, v):
                        valid_vs.append(v)
                
                if not valid_vs:
                    continue
                
                # Try a few random valid_vs
                random.shuffle(valid_vs)
                for v in valid_vs[:10]: # Limit to 10 trials per u
                    new_set = base_set + [v]
                    if is_maximal(new_set):
                        current_set = new_set
                        s_set = set(current_set)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
            
            if not improved:
                # Stuck in local minimum
                # Try a random perturbation?
                # Or just break and try next restart
                break
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found a very small set, we can tighten the time limit for subsequent restarts
            if best_size <= 20: # Just a guess, adjust if needed
                pass

    # Final check and return
    if best_set is None:
        # Fallback to a simple greedy if something went wrong
        best_set = []
        for p in POINTS:
            if can_add(best_set, p):
                best_set.append(p)
        
    return best_set

# Run construct to ensure it works? No, the evaluator calls it.
# But we can test locally if this was a script.
