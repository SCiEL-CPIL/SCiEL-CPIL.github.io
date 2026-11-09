import math
import random
import time

# Precompute neighbors for the triangular lattice
# Points are (a, b) representing a + b*omega
# Neighbors are at distance 1 in the hexagonal metric
_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    """
    Count the number of pairs of points in pts that are at distance exactly 1.
    """
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    """
    Calculate hexagonal distance for coordinates a, b in the triangular lattice.
    """
    return (abs(a) + abs(b) + abs(a+b)) // 2

def construct():
    n = 32
    # We want to maximize edges. Theoretical max is 76.
    
    # Generate a pool of candidate points.
    # A hexagonal region of radius R is a good search space.
    # R=10 gives 181 points, which is manageable for 32 selected points.
    R = 10
    pool = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                pool.append((a,b))
                
    # Initial set: Take points in spiral order (closest to origin first)
    # This creates a compact hexagon-like shape, which is a good starting point
    # because compact shapes tend to have high edge counts.
    def _sort_key(p):
        a, b = p
        hd = _hexdist(a,b)
        # Secondary sort by angle to make it a proper spiral
        x = a + 0.5*b
        y = 0.8660254037844386*b
        ang = math.atan2(y, x)
        return (hd, ang)
        
    pool.sort(key=_sort_key)
    
    # Select the first n points as the initial configuration
    current_set = set(pool[:n])
    current_edges = _count_edges(current_set)
    
    # If we already reached the optimum, return immediately
    if current_edges == 76:
        return [[a,b] for (a,b) in current_set]
        
    # Simulated Annealing
    # Objective: Maximize edge count.
    # We will perform swaps: remove one point from current_set, add one from pool (not in current_set).
    
    # To speed up swaps, we maintain a list of the current points.
    # However, since n is small (32), set operations are fast.
    # We need to map points to indices for O(1) removal if we use a list, 
    # but set difference is O(n) which is fine for n=32.
    
    # Let's use a list for the current points to facilitate random selection of a point to remove.
    current_list = list(current_set)
    
    # Precompute neighbors for all points in the pool to speed up delta calculation?
    # Or just check on the fly? On the fly is 12 lookups, which is very fast.
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Annealing parameters
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999  # Exponential cooling
    
    rng = random.Random(42)
    
    best_edges = current_edges
    best_set = set(current_set)
    
    while time.time() < deadline:
        # Select a random point from the current set to remove
        idx = rng.randrange(n)
        old_pt = current_list[idx]
        
        # Select a random point from the pool to add
        # It must not be in the current set
        # To avoid O(n) check for "in current_set", we could maintain a list of available points,
        # but since pool size is ~180 and set size is 32, the probability of collision is low.
        # We can just pick randomly and retry if it's in the set.
        
        new_pt = None
        for _ in range(10): # Max 10 retries
            candidate = pool[rng.randrange(len(pool))]
            if candidate not in current_set:
                new_pt = candidate
                break
        
        if new_pt is None:
            continue
            
        # Calculate delta in edge count
        # Edges lost: neighbors of old_pt that are in current_set (excluding old_pt itself, which it is)
        # Note: old_pt is in current_set. We are removing it.
        # So we count neighbors of old_pt that are in current_set.
        # Since old_pt is in current_set, we just check its 6 neighbors.
        
        edges_lost = 0
        for da, db in _NEIGH:
            if (old_pt[0]+da, old_pt[1]+db) in current_set:
                edges_lost += 1
                
        # Edges gained: neighbors of new_pt that are in current_set (excluding old_pt, which is being removed)
        # So we check neighbors of new_pt. If a neighbor is in current_set AND is not old_pt, it's a gained edge.
        # If a neighbor IS old_pt, that edge is lost (counted above) and not gained.
        
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (new_pt[0]+da, new_pt[1]+db)
            if nb in current_set and nb != old_pt:
                edges_gained += 1
                
        delta = edges_gained - edges_lost
        
        if delta > 0:
            # Always accept improving moves
            current_set.remove(old_pt)
            current_set.add(new_pt)
            current_list[idx] = new_pt
            current_edges += delta
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_set = set(current_set)
                
            if best_edges == 76:
                break
        else:
            # Accept worsening moves with probability exp(delta / T)
            # delta is negative
            if T > T_min:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_set.remove(old_pt)
                    current_set.add(new_pt)
                    current_list[idx] = new_pt
                    current_edges += delta
                    
        # Cool down
        T *= cooling_rate
        
    # Return the best set found
    return [[a,b] for (a,b) in best_set]
