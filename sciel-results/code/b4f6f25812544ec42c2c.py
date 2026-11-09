import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # 1. Generate a pool of candidate points.
    # We need a pool large enough to find a good configuration, but small enough
    # to allow efficient search.
    # A 15x15 grid centered at 0 with step 10 gives coordinates in [-70, 70].
    # Total 225 points. This is manageable.
    
    coords = list(range(-70, 71, 10))
    pool = []
    for x in coords:
        for y in coords:
            pool.append((x, y))
    
    # Add some random points to break symmetries and potentially find better configs
    random.seed(42)
    for _ in range(50):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in pool:
            pool.append((x, y))
            
    points = pool
    n_pool = len(points)
    
    # 2. Precompute line incidences for the pool.
    # Map from line key to list of point indices
    line_map = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        a, b = dy, -dx
        c = a * x1 + b * y1
        return (a, b, c)

    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            
    # Correct line_map to unique indices
    for key in line_map:
        line_map[key] = list(set(line_map[key]))
        
    # Precompute point_to_lines for faster evaluation
    point_to_lines = [[] for _ in range(n_pool)]
    for key, pts in line_map.items():
        for p in pts:
            point_to_lines[p].append(key)

    def calc_score(subset):
        """
        Returns (triple_count, is_valid)
        triple_count: number of collinear triples in the subset
        is_valid: True if no 4 points are collinear
        """
        line_counts = {}
        for idx in subset:
            for lkey in point_to_lines[idx]:
                line_counts[lkey] = line_counts.get(lkey, 0) + 1
        
        triples = 0
        for lkey, count in line_counts.items():
            if count >= 4:
                return -1, False # Invalid
            if count == 3:
                triples += 1
        
        return triples, True

    # Local Search
    # Start with a random valid subset
    def random_valid_subset():
        while True:
            # Pick 20 random points
            subset = random.sample(range(n_pool), 20)
            _, valid = calc_score(subset)
            if valid:
                return subset

    best_subset = None
    best_triples = -1
    
    # Generate initial population
    for _ in range(30):
        if time.time() > deadline - 15:
            break
        s = random_valid_subset()
        t, _ = calc_score(s)
        if t > best_triples:
            best_triples = t
            best_subset = s[:]
            
    if best_subset is None:
        # Fallback: just take first 20 that are valid
        # This might fail if pool is bad, but pool is grid-like so likely valid subsets exist
        return [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2],[3,3],[0,3],[1,3],[2,3],[4,0],[4,1],[4,2],[4,3]]

    current_subset = best_subset[:]
    current_triples = best_triples
    
    # Simulated Annealing / Hill Climbing
    # Move: Swap one point in subset with one point outside
    outside = set(range(n_pool)) - set(current_subset)
    
    T = 1.0
    T_min = 0.05
    cooling = 0.995
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        improved = False
        
        # Try a random sample of swaps to keep it fast
        num_swaps = 50
        for _ in range(num_swaps):
            if time.time() > deadline - 5:
                break
            
            i = random.randint(0, 19)
            old_idx = current_subset[i]
            
            # Pick a random candidate from outside
            if not outside:
                break
            new_idx = random.choice(list(outside))
            
            # Perform swap
            new_subset = current_subset[:]
            new_subset[i] = new_idx
            
            new_triples, valid = calc_score(new_subset)
            
            if valid:
                if new_triples > current_triples:
                    # Accept
                    current_subset = new_subset
                    current_triples = new_triples
                    improved = True
                    # Update outside set
                    outside.remove(new_idx)
                    outside.add(old_idx)
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best_subset = current_subset[:]
                else:
                    # Metropolis acceptance
                    delta = new_triples - current_triples
                    if random.random() < pow(2.71828, delta / T):
                        current_subset = new_subset
                        current_triples = new_triples
                        outside.remove(new_idx)
                        outside.add(old_idx)
            else:
                # Invalid, reject
                pass
                
        T *= cooling
        
        if not improved and random.random() < 0.05:
            # Random restart to escape local opt
            if time.time() > deadline - 20:
                break
            s = random_valid_subset()
            t, _ = calc_score(s)
            if t > current_triples:
                current_subset = s
                current_triples = t
                outside = set(range(n_pool)) - set(current_subset)
                if t > best_triples:
                    best_triples = t
                    best_subset = current_subset[:]

    # Convert best_subset indices to coordinates
    final_points = [points[i] for i in best_subset]
    
    # Final verification and return
    # Ensure no 4 collinear
    # (Already checked in calc_score, but double check just in case)
    # The problem requires returning the points.
    
    return [[p[0], p[1]] for p in final_points]
