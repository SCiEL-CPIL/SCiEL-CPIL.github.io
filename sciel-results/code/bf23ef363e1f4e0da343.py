"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import gcd

def construct():
    n = 256
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Build conflict graph: edge between i and j if gcd(i,j) > 1
    # Vertices are integers 1..256
    # We want max weight independent set where weight(v) = v
    
    # Precompute gcd conflicts
    conflict = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(i + 1, n + 1):
            if gcd(i, j) > 1:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy initialization: sort by descending value, add if no conflict with selected
    def greedy_init():
        selected = set()
        for v in range(n, 0, -1):
            if not any(c in selected for c in conflict[v]):
                selected.add(v)
        return selected
    
    best_set = greedy_init()
    best_sum = sum(best_set)
    
    # Improved greedy coloring: color by descending weight, return sum of max per color class
    def greedy_coloring_bound(vertices):
        """Compute upper bound via greedy coloring.
        For each color class, we can pick at most one vertex, so bound = sum of max weight per class."""
        if not vertices:
            return 0
        uncolored = set(vertices)
        bound = 0
        while uncolored:
            color_class = set()
            # Process in descending weight order
            for v in sorted(uncolored, key=lambda x: -x):
                if not any(c in color_class for c in conflict[v]):
                    color_class.add(v)
            bound += max(color_class)
            uncolored -= color_class
        return bound
    
    # Branch and bound with proper include/exclude branching
    def bb_search(candidates, current_sum, current_selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_selected)
            return
        
        # Upper bound using greedy coloring
        ub = current_sum + greedy_coloring_bound(candidates)
        if ub <= best_sum:
            return
        
        # Select vertex to branch on: highest degree in candidates
        v = max(candidates, key=lambda x: len(conflict[x] & candidates))
        
        # Branch 1: Include v (remove v and all its neighbors)
        new_cands = candidates - {v} - conflict[v]
        bb_search(new_cands, current_sum + v, current_selected | {v})
        
        # Branch 2: Exclude v (remove v only)
        new_cands = candidates - {v}
        bb_search(new_cands, current_sum, current_selected)
    
    # Start with all vertices
    all_cands = set(range(1, n + 1))
    bb_search(all_cands, 0, set())
    
    # Final validation
    result = list(best_set)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to primes + 1
                result = [1] + [p for p in range(2, n + 1) if all(p % q != 0 for q in range(2, int(math.isqrt(p)) + 1))]
                break
        else:
            continue
        break
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
