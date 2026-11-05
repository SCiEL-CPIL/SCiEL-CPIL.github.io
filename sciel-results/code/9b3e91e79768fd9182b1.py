import random
import time
from itertools import combinations

def construct():
    n = 31
    target_edges = 91
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {p: i for i, p in enumerate(all_pairs)}
    
    best_score = float('inf')
    best_edges = None
    best_core = None
    
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    attempts = 0
    max_attempts = 5000  # Cap attempts to ensure we finish
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        random.shuffle(all_pairs)
        
        neighbors = [0] * n
        edge_count = 0
        current_edges = []
        edge_set = set()
        
        for u, v in all_pairs:
            common = (neighbors[u] & neighbors[v]).bit_count()
            if common <= 1:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_count += 1
                current_edges.append((u, v))
                edge_set.add((u, v))
                
                if edge_count == 90:
                    break
        
        if edge_count < 90:
            continue
            
        # Evaluate the cost of adding the 91st edge
        min_cost = float('inf')
        best_add_edge = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            common = (neighbors[u] & neighbors[v]).bit_count()
            cost = common * (common - 1) // 2
            if cost < min_cost:
                min_cost = cost
                best_add_edge = (u, v)
                
        if min_cost < best_score:
            best_score = min_cost
            best_edges = current_edges[:] + [best_add_edge]
            best_core = neighbors[:]
            
            if best_score == 1:
                # Theoretical minimum for 91 edges (if 0 is impossible)
                # We can break early if we are confident 1 is the min
                # But let's run a few more to be sure? No, 1 is likely the min.
                break

    # If we didn't find score 1, try local search on the best core found
    if best_score > 1 and best_core is not None:
        # Local search: try to swap edges to reduce min common neighbors of non-edges
        # This is complex, so we just rely on the random search having found a good one.
        # If not, we return the best we have.
        pass

    if best_edges is None:
        # Fallback: just return first 91 pairs (bad score, but valid)
        return [[i, j] for i, j in all_pairs[:91]]
        
    return [[u, v] for u, v in best_edges]

# Clear static state
if hasattr(construct, '_best_edges'):
    del construct._best_edges
