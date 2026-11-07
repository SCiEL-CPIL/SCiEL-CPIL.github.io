"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute vertex masks
    vmask = [1 << i for i in range(n)]
    
    def distinct_degrees(subset):
        """Compute number of distinct degrees in induced subgraph on subset."""
        smask = 0
        for v in subset:
            smask |= vmask[v]
        deg_set = set()
        for v in subset:
            d = (adj_mask[v] & smask).bit_count()
            deg_set.add(d)
        return len(deg_set)
    
    best_subset = list(range(min_v))
    best_count = distinct_degrees(best_subset)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Exhaustive enumeration of all subsets of size 8 to 17
    for size in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        # Early exit: max possible distinct degrees for size k is k
        if best_count >= size:
            continue
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            cnt = distinct_degrees(subset)
            if cnt > best_count:
                best_count = cnt
                best_subset = list(subset)
                if cnt == size:
                    break
    
    # Local search polish: try swapping vertices to increase distinct degrees
    if time.time() < deadline:
        current = list(best_subset)
        current_set = set(current)
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                for v in range(n):
                    if v in current_set:
                        continue
                    new_set = current_set - {current[i]} | {v}
                    new_list = list(new_set)
                    cnt = distinct_degrees(new_list)
                    if cnt > current_count:
                        current = new_list
                        current_set = new_set
                        current_count = cnt
                        best_count = current_count
                        best_subset = new_list
                        improved = True
                        break
                if improved:
                    break
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
