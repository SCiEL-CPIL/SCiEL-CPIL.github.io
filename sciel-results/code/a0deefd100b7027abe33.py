import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency bitmasks
N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def induced_degrees(subset_mask):
    """Compute degrees of each vertex in the induced subgraph. Returns dict vertex->degree."""
    degs = {}
    for v in range(N):
        if (subset_mask >> v) & 1:
            # Count neighbors also in subset
            common = adj[v] & subset_mask
            degs[v] = bin(common).count('1')
    return degs

def distinct_degree_count(subset_mask):
    degs = induced_degrees(subset_mask)
    return len(set(degs.values()))

def subset_size(mask):
    return bin(mask).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    best_score = 0
    best_subset = None
    
    # Try all subsets of size 8 to 11 (feasible enumeration)
    # C(17,8)=24310, C(17,9)=24310, C(17,10)=19448, C(17,11)=12376 -> total ~80k, fast
    for size in range(8, 12):
        if time.time() > deadline:
            break
        for combo in combinations(range(N), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            dc = distinct_degree_count(mask)
            if dc > best_score:
                best_score = dc
                best_subset = list(combo)
    
    # Now try sizes 12-17 with local search
    if time.time() < deadline:
        # For larger subsets, use randomized local search
        rng = random.Random(42)
        for size in range(12, N + 1):
            if time.time() > deadline:
                break
            # Multiple random starts
            num_starts = 50
            for s in range(num_starts):
                if time.time() > deadline:
                    break
                # Random subset of given size
                chosen = set(rng.sample(range(N), size))
                mask = 0
                for v in chosen:
                    mask |= (1 << v)
                cur_dc = distinct_degree_count(mask)
                
                # Local search: try swaps
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Try all single swaps
                    for v_in in range(N):
                        if (mask >> v_in) & 1:
                            continue
                        for v_out in range(N):
                            if not ((mask >> v_out) & 1):
                                continue
                            new_mask = mask | (1 << v_in) & ~(1 << v_out)
                            new_dc = distinct_degree_count(new_mask)
                            if new_dc > cur_dc:
                                cur_dc = new_dc
                                mask = new_mask
                                improved = True
                                if cur_dc >= N:  # can't do better than N distinct degrees
                                    break
                        if cur_dc >= N:
                            break
                    if cur_dc >= N:
                        break
                
                if cur_dc > best_score:
                    best_score = cur_dc
                    best_subset = [v for v in range(N) if (mask >> v) & 1]
    
    # Also do a broader local search from the best found so far
    if best_subset is not None and time.time() < deadline:
        mask = 0
        for v in best_subset:
            mask |= (1 << v)
        cur_dc = distinct_degree_count(mask)
        
        # Try additions and removals and swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a vertex
            for v in range(N):
                if (mask >> v) & 1:
                    continue
                new_mask = mask | (1 << v)
                new_dc = distinct_degree_count(new_mask)
                if new_dc > cur_dc:
                    cur_dc = new_dc
                    mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            # Try removing a vertex
            for v in range(N):
                if not ((mask >> v) & 1):
                    continue
                new_mask = mask & ~(1 << v)
                if subset_size(new_mask) < 8:
                    continue
                new_dc = distinct_degree_count(new_mask)
                if new_dc > cur_dc:
                    cur_dc = new_dc
                    mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            # Try swaps
            for v_in in range(N):
                if (mask >> v_in) & 1:
                    continue
                for v_out in range(N):
                    if not ((mask >> v_out) & 1):
                        continue
                    new_mask = (mask | (1 << v_in)) & ~(1 << v_out)
                    new_dc = distinct_degree_count(new_mask)
                    if new_dc > cur_dc:
                        cur_dc = new_dc
                        mask = new_mask
                        improved = True
                        break
                if improved:
                    break
        
        if cur_dc > best_score:
            best_score = cur_dc
            best_subset = [v for v in range(N) if (mask >> v) & 1]
    
    # Ensure we have a valid result
    if best_subset is None or len(best_subset) < 8:
        best_subset = list(range(8))
    
    # Return as list of Python ints
    return [int(v) for v in best_subset]
