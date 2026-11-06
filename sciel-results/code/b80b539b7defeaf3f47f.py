import time
import random
from itertools import combinations

# Precompute all possible 5-combinations for n up to 26
MAX_N = 26
ALL_COMBS = []
for n in range(5, MAX_N + 1):
    combs = list(combinations(range(n), 5))
    ALL_COMBS.append(combs)

def check_non_2_colorable_fast(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable (i.e., no proper 2-coloring exists).
    Uses bitmask representation for speed.
    edges: list of tuples/lists of 5 vertices.
    """
    if not edges:
        return False
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # If n is small (<= 15), brute force is extremely fast
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return False 
        return True

    else:
        # Backtracking with unit propagation for n > 15
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        colors = [-1] * n
        edge_counts = [[0, 0] for _ in edges]
        
        def backtrack(idx):
            target_v = -1
            target_c = -1
            forced = False
            
            for i in range(len(edges)):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 0
                            forced = True
                            break
                if forced:
                    break
            
            if not forced:
                for v in range(n):
                    if colors[v] == -1:
                        target_v = v
                        break
                if target_v == -1:
                    return True
            
            colors[target_v] = target_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if target_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            if forced:
                return False
            
            other_c = 1 - target_c
            colors[target_v] = other_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if other_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            return False
        
        return not backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Start with K_9^5
    n_init = 9
    init_edges = [list(e) for e in combinations(range(n_init), 5)]
    
    if check_non_2_colorable_fast(n_init, init_edges):
        best_edges = init_edges
        best_n = n_init
        best_m = len(init_edges)
    else:
        n_init = 10
        init_edges = [list(e) for e in combinations(range(n_init), 5)]
        if check_non_2_colorable_fast(n_init, init_edges):
            best_edges = init_edges
            best_n = n_init
            best_m = len(init_edges)
        else:
            return {'n': 0, 'edges': []}

    # Multi-start local search for various n
    # We prioritize n=9, 10, 11 as they are likely to yield the smallest edge counts
    n_candidates = [9, 10, 11, 12]
    
    # Shuffle to add randomness in order if time allows, but keep 9 and 10 high priority
    # Actually, let's just iterate through them.
    
    for n_try in n_candidates:
        if time.time() > deadline - 10:
            break
            
        # Initialize with complete graph
        init_edges_try = [list(e) for e in combinations(range(n_try), 5)]
        if not check_non_2_colorable_fast(n_try, init_edges_try):
            continue
            
        cur_n = n_try
        cur_edges = init_edges_try[:]
        cur_m = len(cur_edges)
        
        # If this starting point is already better, update best
        if cur_m < best_m:
            best_m = cur_m
            best_edges = cur_edges[:]
            best_n = cur_n
            
        # Local Search
        # We use a simple hill-climbing with random restarts within the time budget for this n
        # Or a single long run with swaps to escape local minima.
        
        # Precompute all possible edges for this n
        all_possible_try = set(tuple(e) for e in combinations(range(n_try), 5))
        
        # Parameters
        stagnation_limit = 500
        stagnation_count = 0
        restarts = 0
        max_restarts = 10
        
        while time.time() < deadline - 5:
            if cur_m > 0:
                idx = random.randint(0, cur_m - 1)
                e_to_remove = cur_edges[idx]
                new_edges = cur_edges[:idx] + cur_edges[idx+1:]
                
                if check_non_2_colorable_fast(cur_n, new_edges):
                    # Accept removal
                    cur_edges = new_edges
                    cur_m -= 1
                    stagnation_count = 0
                    if cur_m < best_m:
                        best_m = cur_m
                        best_edges = cur_edges[:]
                        best_n = cur_n
                else:
                    # Removal failed. Try a swap to escape local minimum.
                    # Pick a random edge to add (not in current set)
                    current_set = set(tuple(e) for e in cur_edges)
                    available_adds = [e for e in all_possible_try if e not in current_set]
                    
                    if available_adds:
                        e_to_add = random.choice(available_adds)
                        # Swap: remove idx, add e_to_add
                        new_edges_swap = [e for i, e in enumerate(cur_edges) if i != idx]
                        new_edges_swap.append(list(e_to_add))
                        
                        if check_non_2_colorable_fast(cur_n, new_edges_swap):
                            # Accept swap
                            cur_edges = new_edges_swap
                            # cur_m stays same
                            stagnation_count = 0
                        else:
                            # Swap failed
                            stagnation_count += 1
                    else:
                        stagnation_count += 1
            
            # Check for stagnation
            if stagnation_count > stagnation_limit:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Restart with a random subgraph? Or just continue?
                # Let's try a random restart from a random subset of the current best
                # Or just break and try next n
                # Actually, for simplicity, let's just break and try the next n value
                # or restart the local search with a different random seed/state
                # Let's just break out of this n's loop to save time for other n's
                break
            
            # Small sleep to prevent busy-waiting if checks are very fast?
            # No, checks are the bottleneck.
            
    # Final verification
    if best_edges is not None:
        if check_non_2_colorable_fast(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': best_n, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    print("Verifying...")
    assert not check_non_2_colorable_fast(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
