import time
import numpy as np
from itertools import combinations, permutations

def get_edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return sum(n - 1 - k for k in range(i)) + (j - i - 1)

def precompute_structure(n):
    """
    Precompute all 6-cycles and the mapping from edge to cycles.
    Returns:
        num_edges: int
        edge_to_cycles: list of np.array, where edge_to_cycles[e] is array of cycle indices containing edge e.
    """
    num_edges = n * (n - 1) // 2
    
    # Generate all 6-cycles
    # A 6-cycle is defined by 6 distinct vertices.
    # We generate canonical cycles to avoid duplicates.
    # Canonical form: smallest vertex is 0 (relative to the set), and the second smallest is the neighbor of 0 in the direction of increasing index?
    # Actually, simpler: generate all permutations of 5 vertices attached to 0, and keep only those where the sequence is "canonical".
    # Or just generate all combinations of 6 vertices, and for each, generate all 60 cycles, and add them to a set of frozensets of edges?
    # Set of frozensets of edges is memory heavy.
    
    # Better:
    # For each combination of 6 vertices, there are 60 distinct cycles.
    # We can just iterate through all combinations and all 60 cycles, and add the edge tuple to a list.
    # Then deduplicate?
    # No, the 60 cycles on a set of 6 vertices are distinct subgraphs.
    # So we don't need to deduplicate across different vertex sets.
    # Within a vertex set, the 60 cycles are distinct.
    
    cycles_list = []
    
    # To speed up, we can precompute the 60 edge-tuples for a generic 6-vertex set {0,1,2,3,4,5}
    base_cycles = set()
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
        
    base_cycles_list = list(base_cycles)
    # There should be 60 base cycles.
    
    for comb in combinations(range(n), 6):
        mapping = {k: v for k, v in enumerate(comb)}
        for b_cycle in base_cycles_list:
            mapped_cycle = tuple(mapping[v] for v in b_cycle)
            edges = []
            for k in range(6):
                u = mapped_cycle[k]
                v = mapped_cycle[(k + 1) % 6]
                idx = get_edge_index(u, v, n)
                edges.append(idx)
            cycles_list.append(tuple(edges))
            
    num_cycles = len(cycles_list)
    
    # Build edge to cycles map
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cycle in enumerate(cycles_list):
        for e in cycle:
            edge_to_cycles[e].append(c_idx)
            
    # Convert to numpy arrays for fast access
    edge_to_cycles_np = [np.array(lst, dtype=np.int32) for lst in edge_to_cycles]
    
    return num_edges, num_cycles, edge_to_cycles_np

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Try n from 19 down to 17
    # If we find a valid coloring for 19, return it.
    # If not, try 18, then 17.
    
    for n in range(19, 16, -1):
        if time.time() > deadline:
            break
            
        try:
            num_edges, num_cycles, edge_to_cycles_np = precompute_structure(n)
        except MemoryError:
            continue
            
        if num_cycles == 0:
            # n < 6
            return {'n': n, 'colors': [0] * (n*(n-1)//2)}
            
        # counts[cycle, color]
        # int8 is enough (max 6)
        counts = np.zeros((num_cycles, 4), dtype=np.int8)
        
        best_penalty = float('inf')
        best_colors = None
        
        # Run multiple attempts
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random init
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
            
            # Initialize counts
            counts.fill(0)
            for e in range(num_edges):
                c = colors[e]
                counts[edge_to_cycles_np[e], c] += 1
                
            penalty = np.sum(counts == 6)
            
            if penalty < best_penalty:
                best_penalty = penalty
                best_colors = colors.copy()
                
            if penalty == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            # Local Search
            # Simulated Annealing or Greedy with restarts?
            # Let's use a greedy hill-climbing with random restarts and occasional random walks.
            
            steps = 0
            max_steps = 5000
            stuck_counter = 0
            
            while steps < max_steps and time.time() < deadline:
                steps += 1
                
                # Pick random edge
                e = np.random.randint(0, num_edges)
                c_old = colors[e]
                
                best_delta = 0
                best_c_new = c_old
                
                # Try all other colors
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    
                    affected = edge_to_cycles_np[e]
                    
                    # Current counts for c_old and c_new in affected cycles
                    # We need to know how many cycles were mono c_old (count==6) and how many become mono c_new (count==5)
                    
                    # counts[affected, c_old] is the count BEFORE change.
                    # If it is 6, the cycle was mono c_old. After change, it becomes 5. Not mono.
                    # So penalty decreases by 1 for each such cycle.
                    was_mono_old = np.sum(counts[affected, c_old] == 6)
                    
                    # If count[affected, c_new] is 5, after change it becomes 6. Mono c_new.
                    # So penalty increases by 1 for each such cycle.
                    will_be_mono_new = np.sum(counts[affected, c_new] == 5)
                    
                    delta = -was_mono_old + will_be_mono_new
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_c_new = c_new
                        
                # Apply best move
                if best_c_new != c_old:
                    affected = edge_to_cycles_np[e]
                    counts[affected, c_old] -= 1
                    counts[affected, best_c_new] += 1
                    colors[e] = best_c_new
                    penalty += best_delta
                    
                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_colors = colors.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                        
                    if penalty == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                    # If stuck for a while, do a random move to escape
                    if stuck_counter > 50:
                        e_rand = np.random.randint(0, num_edges)
                        c_rand_old = colors[e_rand]
                        c_rand_new = np.random.choice([c for c in range(4) if c != c_rand_old])
                        
                        affected_rand = edge_to_cycles_np[e_rand]
                        was_mono_rand = np.sum(counts[affected_rand, c_rand_old] == 6)
                        will_be_mono_rand = np.sum(counts[affected_rand, c_rand_new] == 5)
                        delta_rand = -was_mono_rand + will_be_mono_rand
                        
                        counts[affected_rand, c_rand_old] -= 1
                        counts[affected_rand, c_rand_new] += 1
                        colors[e_rand] = c_rand_new
                        penalty += delta_rand
                        
                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_colors = colors.copy()
                            stuck_counter = 0
                            
                        if penalty == 0:
                            return {'n': n, 'colors': colors.tolist()}
                            
                else:
                    # No improving move found for this edge
                    # Try a random move
                    if np.random.random() < 0.1:
                        c_new = np.random.choice([c for c in range(4) if c != c_old])
                        affected = edge_to_cycles_np[e]
                        was_mono_old = np.sum(counts[affected, c_old] == 6)
                        will_be_mono_new = np.sum(counts[affected, c_new] == 5)
                        delta = -was_mono_old + will_be_mono_new
                        
                        counts[affected, c_old] -= 1
                        counts[affected, c_new] += 1
                        colors[e] = c_new
                        penalty += delta
                        
                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_colors = colors.copy()
                            stuck_counter = 0
                            
                        if penalty == 0:
                            return {'n': n, 'colors': colors.tolist()}
                            
        # If we finish attempts for this n without finding 0 penalty, continue to smaller n
        # But we should keep the best found? No, we want max n.
        # If 19 fails, we try 18.
        
    # Fallback to known 17 if we didn't return earlier
    # The loop above tries 19, 18, 17.
    # If 17 fails (unlikely), we return a small valid one.
    # But 17 is known to exist.
    
    # If we reached here, we failed to find 17?
    # Return a small valid coloring to be safe.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
