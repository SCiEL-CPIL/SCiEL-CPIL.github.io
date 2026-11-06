import time
import random
import itertools
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback seed for n=9
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': colors_seed}
    best_n = n_seed
    
    # Try n=18 first with a limited time, then n=17, etc.
    target_ns = [18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # Allocate time
    # We want to give 17 a good chance since it's known to exist.
    # 18 is likely hard/impossible, so give it less time.
    time_alloc = {
        18: 20,
        17: 70,
        16: 10,
        15: 5,
        14: 5,
        13: 2,
        12: 2,
        11: 1,
        10: 1
    }
    
    for n in target_ns:
        if time.time() > deadline - 5: # Reserve 5s for return
            break
            
        t_limit = time_alloc.get(n, 5)
        t_start = time.time()
        
        solution = local_search_optimized(n, t_start + t_limit)
        
        if solution is not None:
            # Found valid coloring for n
            # Since we iterate from high to low, this is the best found so far
            # But wait, if we find 17, we should stop? 
            # Yes, because we are looking for MAX n.
            # If we found 17, we don't need to check 16.
            return {'n': n, 'colors': solution}
            
    return best_result

def local_search_optimized(n, deadline):
    """
    Finds a valid 4-coloring of K_n with no monochromatic C6.
    Returns list of colors or None.
    Uses optimized delta calculation by precomputing 'other edges' for each edge in each cycle.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute: for each edge, list of tuples of the other 5 edges in each C6 containing it.
    # This allows O(1) access to the specific edges that need checking for a delta update.
    edge_other_edges = [[] for _ in range(num_edges)]
    all_cycles = []
    
    vertices = list(range(n))
    
    # Optimization: Precompute permutations of 5 elements once?
    # No, the vertices change, so the edge indices change.
    # But the *structure* of permutations is the same.
    # We can just iterate permutations of range(5) and map to vertices.
    
    # Generate all combinations of 6 vertices
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        # Generate 60 unique cycles
        # Fix v0 at start. Permute v1..v5.
        # Restrict perm[0] < perm[-1] to break reversal symmetry.
        # This yields 60 cycles per combination.
        
        # To speed up, we can hardcode the 60 permutations of indices 1..5?
        # Or just use itertools.permutations and filter.
        # itertools.permutations is C-implemented, so it's fast.
        
        for perm in itertools.permutations([v1, v2, v3, v4, v5]):
            if perm[0] > perm[-1]:
                continue
            
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            
            # Unroll the 6 edge lookups for speed?
            # k=0: v0, v1
            # k=1: v1, v2
            # k=2: v2, v3
            # k=3: v3, v4
            # k=4: v4, v5
            # k=5: v5, v0
            
            # Manual unrolling is risky for correctness, let's keep the loop but optimize lookup.
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            all_cycles.append(edge_indices)
            
            # For each edge in this cycle, add the other 5 to its list
            # We store a tuple of the other 5 indices for fast access
            for i in range(6):
                e_idx = edge_indices[i]
                # Construct tuple of other 5
                # Using list comprehension is okay, or manual?
                # Manual might be faster?
                # Let's try list comprehension first.
                others = [edge_indices[j] for j in range(6) if j != i]
                edge_other_edges[e_idx].append(tuple(others))
                
    if time.time() > deadline:
        return None
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Count initial mono cycles
    # We need to iterate all_cycles
    def count_mono_cycles(col):
        count = 0
        for cycle in all_cycles:
            c0 = col[cycle[0]]
            if col[cycle[1]] == c0 and col[cycle[2]] == c0 and col[cycle[3]] == c0 and col[cycle[4]] == c0 and col[cycle[5]] == c0:
                count += 1
        return count
        
    current_cost = count_mono_cycles(colors)
    if current_cost == 0:
        return colors[:]
        
    best_colors = colors[:]
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # Main search loop
    # We will run until deadline
    # Use a step counter or just time
    
    while time.time() < deadline - 0.5:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        
        # Try changing to a different color
        c_old = colors[e_idx]
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        # Calculate delta
        delta = 0
        # Iterate over precomputed 'other edges' for this edge
        # This avoids iterating over all cycles and checking membership
        for others in edge_other_edges[e_idx]:
            # Check if all others are c_old
            # Check if all others are c_new
            # We can do this with a loop or unrolled?
            # 5 elements.
            
            # Unrolled check for speed
            # o0, o1, o2, o3, o4 = others
            
            all_old = True
            all_new = True
            
            # Manual unroll for 5 elements
            c0 = colors[others[0]]
            if c0 != c_old: all_old = False
            if c0 != c_new: all_new = False
            
            c1 = colors[others[1]]
            if c1 != c_old: all_old = False
            if c1 != c_new: all_new = False
            
            c2 = colors[others[2]]
            if c2 != c_old: all_old = False
            if c2 != c_new: all_new = False
            
            c3 = colors[others[3]]
            if c3 != c_old: all_old = False
            if c3 != c_new: all_new = False
            
            c4 = colors[others[4]]
            if c4 != c_old: all_old = False
            if c4 != c_new: all_new = False
            
            if all_old:
                delta -= 1
            if all_new:
                delta += 1
                
        new_cost = current_cost + delta
        
        # Acceptance criterion
        if new_cost < current_cost:
            # Accept
            colors[e_idx] = c_new
            current_cost = new_cost
            
            if current_cost == 0:
                return colors[:]
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                
        else:
            # Metropolis criterion
            # delta is positive (cost increases) or 0
            if delta > 0:
                # Probability exp(-delta / T)
                # delta is integer, T is float
                # If T is small, this is very small
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[e_idx] = c_new
                    current_cost = new_cost
            else:
                # If delta == 0, it's a neutral move. Accept to explore.
                if random.random() < 0.5:
                    colors[e_idx] = c_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return None
