import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

def _build_data():
    # Map edge index to (u, v)
    edge_endpoints = {}
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            edge_endpoints[idx] = (i, j)
    
    # List of all 4-cycles as tuples of 3 edge indices
    cycles = []
    # For each cycle, we need to know which edges are in it.
    # A 4-cycle on vertices a,b,c,d has edges (a,b), (b,c), (c,d), (d,a) -> 3 such cycles per set of 4 vertices?
    # No, the problem says "all three cycles on every four-vertex subset are checked".
    # The 3 cycles are:
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    # These correspond to the 3 ways to pair up 4 vertices into 2 disjoint edges (perfect matchings)?
    # No, a 4-cycle is a sequence of 4 edges.
    # In K4, there are 3 distinct 4-cycles (Hamiltonian cycles).
    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
    
    import itertools
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle 1: a-b-c-d-a
        e1 = (_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a))
        cycles.append(e1)
        # Cycle 2: a-b-d-c-a
        e2 = (_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a))
        cycles.append(e2)
        # Cycle 3: a-c-b-d-a
        e3 = (_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a))
        cycles.append(e3)
        
    # Precompute for each edge, the list of cycles (indices) it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(idx)
            
    return edge_endpoints, cycles, edge_to_cycles

def _is_rainbow(cycle, colors):
    return len({colors[e] for e in cycle}) == 4

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    edge_endpoints, cycles, edge_to_cycles = _build_data()
    num_cycles = len(cycles)
    
    best_score = 1
    best_colors = [0] * NUM_EDGES
    
    # Helper to check if a specific change is valid given current bad_count
    # We will maintain bad_count incrementally.
    
    def count_bad(colors):
        bad = 0
        for cyc in cycles:
            if _is_rainbow(cyc, colors):
                bad += 1
        return bad

    # Local Search
    for _ in range(1000): # Max restarts
        if time.time() > deadline:
            break
            
        # Initialize with a small number of colors
        # Try K=2, 3, etc.
        K_init = random.choice([2, 3, 4])
        colors = [random.randint(0, K_init - 1) for _ in range(NUM_EDGES)]
        
        # Repair phase: eliminate all rainbow cycles
        # While there are bad cycles, fix one
        bad_count = count_bad(colors)
        repair_steps = 0
        while bad_count > 0 and repair_steps < 1000:
            if time.time() > deadline:
                break
            
            # Find a random bad cycle
            # Instead of scanning all, we could maintain a list, but scanning 5460 is fast
            bad_cycle_idx = -1
            for i in range(num_cycles):
                if _is_rainbow(cycles[i], colors):
                    bad_cycle_idx = i
                    break
            
            if bad_cycle_idx == -1:
                bad_count = 0
                break
                
            cyc = cycles[bad_cycle_idx]
            e1, e2, e3, e4 = cyc
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            
            # To fix, make two colors equal.
            # Pick a random edge in the cycle and change it to match another.
            # E.g., change e1 to c2.
            target_edge = random.choice([e1, e2, e3, e4])
            target_color = None
            for e in [e1, e2, e3, e4]:
                if e != target_edge:
                    target_color = colors[e]
                    break
            
            old_color = colors[target_edge]
            colors[target_edge] = target_color
            
            # Update bad_count incrementally? 
            # It's safer to just recount if repair steps are few, or use incremental.
            # Let's just recount for simplicity in repair, it's fast enough.
            bad_count = count_bad(colors)
            repair_steps += 1
            
        if bad_count > 0:
            continue
            
        # Valid coloring found.
        current_score = len(set(colors))
        
        # Improvement phase
        # Try to increase the number of colors.
        # Strategy: Pick a random edge, try to change it to a new color.
        # If valid, keep it.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle edges to avoid bias
            edges_to_try = list(range(NUM_EDGES))
            random.shuffle(edges_to_try)
            
            for e in edges_to_try:
                if time.time() > deadline:
                    break
                
                # Try to assign a new color to edge e
                # The new color should be one that is not currently used, to increase the count.
                # Or just a random color that might be new.
                # Let's try a color from a larger range, e.g., 0..200, but we only care about distinct count.
                # Actually, if we assign a color already in use, the count doesn't increase.
                # So we should try to assign a color NOT in the current set.
                
                used_colors = set(colors)
                # Generate a new color
                # We can just pick a random integer. If it's in used_colors, it doesn't help increase count.
                # So pick from range(200) until we find one not in used_colors?
                # Or just pick random.randint(0, 200) and check if it increases count.
                
                # To maximize speed, let's just try a few random new colors.
                for _ in range(10):
                    new_color = random.randint(0, 200)
                    if new_color in used_colors:
                        continue
                    
                    old_color = colors[e]
                    colors[e] = new_color
                    
                    # Check validity
                    # Only cycles containing e can be affected
                    valid = True
                    for cyc_idx in edge_to_cycles[e]:
                        if _is_rainbow(cycles[cyc_idx], colors):
                            valid = False
                            break
                    
                    if valid:
                        # Accept
                        improved = True
                        current_score = len(set(colors))
                        break
                    else:
                        # Revert
                        colors[e] = old_color
                
            if current_score > best_score:
                best_score = current_score
                best_colors = colors[:]
                
        if time.time() > deadline:
            break

    return best_colors
