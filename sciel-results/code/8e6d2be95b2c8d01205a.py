import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_CYCLES = 1820 # C(16,4)

def construct():
    # Set seed for reproducibility if needed, but random is fine for search
    # random.seed(42)
    
    # Precompute edge indices
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    # Map edge index to (u, v)
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[idx] = (i, j)
            idx += 1
            
    # Precompute for each edge, the list of 4-cycles it participates in
    # A 4-cycle is defined by 4 vertices {a, b, c, d}.
    # The edges in the 3 possible cycles on these vertices are:
    # 1. (a,b), (b,c), (c,d), (d,a)
    # 2. (a,b), (b,d), (d,c), (c,a)
    # 3. (a,c), (c,b), (b,d), (d,a)
    # Note: The set of edges is the same for all 3 cycles: {ab, bc, cd, da}, {ab, bd, dc, ca}, {ac, cb, bd, da}
    # Wait, the "3 cycles on every four-vertex subset" refers to the 3 ways to form a 4-cycle using the 6 edges of K4.
    # Let vertices be 0,1,2,3.
    # Cycle 1: 0-1-2-3-0 -> Edges: (0,1), (1,2), (2,3), (3,0)
    # Cycle 2: 0-1-3-2-0 -> Edges: (0,1), (1,3), (3,2), (2,0)
    # Cycle 3: 0-2-1-3-0 -> Edges: (0,2), (2,1), (1,3), (3,0)
    
    # For each edge e, we need to know which 4-cycles (defined by the set of 4 edges) contain it.
    # There are C(16,4) = 1820 sets of 4 vertices.
    # For each set of 4 vertices, there are 3 cycles.
    # Total cycle instances = 1820 * 3 = 5460.
    # Each cycle has 4 edges.
    # Each edge (u,v) is in C(14,2) = 91 sets of 4 vertices.
    # In each such set, the edge (u,v) appears in exactly 2 of the 3 cycles?
    # Let's check. Vertices {u,v,x,y}.
    # C1: u-v-x-y-u. Contains (u,v).
    # C2: u-v-y-x-u. Contains (u,v).
    # C3: u-x-v-y-u. Contains (u,v)? No, edges are (u,x), (x,v), (v,y), (y,u).
    # So each edge is in 2 cycles per 4-vertex set.
    # Total cycle instances per edge = 91 * 2 = 182.
    
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    
    for i in range(N):
        for j in range(i + 1, N):
            # Find all pairs (x, y) such that x, y != i, j and x < y
            # And x, y are distinct from i, j
            others = [v for v in range(N) if v != i and v != j]
            for k in range(len(others)):
                for l in range(k + 1, len(others)):
                    x = others[k]
                    y = others[l]
                    # Vertices: i, j, x, y
                    # Cycle 1: i-j-x-y-i
                    e1 = edge_idx(i, j)
                    e2 = edge_idx(j, x)
                    e3 = edge_idx(x, y)
                    e4 = edge_idx(y, i)
                    # Cycle 2: i-j-y-x-i
                    e5 = edge_idx(j, y)
                    e6 = edge_idx(y, x) # same as e3
                    e7 = edge_idx(x, i) # same as e4? No, (x,i) is (i,x).
                    # Let's be precise.
                    # Cycle 1 edges: (i,j), (j,x), (x,y), (y,i)
                    # Cycle 2 edges: (i,j), (j,y), (y,x), (x,i)
                    # Cycle 3 edges: (i,x), (x,j), (j,y), (y,i)
                    
                    # Add to edge_cycles for each edge in these cycles
                    # Cycle 1
                    for e in [e1, e2, e3, e4]:
                        # Store the other 3 edges to quickly check validity
                        pass # We will store the list of 4 edge indices for the cycle
                    
    # Better structure: cycle_list[cycle_id] = [e1, e2, e3, e4]
    # And edge_cycles[edge_id] = list of cycle_ids
    
    cycle_list = []
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    
    for i in range(N):
        for j in range(i + 1, N):
            others = [v for v in range(N) if v != i and v != j]
            for k in range(len(others)):
                for l in range(k + 1, len(others)):
                    x = others[k]
                    y = others[l]
                    # Cycle 1: i-j-x-y-i
                    c1_edges = [edge_idx(i, j), edge_idx(j, x), edge_idx(x, y), edge_idx(y, i)]
                    cid = len(cycle_list)
                    cycle_list.append(c1_edges)
                    for e in c1_edges:
                        edge_cycles[e].append(cid)
                        
                    # Cycle 2: i-j-y-x-i
                    c2_edges = [edge_idx(i, j), edge_idx(j, y), edge_idx(y, x), edge_idx(x, i)]
                    cid = len(cycle_list)
                    cycle_list.append(c2_edges)
                    for e in c2_edges:
                        edge_cycles[e].append(cid)
                        
                    # Cycle 3: i-x-j-y-i
                    c3_edges = [edge_idx(i, x), edge_idx(x, j), edge_idx(j, y), edge_idx(y, i)]
                    cid = len(cycle_list)
                    cycle_list.append(c3_edges)
                    for e in c3_edges:
                        edge_cycles[e].append(cid)

    # Initialize random colors
    colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
    
    def get_score(colors):
        used = set(colors)
        violations = 0
        for edges in cycle_list:
            c = [colors[e] for e in edges]
            if len(set(c)) == 4:
                violations += 1
        # Score: Maximize distinct colors, Minimize violations
        # We want to maximize distinct colors.
        # If violations > 0, it's invalid.
        # Let's define a fitness function for SA.
        # Primary goal: 0 violations.
        # Secondary goal: Max distinct colors.
        # If we have violations, we should reduce them first.
        # Score = -violations * 1000 + len(used)
        return -violations * 1000 + len(used)

    current_score = get_score(colors)
    best_colors = colors[:]
    best_score = current_score
    
    start_time = time.time()
    timeout = 55 # Leave 5s buffer
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if T < 0.1:
            break
            
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        
        # Pick a new color
        # Bias towards new colors if we have few, or random
        new_color = random.randint(0, 119)
        if new_color == old_color:
            continue
            
        # Calculate change in violations
        # Only cycles containing e_idx are affected
        delta_violations = 0
        for cid in edge_cycles[e_idx]:
            edges = cycle_list[cid]
            # Get colors of the 4 edges
            # Replace e_idx with new_color
            c_old = [colors[e] for e in edges]
            c_new = c_old[:]
            # Find index of e_idx in edges
            # This is inefficient. Let's store indices relative to cycle?
            # Or just check before and after.
            # Since we only change one edge, we can check:
            # old_valid = len(set(c_old)) != 4
            # new_valid = len(set(c_new)) != 4
            
            # To avoid copying lists, we can do:
            old_set = set(c_old)
            old_is_violation = (len(old_set) == 4)
            
            # Construct new set
            # Remove old_color from set, add new_color
            # But careful if old_color appears multiple times in the cycle
            # It's easier to just build the new list for the 4 elements
            # There are only 4 elements.
            new_colors_in_cycle = []
            for e in edges:
                if e == e_idx:
                    new_colors_in_cycle.append(new_color)
                else:
                    new_colors_in_cycle.append(colors[e])
            new_set = set(new_colors_in_cycle)
            new_is_violation = (len(new_set) == 4)
            
            if not old_is_violation and new_is_violation:
                delta_violations += 1
            elif old_is_violation and not new_is_violation:
                delta_violations -= 1
            # else: no change in violation count for this cycle
            
        # Calculate change in distinct colors
        # This is harder to compute incrementally without a global count array.
        # Let's maintain a count of each color.
        
        # Wait, I didn't initialize the count array. Let's do it now.
        # Actually, let's just recompute score for simplicity first? 
        # 5460 cycles * 4 edges = 21k operations per score calc.
        # 100k iterations * 21k = 2.1e9 ops. Too slow in Python.
        
        # We MUST use incremental updates.
        
        # Let's restart with a proper incremental setup.
        
    # RESTART with proper incremental SA
    
    colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
    color_counts = [0] * 120
    for c in colors:
        color_counts[c] += 1
    distinct_colors = sum(1 for c in color_counts if c > 0)
    
    # Precompute violation counts per cycle? No, just total violations.
    # We need to know for each cycle if it is a violation.
    # is_violation[cid] = boolean
    is_violation = [False] * len(cycle_list)
    total_violations = 0
    
    for cid, edges in enumerate(cycle_list):
        c = [colors[e] for e in edges]
        if len(set(c)) == 4:
            is_violation[cid] = True
            total_violations += 1
            
    current_score = -total_violations * 1000 + distinct_colors
    best_colors = colors[:]
    best_score = current_score
    
    start_time = time.time()
    timeout = 55
    
    T = 10.0
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        if T < 0.05:
            break
            
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        
        # Strategy for new color:
        # If we want to increase distinct colors, try a color with count 0.
        # If we want to fix violations, try a color that matches another edge in the cycle.
        # Simple: Random.
        new_color = random.randint(0, 119)
        if new_color == old_color:
            continue
            
        # Calculate delta_violations
        delta_violations = 0
        for cid in edge_cycles[e_idx]:
            edges = cycle_list[cid]
            # Get the other 3 colors
            # Since we only change e_idx, we can check the set of colors in the cycle
            # old set vs new set
            
            # Optimization: Store the set of colors for each cycle?
            # No, sets are expensive to update.
            # Just compute the 4 values.
            
            # Find the index of e_idx in edges?
            # We can just iterate.
            old_c1, old_c2, old_c3, old_c4 = colors[edges[0]], colors[edges[1]], colors[edges[2]], colors[edges[3]]
            
            # Determine which one is e_idx
            # This is a bit slow.
            # Alternative: Store in cycle_list the edge indices.
            # We know e_idx is one of them.
            
            # Let's just build the new list
            new_c1 = new_color if edges[0] == e_idx else colors[edges[0]]
            new_c2 = new_color if edges[1] == e_idx else colors[edges[1]]
            new_c3 = new_color if edges[2] == e_idx else colors[edges[2]]
            new_c4 = new_color if edges[3] == e_idx else colors[edges[3]]
            
            old_set = {old_c1, old_c2, old_c3, old_c4}
            new_set = {new_c1, new_c2, new_c3, new_c4}
            
            old_v = (len(old_set) == 4)
            new_v = (len(new_set) == 4)
            
            if not old_v and new_v:
                delta_violations += 1
            elif old_v and not new_v:
                delta_violations -= 1
                
        # Calculate delta_distinct
        # old_color count decreases by 1
        # new_color count increases by 1
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1 # Lost a color
        if color_counts[new_color] == 0:
            delta_distinct += 1 # Gained a color
            
        delta_score = -delta_violations * 1000 + delta_distinct
        
        if delta_score >= 0 or random.random() < pow(2.718, delta_score / T):
            # Accept
            colors[e_idx] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Update is_violation for affected cycles
            for cid in edge_cycles[e_idx]:
                edges = cycle_list[cid]
                # Recalculate violation status for this cycle
                c = [colors[e] for e in edges]
                is_violation[cid] = (len(set(c)) == 4)
                
            total_violations += delta_violations
            distinct_colors += delta_distinct
            current_score += delta_score
            
            if current_score > best_score:
                best_score = current_score
                best_colors = colors[:]
                
        T *= cooling
        
    # Ensure best_colors is valid (should be, but just in case)
    # If best_colors has violations, we might want to return a valid one with fewer colors?
    # The task says "No four-cycle may have four distinct edge colors".
    # So invalid results score 0.
    # We must return a valid coloring.
    # If best_score implies violations > 0, we need to fix them or find a valid one.
    
    # Check if best_colors is valid
    def is_valid(colors):
        for edges in cycle_list:
            if len(set(colors[e] for e in edges)) == 4:
                return False
        return True
        
    if not is_valid(best_colors):
        # Fallback: Greedy fix or return a known valid sparse solution
        # Let's try to fix best_colors
        # Or just use the 9-color solution from the seed as a safety net?
        # The seed solution is valid.
        # But we want to beat it.
        # If SA found a high score with violations, it's useless.
        # We should prioritize validity.
        # In the SA loop, we penalized violations heavily.
        # So best_score should have 0 violations if possible.
        
        # If we still have violations, let's try a quick greedy repair
        # Or just return the seed solution.
        # Let's check the score of the seed solution.
        # Seed: 9 colors, 0 violations. Score = 9.
        # If best_score < 9, return seed.
        
        # Let's construct the seed solution
        seed_colors = [0] * NUM_EDGES
        def edge_idx_seed(i, j):
            if i > j: i, j = j, i
            return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
            
        for p in range(8):
            u = 2*p
            v = 2*p + 1
            idx = edge_idx_seed(u, v)
            seed_colors[idx] = p + 1
            
        if best_score < 9 or not is_valid(best_colors):
             return seed_colors
             
    return best_colors
