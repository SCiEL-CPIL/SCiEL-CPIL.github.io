import random
import time
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    # A 4-cycle is defined by 4 vertices.
    # There are 3 cycles in each K4.
    # We store the list of 4 edge indices for each cycle.
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        # Edges in the K4
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 2: v1-v2-v4-v3-v1
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 3: v1-v3-v2-v4-v1
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    # Helper to check if a cycle is valid given a specific edge color override
    def is_cycle_valid(cycle_edges, edge_idx, new_color, colors):
        col_set = set()
        for ed in cycle_edges:
            if ed == edge_idx:
                col_set.add(new_color)
            else:
                col_set.add(colors[ed])
        return len(col_set) < 4

    # Initialize colors with a recursive split structure to get a high starting point
    # This ensures we start with at least N-1 colors and a valid configuration.
    colors = [0] * NUM_EDGES
    used_colors = set()
    
    def init_split(vertices, color_offset):
        if len(vertices) <= 1:
            return color_offset
        # Split into two halves
        mid = len(vertices) // 2
        v1 = vertices[:mid]
        v2 = vertices[mid:]
        
        # Recursively color subgraphs
        next_offset = init_split(v1, color_offset)
        next_offset = init_split(v2, next_offset)
        
        # Color edges between v1 and v2 with a new color
        new_color = next_offset
        used_colors.add(new_color)
        for u in v1:
            for v in v2:
                idx = edge_to_idx[(min(u,v), max(u,v))]
                colors[idx] = new_color
        return next_offset + 1

    init_split(list(range(N)), 0)
    
    best_num_colors = len(used_colors)
    best_colors = colors[:]

    # Local Search
    # We try to introduce new colors.
    # We also try to swap colors to potentially free up constraints.
    
    # To speed up, we can prioritize edges that are in fewer "tight" cycles?
    # Or just random.
    
    # Optimization: Precompute the set of cycles for each edge for faster access.
    # Already done in edge_to_cycles.
    
    # Random restarts if stuck?
    # For now, single run with random restarts of the search process (not the init) if time permits.
    
    current_colors = best_colors[:]
    current_used = set(best_colors)
    
    # Shuffle edges for random order
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    while time.time() < deadline:
        # Try to add a new color
        # Pick a random edge
        e = random.choice(edge_order)
        
        # Pick a random unused color
        if len(current_used) < 120:
            unused = [c for c in range(120) if c not in current_used]
            if unused:
                c = random.choice(unused)
            else:
                c = random.randint(0, 119)
        else:
            c = random.randint(0, 119)
            
        if c == current_colors[e]:
            continue
            
        # Check validity
        valid = True
        for c_idx in edge_to_cycles[e]:
            cycle_edges = cycles[c_idx]
            if not is_cycle_valid(cycle_edges, e, c, current_colors):
                valid = False
                break
                
        if valid:
            current_colors[e] = c
            current_used.add(c)
            if len(current_used) > best_num_colors:
                best_num_colors = len(current_used)
                best_colors = current_colors[:]
        
        # Occasionally try a swap to break stagnation?
        # If we haven't improved in a long time, we might try to swap two edges' colors.
        # But swapping doesn't change the number of colors, just the configuration.
        # It might allow a new color to be added later.
        
        # Simple move: try to change an edge to an existing color to see if it opens up space?
        # No, we want to MAXIMIZE colors. Changing to an existing color reduces or keeps same.
        # So we should only try to add new colors or swap.
        
        # Let's try a swap move every 100 steps.
        # Pick two edges, swap their colors.
        # This is valid if all cycles involving either edge remain valid.
        # This is expensive to check.
        
        # Alternative: Just keep trying to add new colors.
        # If stuck, restart with a different random seed?
        # The init is deterministic. The search is random.
        
        # To diversify, we can occasionally reset to best_colors and try a different random path?
        # Or just keep going.
        
        # If we are stuck (no improvement in many steps), we might try to "shake" the configuration.
        # But we don't track steps explicitly.
        
        # Let's add a mechanism to try to swap colors of two edges if we are stuck.
        # We'll track a "stagnation" counter.
        
        # For simplicity, let's just run the add-new-color loop.
        # If we want to be more aggressive, we can try to find an edge that CAN take a new color
        # by checking all unused colors for a random edge.
        
        # Improved step:
        # Pick an edge.
        # Find all unused colors that are valid for this edge.
        # If any, pick one.
        # This is more efficient than random guessing if the space is tight.
        
        # Let's implement this "smart" pick.
        
        # But checking all 120 colors for each edge is 120 * (num cycles) ops.
        # Num cycles per edge is ~ O(N^2) = 256?
        # 120 * 256 = 30k ops per edge.
        # With 120 edges, 3.6M ops per full pass.
        # In 55s, we can do many passes.
        
        # Let's try this smart pick.
        
        # We'll alternate between random edge + random unused color
        # and random edge + best unused color.
        
        # Actually, the current loop is fine. Let's just run it.
        
        # One optimization: if we have many unused colors, random pick is fine.
        # If few, we should check all.
        
        # Let's stick to the current logic but ensure we don't waste time.
        
        # If time is running out, return best.
        
    return best_colors

# Run construct to test? No, just return the function.
# The evaluator will call construct().
