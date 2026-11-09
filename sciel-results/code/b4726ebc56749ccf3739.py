import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute all 4-cycles (three per 4-vertex subset)
def _precompute_cycles():
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on these 4 vertices
        cycles.append([(a, b), (b, c), (c, d), (d, a)])
        cycles.append([(a, b), (b, d), (d, c), (c, a)])
        cycles.append([(a, c), (c, b), (b, d), (d, a)])
    return cycles

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _is_rainbow(cycle, colors):
    """Check if a 4-cycle has 4 distinct edge colors."""
    c0 = colors[_edge_idx(cycle[0][0], cycle[0][1])]
    c1 = colors[_edge_idx(cycle[1][0], cycle[1][1])]
    c2 = colors[_edge_idx(cycle[2][0], cycle[2][1])]
    c3 = colors[_edge_idx(cycle[3][0], cycle[3][1])]
    return len({c0, c1, c2, c3}) == 4

def _count_violations(colors, cycles):
    count = 0
    for cycle in cycles:
        if _is_rainbow(cycle, colors):
            count += 1
    return count

def _count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline with 15s margin
    
    # Precompute
    cycles = _precompute_cycles()
    
    # Initial coloring: min(i,j) gives 15 colors, guaranteed valid
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[_edge_idx(i, j)] = i
    
    best_colors = colors[:]
    best_score = _count_distinct_colors(best_colors)  # 15
    
    # Phase 1: Greedy color addition
    # Try to add new colors by recoloring edges
    used_colors = set(colors)
    
    # Try adding colors one by one
    for target_color in range(15, 119):
        if time.time() > deadline - 5:
            break
        
        # Try to find an edge to recolor to target_color
        # Prefer edges that are part of fewer cycles
        edges_tried = list(range(NUM_EDGES))
        random.shuffle(edges_tried)
        
        found = False
        for e in edges_tried:
            if time.time() > deadline - 5:
                break
            old_color = colors[e]
            if old_color == target_color:
                continue
            
            colors[e] = target_color
            if _count_violations(colors, cycles) == 0:
                found = True
                break
            colors[e] = old_color
        
        if not found:
            # Try pairs of edges
            pairs_tried = list(itertools.combinations(range(NUM_EDGES), 2))
            random.shuffle(pairs_tried)
            for e1, e2 in pairs_tried[:200]:  # Limit pairs
                if time.time() > deadline - 5:
                    break
                c1 = colors[e1]
                c2 = colors[e2]
                if c1 == target_color or c2 == target_color:
                    continue
                colors[e1] = target_color
                colors[e2] = target_color
                if _count_violations(colors, cycles) == 0:
                    found = True
                    break
                colors[e1] = c1
                colors[e2] = c2
            
            if not found:
                # Try single edge with a different existing color
                for e in edges_tried[:100]:
                    if time.time() > deadline - 5:
                        break
                    old_color = colors[e]
                    # Try recoloring to a new color
                    colors[e] = target_color
                    if _count_violations(colors, cycles) == 0:
                        found = True
                        break
                    colors[e] = old_color
                
                if not found:
                    continue  # Can't add this color yet
        
        # Update best
        current_distinct = _count_distinct_colors(colors)
        if current_distinct > best_score:
            best_score = current_distinct
            best_colors = colors[:]
    
    # Phase 2: Local search to add more colors
    # Try random recoloring moves that don't create violations
    for iteration in range(20000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random edge and a random color
        e = random.randint(0, NUM_EDGES - 1)
        new_color = random.randint(0, 119)
        
        if new_color == colors[e]:
            continue
        
        old_color = colors[e]
        colors[e] = new_color
        
        if _count_violations(colors, cycles) == 0:
            current_distinct = _count_distinct_colors(colors)
            if current_distinct > best_score:
                best_score = current_distinct
                best_colors = colors[:]
        else:
            colors[e] = old_color
    
    # Phase 3: Try to add specific new colors with more targeted search
    for target_color in range(15, 119):
        if time.time() > deadline - 5:
            break
        if target_color in set(best_colors):
            continue
        
        # Start from best_colors
        colors = best_colors[:]
        
        # Try single edge changes
        edges_tried = list(range(NUM_EDGES))
        random.shuffle(edges_tried)
        
        for e in edges_tried:
            if time.time() > deadline - 5:
                break
            old_color = colors[e]
            colors[e] = target_color
            if _count_violations(colors, cycles) == 0:
                current_distinct = _count_distinct_colors(colors)
                if current_distinct > best_score:
                    best_score = current_distinct
                    best_colors = colors[:]
                break
            colors[e] = old_color
        
        # Try pair changes
        if time.time() > deadline - 5:
            break
        pairs = list(itertools.combinations(range(NUM_EDGES), 2))
        random.shuffle(pairs)
        for e1, e2 in pairs[:500]:
            if time.time() > deadline - 5:
                break
            c1 = colors[e1]
            c2 = colors[e2]
            colors[e1] = target_color
            colors[e2] = target_color
            if _count_violations(colors, cycles) == 0:
                current_distinct = _count_distinct_colors(colors)
                if current_distinct > best_score:
                    best_score = current_distinct
                    best_colors = colors[:]
                break
            colors[e1] = c1
            colors[e2] = c2
    
    return best_colors

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
