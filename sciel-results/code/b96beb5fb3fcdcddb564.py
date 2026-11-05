import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def greedy_coloring(edge_order, rng):
    """
    Greedy: process edges in given order, assign each edge a color that
    keeps all 4-cycles valid. Try to maximize distinct colors by preferring
    unused colors when possible.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        # Find all 4-cycles that contain edge e and are now "complete"
        # (all 4 edges have been assigned)
        candidate_colors = []
        
        # Try colors in an order that favors diversity
        # First try unused colors (to increase distinct count)
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        # Then try used colors
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        all_candidates = unused + used_list
        
        found = False
        for c in all_candidates:
            colors[e] = c
            # Check all cycles containing e
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                # Only check if all 4 edges in the cycle are assigned
                if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                    continue
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            if valid:
                found = True
                break
        
        if not found:
            # Shouldn't happen, but fallback to 0
            colors[e] = 0
        else:
            used_colors.add(colors[e])
    
    return colors

def local_improve_aggressive(colors, rng, deadline):
    """
    More aggressive local search:
    1. Try to introduce new colors by recoloring edges
    2. Try 2-edge swaps to free up color slots
    3. Use constraint propagation to guide which edges to try
    """
    colors = colors[:]
    best_count = count_colors(colors)
    current_count = best_count
    used_colors = set(colors)
    
    # Count how many times each color is used
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to introduce new colors by recoloring single edges
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            
            # Try unused colors first
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                rng.shuffle(unused)
                candidates = unused[:15]
            else:
                break
            
            for new_c in candidates:
                if new_c == old:
                    continue
                colors[e] = new_c
                # Check only cycles containing edge e
                valid = True
                for cyc in _CYCLES_BY_EDGE[e]:
                    if not is_valid_cycle(colors, cyc):
                        valid = False
                        break
                if valid:
                    used_colors.add(new_c)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                    improved = True
                    break
                else:
                    colors[e] = old
        
        # Phase 2: Try 2-edge swaps to enable new colors
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            # Pick two edges that share a cycle
            e1 = rng.randint(0, NUM_EDGES - 1)
            # Find edges that share a cycle with e1
            shared_edges = set()
            for cyc in _CYCLES_BY_EDGE[e1]:
                for other_e in cyc:
                    if other_e != e1:
                        shared_edges.add(other_e)
            
            if not shared_edges:
                continue
            
            e2 = rng.choice(list(shared_edges))
            
            old1, old2 = colors[e1], colors[e2]
            
            # Try swapping or recoloring
            # Option 1: Give e1 a new unused color, keep e2
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                rng.shuffle(unused)
                for new_c in unused[:10]:
                    colors[e1] = new_c
                    valid = True
                    for cyc in _CYCLES_BY_EDGE[e1]:
                        if not is_valid_cycle(colors, cyc):
                            valid = False
                            break
                    if valid:
                        # Check if this allows e2 to also get a new color
                        unused2 = [c for c in range(MAX_COLOR + 1) if c not in used_colors and c != new_c]
                        if unused2:
                            rng.shuffle(unused2)
                            for new_c2 in unused2[:5]:
                                colors[e2] = new_c2
                                valid2 = True
                                for cyc in _CYCLES_BY_EDGE[e2]:
                                    if not is_valid_cycle(colors, cyc):
                                        valid2 = False
                                        break
                                if valid2:
                                    used_colors.add(new_c)
                                    used_colors.add(new_c2)
                                    current_count += 2
                                    if current_count > best_count:
                                        best_count = current_count
                                    improved = True
                                    break
                                else:
                                    colors[e2] = old2
                            else:
                                # Only e1 gets new color
                                used_colors.add(new_c)
                                current_count += 1
                                if current_count > best_count:
                                    best_count = current_count
                                improved = True
                                break
                        else:
                            used_colors.add(new_c)
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                            improved = True
                            break
                    else:
                        colors[e1] = old1
                if improved:
                    break
            else:
                break
        
        # Phase 3: Try to consolidate colors to free up more unused colors
        # (This helps when we're stuck with many used colors)
        if current_count >= 100:
            for _ in range(50):
                if time.time() >= deadline:
                    break
                
                # Pick a color that's used many times
                if not color_counts:
                    break
                target_color = max(color_counts, key=lambda c: color_counts[c])
                
                # Find edges with this color
                edges_with_color = [e for e in range(NUM_EDGES) if colors[e] == target_color]
                
                if len(edges_with_color) <= 2:
                    continue
                
                # Try to recolor one edge to another existing color
                e = rng.choice(edges_with_color)
                other_colors = [c for c in used_colors if c != target_color]
                rng.shuffle(other_colors)
                
                for new_c in other_colors[:10]:
                    colors[e] = new_c
                    valid = True
                    for cyc in _CYCLES_BY_EDGE[e]:
                        if not is_valid_cycle(colors, cyc):
                            valid = False
                            break
                    if valid:
                        # Update color counts
                        color_counts[target_color] -= 1
                        if color_counts[target_color] == 0:
                            del color_counts[target_color]
                        color_counts[new_c] = color_counts.get(new_c, 0) + 1
                        improved = True
                        break
                    else:
                        colors[e] = target_color
    
    # Rebuild color_counts for consistency
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    used_colors = set(colors)
    current_count = len(used_colors)
    
    return colors, max(best_count, current_count)

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Try multiple greedy orderings
    edge_indices = list(range(NUM_EDGES))
    
    # Different orderings to try
    orderings = []
    
    # Natural order
    orderings.append(list(edge_indices))
    
    # Reverse order
    orderings.append(list(reversed(edge_indices)))
    
    # Random shuffles
    for _ in range(10):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Process edges in order of increasing degree in the 4-cycle hypergraph
    edge_cycle_count = [len(_CYCLES_BY_EDGE[e]) for e in range(NUM_EDGES)]
    degree_order = sorted(edge_indices, key=lambda e: edge_cycle_count[e])
    orderings.append(degree_order)
    orderings.append(list(reversed(degree_order)))
    
    # Try orderings that group edges by vertex pairs
    # Edges sharing a vertex are more constrained
    vertex_edges = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            e = edge_index(i, j)
            vertex_edges[i].append(e)
            vertex_edges[j].append(e)
    
    # Order by vertex: process all edges incident to vertex 0, then 1, etc.
    vertex_order = []
    for v in range(N):
        vertex_order.extend(vertex_edges[v])
    # Remove duplicates while preserving order
    seen = set()
    unique_vertex_order = []
    for e in vertex_order:
        if e not in seen:
            seen.add(e)
            unique_vertex_order.append(e)
    orderings.append(unique_vertex_order)
    
    # Reverse vertex order
    reverse_vertex_order = []
    for v in range(N - 1, -1, -1):
        reverse_vertex_order.extend(vertex_edges[v])
    seen = set()
    unique_rev_vertex_order = []
    for e in reverse_vertex_order:
        if e not in seen:
            seen.add(e)
            unique_rev_vertex_order.append(e)
    orderings.append(unique_rev_vertex_order)
    
    candidates = []
    for order in orderings:
        if time.time() >= deadline - 15:
            break
        c = greedy_coloring(order, rng)
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
    
    # Sort by count descending
    candidates.sort(key=lambda x: -x[0])
    
    if candidates:
        best_count = candidates[0][0]
        best_colors = candidates[0][1][:]
    
    # Local improvement on top candidates - use aggressive version
    top_candidates = candidates[:8]
    for cnt, c in top_candidates:
        if time.time() >= deadline - 5:
            break
        improved_c, improved_cnt = local_improve_aggressive(c, rng, deadline)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c[:]
    
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
