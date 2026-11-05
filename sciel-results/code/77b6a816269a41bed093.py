"""Program for the edge-coloring task: maximize distinct colors in K16 with no rainbow 4-cycle."""
import time
import random
import itertools

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + j - i - 1

def _precompute_cycles():
    """Precompute all 4-cycles as lists of 4 edge indices."""
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Three 4-cycles on {a,b,c,d}
        # Cycle 1: a-b, b-c, c-d, d-a
        cycles.append([_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)])
        # Cycle 2: a-b, b-d, d-c, c-a
        cycles.append([_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(c,a)])
        # Cycle 3: a-c, c-b, b-d, d-a
        cycles.append([_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(d,a)])
    return cycles

def _precompute_edge_cycles():
    """For each edge, precompute the list of cycles it participates in."""
    all_cycles = _precompute_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return all_cycles, edge_to_cycles

def _check_validity(colors, all_cycles):
    """Check that no 4-cycle has four distinct colors."""
    for cyc in all_cycles:
        if len(set(colors[e] for e in cyc)) == 4:
            return False
    return True

def _is_valid_change(colors, edge_idx, new_color, all_cycles, edge_to_cycles):
    """Check if changing edge_idx to new_color keeps all cycles valid."""
    for ci in edge_to_cycles[edge_idx]:
        cyc = all_cycles[ci]
        # Get the other 3 edges in this cycle
        other_edges = [e for e in cyc if e != edge_idx]
        other_colors = set(colors[e] for e in other_edges)
        # The cycle would have colors: {new_color} | other_colors
        if len(other_colors) == 3 and new_color not in other_colors:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for safety
    
    random.seed(42)
    
    all_cycles, edge_to_cycles = _precompute_edge_cycles()
    
    # Initial structured 3-coloring
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            if i < 8 and j < 8:
                colors[_edge_index(i, j)] = 0
            elif i >= 8 and j >= 8:
                colors[_edge_index(i, j)] = 1
            else:
                colors[_edge_index(i, j)] = 2
    
    if _check_validity(colors, all_cycles):
        best_colors = colors[:]
        best_count = _count_colors(colors)
    else:
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    # Local search: try to change edges to new colors while maintaining validity
    edges = list(range(NUM_EDGES))
    
    # Multiple passes with different strategies
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(edges)
        progress = False
        
        for e in edges:
            if time.time() > deadline:
                break
            
            old_color = colors[e]
            used_colors = set(colors)
            
            # Prefer unused colors
            unused = [c for c in range(NUM_COLORS) if c not in used_colors]
            
            if unused:
                # Try unused colors in random order
                random.shuffle(unused)
                for new_color in unused:
                    if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
                        colors[e] = new_color
                        progress = True
                        break
            else:
                # All colors used; try to rearrange
                candidates = [c for c in range(NUM_COLORS) if c != old_color]
                random.shuffle(candidates)
                for new_color in candidates:
                    if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
                        colors[e] = new_color
                        progress = True
                        break
        
        cur_count = _count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if not progress and iteration > 5:
            # Try a restart with perturbation
            if time.time() > deadline - 10:
                break
            # Perturb: randomly change some edges back to base colors
            for _ in range(10):
                e = random.choice(edges)
                colors[e] = random.choice([0, 1, 2])
    
    # Final systematic pass: try to add each unused color
    if time.time() < deadline - 5:
        used = set(best_colors)
        for new_c in range(NUM_COLORS):
            if new_c in used:
                continue
            if time.time() > deadline:
                break
            
            # Try to add new_c to one edge at a time
            for e in range(NUM_EDGES):
                if time.time() > deadline:
                    break
                old = best_colors[e]
                if _is_valid_change(best_colors, e, new_c, all_cycles, edge_to_cycles):
                    best_colors[e] = new_c
                    best_count = _count_colors(best_colors)
                    used.add(new_c)
                    break
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors
