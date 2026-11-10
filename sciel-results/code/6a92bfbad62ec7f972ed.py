import time
import random
import numpy as np
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _get_edge_indices():
    """Return edge index mapping and cycle structures."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))
    
    cycles_np = np.array(cycles, dtype=np.int32)
    
    # For each edge, find which cycles contain it
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    return idx_map, cycles_np, edge_to_cycles

# Precompute
IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _is_valid_vectorized(colors_np):
    """Check if no cycle has 4 distinct colors."""
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _is_edge_change_valid(colors, e, new_color):
    """
    Check if changing colors[e] to new_color preserves validity.
    Only checks cycles containing edge e.
    """
    if colors[e] == new_color:
        return True
        
    for ci in EDGE_TO_CYCLES[e]:
        cycle = CYCLES[ci]
        # Get colors of the other 3 edges
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        
        # Replace the color of edge e with new_color
        if cycle[0] == e:
            vals = (new_color, c1, c2, c3)
        elif cycle[1] == e:
            vals = (c0, new_color, c2, c3)
        elif cycle[2] == e:
            vals = (c0, c1, new_color, c3)
        else:
            vals = (c0, c1, c2, new_color)
            
        # Check if all 4 are distinct
        if vals[0] != vals[1] and vals[0] != vals[2] and vals[0] != vals[3] and \
           vals[1] != vals[2] and vals[1] != vals[3] and \
           vals[2] != vals[3]:
            return False
            
    return True

def _initial_valid_coloring():
    """Generate a valid 2-coloring as starting point."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _greedy_expansion(colors, unused_colors, rng, deadline):
    """
    Try to add unused colors to the coloring.
    Returns (new_colors, new_unused_set, distinct_count)
    """
    current_colors = colors.copy()
    current_unused = set(unused_colors)
    distinct_count = _count_distinct(current_colors)
    
    # Sort unused colors randomly to break symmetry
    unused_list = list(current_unused)
    rng.shuffle(unused_list)
    
    for c in unused_list:
        if time.time() > deadline:
            break
        if c not in current_unused:
            continue
            
        # Try to find an edge where we can place color c
        # Shuffle edges to avoid bias
        edges = list(range(NEDGES))
        rng.shuffle(edges)
        
        placed = False
        for e in edges:
            if time.time() > deadline:
                break
            # Check if placing c on e is valid
            if _is_edge_change_valid(current_colors, e, c):
                current_colors[e] = c
                current_unused.remove(c)
                distinct_count += 1
                placed = True
                break
                
        if not placed:
            # If we couldn't place this color, we might need to rearrange existing colors
            # to make room. We'll handle this in the local search phase.
            # For now, just skip this color in this pass.
            pass
            
    return current_colors, current_unused, distinct_count

def _local_search_rearrange(colors, unused_colors, rng, deadline):
    """
    Perform local moves to rearrange existing colors to potentially allow new colors.
    This is a "shuffling" phase.
    """
    current_colors = colors.copy()
    used_colors = set(current_colors)
    
    # We want to move edges around.
    # Strategy: Pick two edges e1, e2 with different colors. Swap them if valid.
    # Or: Pick an edge e, change its color to another used color if valid.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        e1 = rng.randint(0, NEDGES - 1)
        e2 = rng.randint(0, NEDGES - 1)
        
        if e1 == e2:
            continue
            
        c1 = current_colors[e1]
        c2 = current_colors[e2]
        
        if c1 == c2:
            continue
            
        # Try swapping
        if _is_edge_change_valid(current_colors, e1, c2) and _is_edge_change_valid(current_colors, e2, c1):
            current_colors[e1] = c2
            current_colors[e2] = c1
            
        # Or try changing e1 to c2 (if e2 stays c1, but e1 was c1... wait, if c1 != c2, 
        # changing e1 to c2 is a "move" not a swap)
        # Actually, just random valid recoloring to a used color is enough to perturb.
        else:
            # Try recoloring e1 to a random used color
            new_c = rng.choice(list(used_colors))
            if new_c != c1 and _is_edge_change_valid(current_colors, e1, new_c):
                current_colors[e1] = new_c
                # Note: used_colors set doesn't change, but distribution does.
                
    return current_colors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    # Start with a valid 2-coloring
    current_colors = _initial_valid_coloring()
    used_colors = set(current_colors)
    unused_colors = set(range(MAX_COLOR)) - used_colors
    
    # Phase 1: Greedy Expansion
    # Try to add as many colors as possible greedily
    current_colors, unused_colors, distinct = _greedy_expansion(current_colors, unused_colors, rng, deadline - 5.0)
    best_overall_colors = current_colors.copy()
    best_overall_distinct = distinct
    
    # Phase 2: Local Search to break plateaus
    # If we are stuck, rearrange existing colors and try greedy again
    iterations = 0
    while time.time() < deadline - 2.0:
        iterations += 1
        
        # Rearrange to change the structure
        current_colors = _local_search_rearrange(current_colors, unused_colors, rng, deadline - 2.0)
        
        # Try greedy expansion again
        new_colors, new_unused, new_distinct = _greedy_expansion(current_colors, unused_colors, rng, deadline - 2.0)
        
        if new_distinct > best_overall_distinct:
            best_overall_distinct = new_distinct
            best_overall_colors = new_colors.copy()
            unused_colors = new_unused
            current_colors = new_colors
        else:
            # If no improvement, keep the rearranged state for next iteration?
            # Or revert? Let's keep the rearranged state as it might be a better starting point.
            current_colors = new_colors
            unused_colors = new_unused
            
        # If we reached max colors, break
        if best_overall_distinct == MAX_COLOR:
            break
            
    # Phase 3: Final Polish
    # Try to add any remaining unused colors one by one with more extensive search
    if best_overall_colors is not None and time.time() < deadline - 1.0:
        final_colors = best_overall_colors.copy()
        final_used = set(final_colors)
        final_unused = list(set(range(MAX_COLOR)) - final_used)
        rng.shuffle(final_unused)
        
        for c in final_unused:
            if time.time() > deadline - 0.5:
                break
            # Try all edges
            for e in range(NEDGES):
                if _is_edge_change_valid(final_colors, e, c):
                    final_colors[e] = c
                    final_used.add(c)
                    break
                    
        if len(final_used) > best_overall_distinct:
            best_overall_distinct = len(final_used)
            best_overall_colors = final_colors

    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()

# Run a quick self-test to ensure validity and measure time
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Distinct colors: {len(set(res))}")
    # Verify
    colors_np = np.array(res)
    print(f"Valid: {_is_valid_vectorized(colors_np)}")
