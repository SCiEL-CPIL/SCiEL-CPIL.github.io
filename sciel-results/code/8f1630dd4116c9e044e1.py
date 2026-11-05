import time
import random
import sys

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

def is_valid_assignment(e, c, colors):
    """Check if assigning color c to edge e is valid given current colors."""
    for cyc in _CYCLES_BY_EDGE[e]:
        # Get the other 3 edges in the cycle
        # Since e is in cyc, we can find the others
        # cyc is a tuple of 4 edges
        if cyc[0] == e:
            others = (cyc[1], cyc[2], cyc[3])
        elif cyc[1] == e:
            others = (cyc[0], cyc[2], cyc[3])
        elif cyc[2] == e:
            others = (cyc[0], cyc[1], cyc[3])
        else:
            others = (cyc[0], cyc[1], cyc[2])
        
        c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
        
        # If the other 3 colors are all distinct, c must match one of them
        if c1 != c2 and c1 != c3 and c2 != c3:
            if c != c1 and c != c2 and c != c3:
                return False
    return True

def count_colors(colors):
    return len(set(colors))

def greedy_fill(rng):
    """
    Greedily assign new colors to edges to maximize distinct colors.
    """
    colors = [0] * NUM_EDGES
    used_colors = {0}
    available_new = list(range(1, MAX_COLOR + 1))
    rng.shuffle(available_new)
    
    edge_order = list(range(NUM_EDGES))
    rng.shuffle(edge_order)
    
    # We want to try to assign as many new colors as possible
    # Iterate through edges, for each try to find a valid new color
    for e in edge_order:
        # Try colors from available_new
        # We iterate through the list, but we need to remove used ones
        # To avoid O(N^2) removal, we can just check if it's in used_colors?
        # No, available_new contains only unused colors.
        # But if we use one, we should not try it again for subsequent edges?
        # Yes, we want distinct colors.
        
        assigned = False
        for i in range(len(available_new)):
            c = available_new[i]
            if is_valid_assignment(e, c, colors):
                colors[e] = c
                used_colors.add(c)
                # Remove from available
                available_new[i] = available_new[-1]
                available_new.pop()
                assigned = True
                break
        
        # If no new color works, leave as 0 (or current)
        # We don't try to change to an existing color here, that's for local search
        
    return colors, len(used_colors)

def local_search(colors, rng, deadline):
    """
    Try to increase the number of distinct colors by local search.
    """
    colors = colors[:]
    used_colors = set(colors)
    unavailable = set(range(MAX_COLOR + 1)) - used_colors
    
    best_count = len(used_colors)
    
    while time.time() < deadline:
        if not unavailable:
            break
            
        # Try to introduce a new color
        improved = False
        # Try a few random moves
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            
            # Pick a random new color
            c = rng.choice(list(unavailable))
            
            if is_valid_assignment(e, c, colors):
                colors[e] = c
                used_colors.add(c)
                unavailable.discard(c)
                best_count = len(used_colors)
                improved = True
                break
        
        if not improved:
            # Try to do a neutral move to escape local minimum
            # Change an edge to a different existing color
            # This might allow a neighbor to take a new color later
            for _ in range(100):
                if time.time() >= deadline:
                    break
                
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = colors[e]
                
                # Pick a random existing color different from old_c
                # If only 1 color, skip
                if len(used_colors) < 2:
                    break
                    
                new_c = rng.choice(list(used_colors))
                if new_c == old_c:
                    continue
                
                if is_valid_assignment(e, new_c, colors):
                    colors[e] = new_c
                    # used_colors set doesn't change, but distribution does
                    # This is a valid move
                    break
    
    return colors, best_count

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Phase 1: Greedy Fills
    # Run multiple greedy fills to find a good starting point
    for i in range(50):
        if time.time() > deadline - 10:
            break
        c, cnt = greedy_fill(rng)
        if cnt > best_count:
            best_count = cnt
            best_colors = c[:]
    
    # Phase 2: Local Search on the best candidate
    if best_count < MAX_COLOR + 1:
        improved_c, improved_cnt = local_search(best_colors, rng, deadline - 2)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c
    
    return best_colors
