"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Returns a list `cycles_for_edge` of length NUM_EDGES.
    cycles_for_edge[e] is a list of tuples (e1, e2, e3) where 
    {e, e1, e2, e3} is a 4-cycle.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    def get_idx(u, v):
                        if u > v: u, v = v, u
                        return edge_to_idx[(u, v)]
                    
                    # Cycle 1: a-b-c-d-a
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, c)
                    e3 = get_idx(c, d)
                    e4 = get_idx(d, a)
                    cyc1 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc1):
                        others = [cyc1[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, d)
                    e3 = get_idx(d, c)
                    e4 = get_idx(c, a)
                    cyc2 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc2):
                        others = [cyc2[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = get_idx(a, c)
                    e2 = get_idx(c, b)
                    e3 = get_idx(b, d)
                    e4 = get_idx(d, a)
                    cyc3 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc3):
                        others = [cyc3[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                        
    return cycles_for_edge

CYCLES_FOR_EDGE = _precompute_cycles()

def _is_valid_coloring(colors):
    """Check if the coloring is valid (no rainbow 4-cycles)."""
    for e in range(NUM_EDGES):
        for others in CYCLES_FOR_EDGE[e]:
            c0 = colors[e]
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            if len({c0, c1, c2, c3}) == 4:
                return False
    return True

def _can_assign_new_color(e, colors):
    """
    Check if edge e can be assigned a new unique color.
    Condition: For all 4-cycles containing e, the other 3 edges do not have 3 distinct colors.
    """
    for others in CYCLES_FOR_EDGE[e]:
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        if len({c1, c2, c3}) == 3:
            return False
    return True

def _local_search(colors, max_iterations=5000, seed=0):
    """
    Perform local search to increase the number of colors.
    """
    random.seed(seed)
    colors = colors[:]
    current_count = len(set(colors))
    max_color = max(colors) if colors else 0
    
    for _ in range(max_iterations):
        e = random.randint(0, NUM_EDGES - 1)
        c_e = colors[e]
        
        # Try to assign a new color
        if _can_assign_new_color(e, colors):
            colors[e] = max_color + 1
            max_color += 1
            current_count += 1
            continue
            
        # Try to swap with a random other edge
        e2 = random.randint(0, NUM_EDGES - 1)
        if e2 == e:
            continue
        c_e2 = colors[e2]
        if c_e == c_e2:
            continue
            
        # Tentatively swap
        colors[e] = c_e2
        colors[e2] = c_e
        
        # Check validity
        # We only need to check cycles involving e and e2
        valid = True
        for others in CYCLES_FOR_EDGE[e]:
            c0 = colors[e]
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            if len({c0, c1, c2, c3}) == 4:
                valid = False
                break
        if valid:
            for others in CYCLES_FOR_EDGE[e2]:
                c0 = colors[e2]
                c1 = colors[others[0]]
                c2 = colors[others[1]]
                c3 = colors[others[2]]
                if len({c0, c1, c2, c3}) == 4:
                    valid = False
                    break
                    
        if valid:
            # Accept swap if it doesn't reduce count (it won't) or if we want to explore
            # Swapping doesn't change count, but might enable future new colors
            pass
        else:
            # Revert
            colors[e] = c_e
            colors[e2] = c_e2
            
        # Try to change to an existing color
        # Pick a random existing color
        existing_colors = list(set(colors))
        if existing_colors:
            new_c = random.choice(existing_colors)
            if new_c != c_e:
                colors[e] = new_c
                valid = True
                for others in CYCLES_FOR_EDGE[e]:
                    c0 = colors[e]
                    c1 = colors[others[0]]
                    c2 = colors[others[1]]
                    c3 = colors[others[2]]
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    # Accept
                    pass
                else:
                    colors[e] = c_e

    return colors, len(set(colors))

def _greedy_max_colors(seed):
    random.seed(seed)
    colors = [0] * NUM_EDGES
    next_color = 1
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e in edge_order:
        if _can_assign_new_color(e, colors):
            colors[e] = next_color
            next_color += 1
        else:
            # Pick a random existing color to break symmetry
            existing = [c for c in set(colors) if c != 0] # Avoid 0 if possible? No, 0 is fine.
            if existing:
                colors[e] = random.choice(existing)
            else:
                colors[e] = 0

    return colors

def construct():
    start_time = time.time()
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Run multiple seeds
    for seed in range(200):
        if time.time() - start_time > 55:
            break
        
        # Phase 1: Greedy
        colors = _greedy_max_colors(seed)
        count = len(set(colors))
        
        # Phase 2: Local Search
        colors_ls, count_ls = _local_search(colors, max_iterations=2000, seed=seed)
        
        if count_ls > best_count:
            best_count = count_ls
            best_colors = colors_ls[:]
            
    return best_colors

# EVOLVE-BLOCK-END
