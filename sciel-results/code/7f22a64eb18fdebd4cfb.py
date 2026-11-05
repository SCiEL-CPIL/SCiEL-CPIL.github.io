import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles. For each set of 4 vertices {a,b,c,d},
    there are 3 distinct 4-cycles (the 3 ways to pair up the 4 vertices
    into 2 pairs, then the cycle uses all 4 edges).
    
    Actually, a 4-cycle on vertices {a,b,c,d} is a cycle of length 4 using
    4 edges. The 3 distinct 4-cycles (up to reversal) are:
    - a-b-c-d-a (edges: ab, bc, cd, da)
    - a-b-d-c-a (edges: ab, bd, dc, ca)
    - a-c-b-d-a (edges: ac, cb, bd, da)
    """
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = [edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)]
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = [edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)]
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = [edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)]
                    cycles.append(e3)
    return cycles

def count_distinct_colors(colors, cycles):
    """Count how many colors are actually used."""
    return len(set(colors))

def is_valid_change(colors, edge_idx, new_color, cycles_by_edge):
    """Check if changing edge_idx to new_color maintains validity."""
    if colors[edge_idx] == new_color:
        return True
    old_color = colors[edge_idx]
    
    # Check all cycles containing this edge
    for cycle in cycles_by_edge[edge_idx]:
        # Get the 4 edge colors in this cycle
        colors_in_cycle = [colors[e] for e in cycle]
        # Replace the old color with new color
        idx_in_cycle = colors_in_cycle.index(old_color)
        colors_in_cycle[idx_in_cycle] = new_color
        # Check if 4 distinct colors
        if len(set(colors_in_cycle)) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Precompute cycles and index by edge
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    
    # Local search: try to recolor edges to new colors
    best_colors = colors[:]
    best_count = 1
    
    # Try multiple random restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Start from current best or random
        if rng.random() < 0.3:
            colors = best_colors[:]
        else:
            colors = [0] * NUM_EDGES
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try random edge recolorings
            for _ in range(1000):
                if time.time() > deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                # Try a random new color
                new_color = rng.randint(0, NUM_EDGES - 1)
                if new_color == colors[e]:
                    continue
                if is_valid_change(colors, e, new_color, cycles_by_edge):
                    colors[e] = new_color
                    improved = True
        
        # Update best
        cur_count = count_distinct_colors(colors, cycles)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    return best_colors
