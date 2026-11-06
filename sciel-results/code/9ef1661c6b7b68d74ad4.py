import time
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (3 per 4-vertex subset)
CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    # Three distinct 4-cycles on {a,b,c,d}
    CYCLES.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
    CYCLES.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
    CYCLES.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))

NUM_CYCLES = len(CYCLES)

def count_distinct_colors(colors):
    return len(set(colors))

def count_violations(colors):
    """Count 4-cycles with 4 distinct edge colors."""
    v = 0
    for cycle in CYCLES:
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            v += 1
    return v

def is_valid(colors):
    return count_violations(colors) == 0

def greedy_maximize_colors(colors, deadline):
    """
    Starting from a valid coloring, try to increase the number of distinct colors
    by changing edges to new colors, maintaining validity.
    """
    cur_colors = colors[:]
    cur_distinct = count_distinct_colors(cur_colors)
    best_colors = cur_colors[:]
    best_distinct = cur_distinct
    
    # Precompute edge-to-cycles mapping
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(CYCLES):
        for edge in cycle:
            edge_to_cycles[edge].append(ci)
    
    # Track which colors are in use
    color_count = [0] * MAX_COLOR
    for c in cur_colors:
        color_count[c] += 1
    
    # Try to assign new colors to edges
    # Order edges by some heuristic (e.g., high degree in cycle graph)
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for edge in edge_order:
            if time.time() > deadline:
                break
            old_color = cur_colors[edge]
            
            # Try each unused or less-used color
            # First try colors not currently used (to increase distinct count)
            for new_color in range(MAX_COLOR):
                if new_color == old_color:
                    continue
                if color_count[new_color] > 0:
                    # This color is already used, so distinct count won't increase
                    # But it might allow other edges to take new colors later
                    # For now, focus on unused colors
                    continue
                
                # Check if changing to new_color maintains validity
                cur_colors[edge] = new_color
                valid = True
                for ci in edge_to_cycles[edge]:
                    cycle = CYCLES[ci]
                    c0 = cur_colors[cycle[0]]
                    c1 = cur_colors[cycle[1]]
                    c2 = cur_colors[cycle[2]]
                    c3 = cur_colors[cycle[3]]
                    if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                        valid = False
                        break
                
                if valid:
                    # Accept the change
                    color_count[old_color] -= 1
                    color_count[new_color] += 1
                    cur_distinct += 1
                    improved = True
                    
                    # Update best
                    if cur_distinct > best_distinct:
                        best_distinct = cur_distinct
                        best_colors = cur_colors[:]
                    break
                else:
                    # Revert
                    cur_colors[edge] = old_color
            
            # Also try colors that are used, in case they help later
            # But this doesn't increase distinct count directly, so skip for now
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Start with the known valid coloring: color(i,j) = i
    # This uses 15 colors (0..14)
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = i
            idx += 1
    
    # Verify it's valid (it should be)
    if not is_valid(colors):
        # Fallback: all zeros
        return [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_distinct = count_distinct_colors(colors)
    
    # Try greedy improvement
    improved_colors, improved_distinct = greedy_maximize_colors(colors, deadline)
    if improved_distinct > best_distinct:
        best_colors = improved_colors
        best_distinct = improved_distinct
    
    # Now try simulated annealing to find even better solutions
    # Start from the best coloring found
    cur_colors = best_colors[:]
    cur_distinct = count_distinct_colors(cur_colors)
    
    # Precompute edge-to-cycles
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(CYCLES):
        for edge in cycle:
            edge_to_cycles[edge].append(ci)
    
    # Track color counts
    color_count = [0] * MAX_COLOR
    for c in cur_colors:
        color_count[c] += 1
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 10000
    
    while T > T_min and time.time() < deadline:
        # Pick a random edge
        edge = random.randrange(NUM_EDGES)
        old_color = cur_colors[edge]
        
        # Try a new color: prefer unused colors
        if random.random() < 0.7:
            # Pick an unused color
            unused = [c for c in range(MAX_COLOR) if color_count[c] == 0]
            if not unused:
                continue
            new_color = random.choice(unused)
        else:
            new_color = random.randrange(MAX_COLOR)
        
        if new_color == old_color:
            continue
        
        # Check validity
        cur_colors[edge] = new_color
        valid = True
        for ci in edge_to_cycles[edge]:
            cycle = CYCLES[ci]
            c0 = cur_colors[cycle[0]]
            c1 = cur_colors[cycle[1]]
            c2 = cur_colors[cycle[2]]
            c3 = cur_colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                valid = False
                break
        
        if not valid:
            cur_colors[edge] = old_color
            continue
        
        # Compute new distinct count
        if color_count[old_color] == 1:
            # Old color is being removed
            new_distinct = cur_distinct - 1 + (1 if color_count[new_color] == 0 else 0)
        else:
            new_distinct = cur_distinct + (1 if color_count[new_color] == 0 else 0)
        
        # Accept/reject
        if new_distinct >= cur_distinct:
            accept = True
        else:
            delta = new_distinct - cur_distinct
            accept_prob = math.exp(delta / T)
            accept = random.random() < accept_prob
        
        if accept:
            color_count[old_color] -= 1
            color_count[new_color] += 1
            cur_distinct = new_distinct
            
            if cur_distinct > best_distinct:
                best_distinct = cur_distinct
                best_colors = cur_colors[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve and T < 1.0:
                break
        else:
            cur_colors[edge] = old_color
        
        T *= cooling
    
    return best_colors
