import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices
# edge_index[i][j] for i < j
edge_indices = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_indices[(i, j)] = idx
        idx += 1

# Precompute cycles containing each edge
# A 4-cycle is a set of 4 edges.
# For a given edge (u, v), the other two vertices x, y are chosen from the remaining 14.
# The cycles are (u, v, x, y, u) and (u, v, y, x, u).
# Edges in (u, v, x, y, u): (u,v), (v,x), (x,y), (y,u)
# Edges in (u, v, y, x, u): (u,v), (v,y), (y,x), (x,u)

cycles_containing_edge = [[] for _ in range(NUM_EDGES)]

for u in range(N):
    for v in range(u + 1, N):
        e_idx = edge_indices[(u, v)]
        # Find all x, y != u, v, x < y
        for x in range(N):
            if x == u or x == v:
                continue
            for y in range(x + 1, N):
                if y == u or y == v:
                    continue
                # Cycle 1: u-v-x-y-u
                e1 = edge_indices[(u, v)]
                e2 = edge_indices[(v, x) if v < x else (x, v)]
                e3 = edge_indices[(x, y)]
                e4 = edge_indices[(y, u) if y < u else (u, y)]
                cycles_containing_edge[e_idx].append((e1, e2, e3, e4))
                
                # Cycle 2: u-v-y-x-u
                e1 = edge_indices[(u, v)]
                e2 = edge_indices[(v, y) if v < y else (y, v)]
                e3 = edge_indices[(y, x) if y < x else (x, y)] # same as (x,y)
                e4 = edge_indices[(x, u) if x < u else (u, x)]
                cycles_containing_edge[e_idx].append((e1, e2, e3, e4))

def is_valid_change(colors, e_idx, new_color):
    """
    Check if changing colors[e_idx] to new_color maintains validity.
    Only cycles containing e_idx need to be checked.
    """
    for cycle in cycles_containing_edge[e_idx]:
        c1, c2, c3, c4 = cycle
        # One of these is e_idx
        # Get the other 3 colors
        col_set = {new_color}
        for c in (c1, c2, c3, c4):
            if c != e_idx:
                col_set.add(colors[c])
        if len(col_set) == 4:
            return False
    return True

def is_valid_swap(colors, e1_idx, e2_idx):
    """
    Check if swapping colors of e1_idx and e2_idx maintains validity.
    Only cycles containing either e1 or e2 need to be checked.
    """
    c1 = colors[e1_idx]
    c2 = colors[e2_idx]
    if c1 == c2:
        return True
    # Check cycles containing e1_idx
    for cycle in cycles_containing_edge[e1_idx]:
        col_set = set()
        for c in cycle:
            if c == e1_idx:
                col_set.add(c2)
            elif c == e2_idx:
                col_set.add(c1)
            else:
                col_set.add(colors[c])
        if len(col_set) == 4:
            return False
    # Check cycles containing e2_idx
    for cycle in cycles_containing_edge[e2_idx]:
        col_set = set()
        for c in cycle:
            if c == e1_idx:
                col_set.add(c2)
            elif c == e2_idx:
                col_set.add(c1)
            else:
                col_set.add(colors[c])
        if len(col_set) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Initialize with 0
    colors = [0] * NUM_EDGES
    color_counts = [0] * MAX_COLOR
    for c in colors:
        color_counts[c] += 1
        
    used_colors = set(c for c in range(MAX_COLOR) if color_counts[c] > 0)
    best_score = len(used_colors)
    best_colors = colors[:]
    
    # Simulated annealing parameters
    temperature = 10.0
    cooling_rate = 0.9995
    current_score = len(used_colors)
    
    # Local search with simulated annealing
    # We want to maximize the number of distinct colors.
    # Strategy: Try to introduce new colors, allow temporary worsening.
    
    while time.time() < deadline:
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        
        # Decide which color to try
        # Bias towards unused colors to increase the count
        if random.random() < 0.7:
            # Pick an unused color
            unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
            if not unused:
                new_color = random.randint(0, MAX_COLOR - 1)
            else:
                new_color = random.choice(unused)
        else:
            # Pick any color
            new_color = random.randint(0, MAX_COLOR - 1)
            
        # Skip if no change
        if colors[e_idx] == new_color:
            continue
            
        # Check validity
        if is_valid_change(colors, e_idx, new_color):
            # Apply change
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            
            # Update counts
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                used_colors.discard(old_color)
                
            color_counts[new_color] += 1
            if color_counts[new_color] == 1:
                used_colors.add(new_color)
                
            current_score = len(used_colors)
            if current_score > best_score:
                best_score = current_score
                best_colors = colors[:]
        else:
            # Try swap move
            if random.random() < 0.3:
                e2_idx = random.randint(0, NUM_EDGES - 1)
                if e2_idx != e_idx and colors[e_idx] != colors[e2_idx]:
                    if is_valid_swap(colors, e_idx, e2_idx):
                        # Apply swap
                        colors[e_idx], colors[e2_idx] = colors[e2_idx], colors[e_idx]
                        
                        # Swap doesn't change color counts, so used_colors stays the same
                        pass
        
        # Cooling
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    # Final polish: try to add more colors with greedy approach
    for _ in range(10000):
        if time.time() >= deadline:
            break
        e_idx = random.randint(0, NUM_EDGES - 1)
        unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
        if not unused:
            break
        new_color = random.choice(unused)
        if colors[e_idx] == new_color:
            continue
        if is_valid_change(colors, e_idx, new_color):
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                used_colors.discard(old_color)
            color_counts[new_color] += 1
            if color_counts[new_color] == 1:
                used_colors.add(new_color)
            current_score = len(used_colors)
            if current_score > best_score:
                best_score = current_score
                best_colors = colors[:]
    
    return best_colors
