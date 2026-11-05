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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Initialize with 0
    colors = [0] * NUM_EDGES
    color_counts = [0] * MAX_COLOR
    for c in colors:
        color_counts[c] += 1
        
    used_colors = set(c for c in range(MAX_COLOR) if color_counts[c] > 0)
    best_score = len(used_colors)
    best_colors = colors[:]
    
    # Simulated annealing parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute list of all edges for swap moves
    all_edges = list(range(NUM_EDGES))
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Occasionally restart temperature to escape deep local optima
        if step % 5000 == 0 and T < 0.1:
            T = 0.5
        
        # Choose move type: recolor (70%) or swap (30%)
        if random.random() < 0.7:
            # Recolor move
            e_idx = random.randint(0, NUM_EDGES - 1)
            
            # Decide which color to try
            # Bias towards unused colors to increase the count
            if random.random() < 0.6:
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
            if not is_valid_change(colors, e_idx, new_color):
                continue
            
            # Compute score delta
            old_color = colors[e_idx]
            old_score = len(used_colors)
            
            # Apply change
            colors[e_idx] = new_color
            
            # Update counts
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                used_colors.discard(old_color)
                
            color_counts[new_color] += 1
            if color_counts[new_color] == 1:
                used_colors.add(new_color)
                
            new_score = len(used_colors)
            delta_score = new_score - old_score
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta_score >= 0 or random.random() < math.exp(delta_score / T):
                # Keep the change
                if new_score > best_score:
                    best_score = new_score
                    best_colors = colors[:]
            else:
                # Revert
                colors[e_idx] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                if color_counts[new_color] == 0:
                    used_colors.discard(new_color)
                if color_counts[old_color] == 1:
                    used_colors.add(old_color)
            
        else:
            # Swap move: swap colors of two different edges
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            if c1 == c2:
                continue
            
            # Check if swapping maintains validity
            # We need to check all cycles containing e1 or e2
            valid = True
            for cycle in cycles_containing_edge[e1]:
                c_a, c_b, c_c, c_d = cycle
                # Build color set with swapped colors
                col_set = set()
                for c in (c_a, c_b, c_c, c_d):
                    if c == e1:
                        col_set.add(c2)
                    elif c == e2:
                        col_set.add(c1)
                    else:
                        col_set.add(colors[c])
                if len(col_set) == 4:
                    valid = False
                    break
            
            if valid:
                for cycle in cycles_containing_edge[e2]:
                    c_a, c_b, c_c, c_d = cycle
                    col_set = set()
                    for c in (c_a, c_b, c_c, c_d):
                        if c == e1:
                            col_set.add(c2)
                        elif c == e2:
                            col_set.add(c1)
                        else:
                            col_set.add(colors[c])
                    if len(col_set) == 4:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Compute score delta
            old_score = len(used_colors)
            
            # Apply swap
            colors[e1] = c2
            colors[e2] = c1
            
            # Update counts
            color_counts[c1] -= 1
            if color_counts[c1] == 0:
                used_colors.discard(c1)
            color_counts[c2] -= 1
            if color_counts[c2] == 0:
                used_colors.discard(c2)
            
            color_counts[c2] += 1
            if color_counts[c2] == 1:
                used_colors.add(c2)
            color_counts[c1] += 1
            if color_counts[c1] == 1:
                used_colors.add(c1)
            
            new_score = len(used_colors)
            delta_score = new_score - old_score
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta_score >= 0 or random.random() < math.exp(delta_score / T):
                # Keep the swap
                if new_score > best_score:
                    best_score = new_score
                    best_colors = colors[:]
            else:
                # Revert
                colors[e1] = c1
                colors[e2] = c2
                color_counts[c1] -= 1
                if color_counts[c1] == 0:
                    used_colors.discard(c1)
                color_counts[c2] -= 1
                if color_counts[c2] == 0:
                    used_colors.discard(c2)
                
                color_counts[c1] += 1
                if color_counts[c1] == 1:
                    used_colors.add(c1)
                color_counts[c2] += 1
                if color_counts[c2] == 1:
                    used_colors.add(c2)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_colors
